import torch
import numpy as np
import cv2
from abc import ABC, abstractmethod

def _extract_logits(output):
    """
    Zwraca tensor logitów klasyfikacyjnych o kształcie [N, C].
    - Jeśli model zwraca (logits, coś) -> weź logits.
    - Jeśli dostajemy [C] -> zrób [1, C].
    """
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Model forward must return a Tensor or (Tensor, ...), got: {type(output)}")
    if output.ndim == 1:
        output = output.unsqueeze(0)
    return output


class BaseHeatmapGenerator(ABC):
    """Abstrakcyjna klasa bazowa dla metod generujących heatmapy."""

    @abstractmethod
    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        pass

    @abstractmethod
    def clear_hooks(self):
        pass

    @abstractmethod
    def verify_hooks(self) -> bool:
        pass

    @abstractmethod
    def get_hook_status(self) -> str:
        pass


class GradCAM(BaseHeatmapGenerator):
    """Implementacja metody Grad-CAM."""

    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = None
        self.backward_handle = None
        self._register_hooks()

    def _find_target_layer(self) -> torch.nn.Module:
        module = self.model
        try:
            for part in self.target_layer.split('.'):
                if '[' in part:
                    name = part.split('[')[0]
                    idx = int(part[part.find('[') + 1:part.find(']')])
                    module = getattr(module, name)[idx]
                else:
                    module = getattr(module, part)
            return module
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Layer search error: {str(e)}")

    def _register_hooks(self):
        target_module = self._find_target_layer()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            return None

        self.clear_hooks()  # Usuń istniejące hooki
        self.forward_handle = target_module.register_forward_hook(forward_hook)
        self.backward_handle = target_module.register_full_backward_hook(backward_hook)

    def verify_hooks(self) -> bool:
        return (self.forward_handle is not None and self.backward_handle is not None)

    def get_hook_status(self) -> str:
        status = f"GradCAM | Layer: {self.target_layer} | "
        status += f"Forward: {'active' if self.forward_handle else 'inactive'}, "
        status += f"Backward: {'active' if self.backward_handle else 'inactive'}"
        return status

    def clear_hooks(self):
        if self.forward_handle:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle:
            self.backward_handle.remove()
            self.backward_handle = None
        self.activations = None
        self.gradients = None

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        if not self.verify_hooks():
            raise RuntimeError("Hooks are not active")

        # forward -> logits [N, C]
        output_raw = self.model(input_tensor)
        output = _extract_logits(output_raw)

        if class_idx < 0 or class_idx >= output.shape[1]:
            raise ValueError(f"Invalid class index: {class_idx} for output shape {tuple(output.shape)}")

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # weights: [C,1,1] po średniej po H,W; activations: [N,C,H,W] (hook)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [N,1,H,W]
        cam = torch.relu(cam.squeeze()).cpu().numpy()  # [H,W]
        cam = cv2.normalize(cam, None, 0, 1, cv2.NORM_MINMAX)
        return cam

    def __del__(self):
        self.clear_hooks()


def create_heatmap_plot(
        model: torch.nn.Module,
        target_layer: str,
        image_tensor: torch.Tensor,
        original_image,
        true_label: int,
        pred_label: int,
        confidence: float,
        output_path: str,
        cfg_visualization: dict
):
    """
    Generates and saves a plot containing the original image and multiple heatmaps.
    """
    import matplotlib.pyplot as plt

    # Ensure model is in evaluation mode
    model.eval()

    methods = {
        "gradcam": GradCAM(model, target_layer),
        "gradcam++": GradCAMPP(model, target_layer),
        "guided_backprop": GuidedBackprop(model)
    }

    # The class index for CAM methods is 0 for class '1' and 1 for class '2'
    pred_class_idx = pred_label - 1

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(np.array(original_image))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, (name, method) in enumerate(methods.items(), 1):
        try:
            # Re-register hooks to ensure they are clean
            method._register_hooks()
            heatmap = method.generate(image_tensor, pred_class_idx)

            if name == "guided_backprop":
                axes[i].imshow(heatmap, cmap='gray')
            else:
                heatmap = cv2.resize(heatmap, original_image.size)
                axes[i].imshow(original_image)
                axes[i].imshow(heatmap, cmap=cfg_visualization['colormap'], alpha=cfg_visualization['alpha'])

            axes[i].set_title(name.upper())
        except Exception as e:
            print(f"⚠️ Heatmap generation for '{name}' failed: {e}")
            # Create a black image as a placeholder on error
            placeholder = np.zeros_like(np.array(original_image))
            axes[i].imshow(placeholder)
            axes[i].set_title(f"{name.upper()} (Error)")
        finally:
            axes[i].axis('off')
            # Explicitly clear hooks to release resources
            method.clear_hooks()

    fig.suptitle(
        f"True: {true_label} | Pred: {pred_label} | Confidence: {confidence:.2f}%",
        fontsize=16,
        fontweight="bold"
    )
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


class GradCAMPP(GradCAM):
    """Rozszerzenie Grad-CAM++."""

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        if not self.verify_hooks():
            raise RuntimeError("Hooks are not active")

        output_raw = self.model(input_tensor)
        output = _extract_logits(output_raw)

        if class_idx < 0 or class_idx >= output.shape[1]:
            raise ValueError(f"Invalid class index: {class_idx} for output shape {tuple(output.shape)}")

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients[0].cpu().numpy()  # [C,H,W]
        acts = self.activations[0].cpu().numpy()  # [C,H,W]

        # klasyczny wariant wag dla CAM++
        weights = np.mean(np.maximum(grads, 0.0) ** 2, axis=(1, 2), keepdims=True)  # [C,1,1]
        cam = np.sum(acts * weights, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)
        cam = cv2.normalize(cam, None, 0, 1, cv2.NORM_MINMAX)
        return cam


class GuidedBackprop(BaseHeatmapGenerator):
    """Implementacja Guided Backpropagation."""

    def __init__(self, model: torch.nn.Module):
        self.model = model.eval()
        self.handles = []
        self._disable_inplace_relu()
        self._register_hooks()

        if not self.verify_hooks():
            raise RuntimeError("Failed to register all hooks")

    def _disable_inplace_relu(self):
        """Wyłącza inplace w modułach ReLU."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False

    def _register_hooks(self):
        """Rejestruje hooki dla ReLU."""

        def relu_backward_hook(module, grad_input, grad_output):
            grad = grad_input[0].clone()
            grad[grad < 0] = 0
            return (grad,)

        self.clear_hooks()

        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                handle = module.register_full_backward_hook(relu_backward_hook)
                self.handles.append(handle)

    def verify_hooks(self) -> bool:
        """Sprawdza czy liczba hooków zgadza się z liczbą ReLU."""
        relu_count = sum(1 for m in self.model.modules() if isinstance(m, torch.nn.ReLU))
        return len(self.handles) == relu_count

    def get_hook_status(self) -> str:
        relu_count = sum(1 for m in self.model.modules() if isinstance(m, torch.nn.ReLU))
        return f"GuidedBackprop | ReLU hooks: {len(self.handles)}/{relu_count}"

    def clear_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        if not self.verify_hooks():
            raise RuntimeError("Not all hooks are active")

        x = input_tensor.clone().requires_grad_(True)
        output_raw = self.model(x)
        output = _extract_logits(output_raw)  # [N,C]

        if class_idx < 0 or class_idx >= output.shape[1]:
            raise ValueError(f"Invalid class index: {class_idx} for output shape {tuple(output.shape)}")

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        if x.grad is None:
            raise RuntimeError("No gradients - hooks failed")

        saliency = x.grad[0].cpu().numpy()  # [3,H,W]
        saliency = np.max(np.abs(saliency), axis=0)  # [H,W]
        saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
        saliency = np.clip(saliency, 0, 1)
        saliency = np.log1p(saliency)
        saliency = saliency / (np.max(saliency) + 1e-12)
        return saliency

    def __del__(self):
        self.clear_hooks()
