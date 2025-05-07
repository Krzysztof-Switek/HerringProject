import torch
import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Optional, List


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

        output = self.model(input_tensor)

        if class_idx >= output.shape[1]:
            raise ValueError(f"Invalid class index: {class_idx}")

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam.squeeze()).cpu().numpy()
        cam = cv2.normalize(cam, None, 0, 1, cv2.NORM_MINMAX)

        return cam

    def __del__(self):
        self.clear_hooks()


class GradCAMPP(GradCAM):
    """Rozszerzenie Grad-CAM++."""

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        if not self.verify_hooks():
            raise RuntimeError("Hooks are not active")

        output = self.model(input_tensor)

        if class_idx >= output.shape[1]:
            raise ValueError(f"Invalid class index: {class_idx}")

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients[0].cpu().numpy()
        acts = self.activations[0].cpu().numpy()

        weights = np.mean(np.maximum(grads, 0) ** 2, axis=(1, 2), keepdims=True)
        cam = np.sum(acts * weights, axis=0)
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

        input_tensor = input_tensor.clone().requires_grad_(True)
        output = self.model(input_tensor)

        if class_idx >= output.shape[1]:
            raise ValueError(f"Invalid class index: {class_idx}")

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        if input_tensor.grad is None:
            raise RuntimeError("No gradients - hooks failed")

        saliency = input_tensor.grad[0].cpu().numpy()
        saliency = np.max(np.abs(saliency), axis=0)

        # Normalizacja saliency do zakresu [0, 1]
        saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
        saliency = np.clip(saliency, 0, 1)  # Upewnienie się, że wartości są w zakresie [0, 1]

        # Zastosowanie logarytmicznej normalizacji
        saliency = np.log1p(saliency)  # Logarytmiczne uwydatnienie różnic
        saliency = saliency / np.max(saliency)  # Normalizacja do zakresu [0, 1]

        # Tworzymy obraz z 3 kategoriami kolorów
        saliency_colored = np.zeros((*saliency.shape, 3))  # Nowy obraz w przestrzeni RGB

        for i in range(saliency.shape[0]):
            for j in range(saliency.shape[1]):
                if saliency[i, j] == 0:
                    saliency_colored[i, j] = [0, 0, 0]  # Kategoria 1 - brak aktywności (czarne)
                elif saliency[i, j] <= 0.5:
                    saliency_colored[i, j] = [1, 1, 0]  # Kategoria 2 - aktywność do 50% (żółte)
                else:
                    saliency_colored[i, j] = [1, 0, 0]  # Kategoria 3 - aktywność powyżej 50% (czerwone)

        return saliency_colored

    def __del__(self):
        self.clear_hooks()
