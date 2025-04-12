import torch
import numpy as np
import cv2
from abc import ABC, abstractmethod


class BaseHeatmapGenerator(ABC):
    """Abstrakcyjna klasa bazowa dla metod generujących heatmapy."""

    @abstractmethod
    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generuje heatmapę dla określonego indeksu klasy."""
        pass

    @abstractmethod
    def clear_hooks(self):
        """Czyści wszystkie hooki przypisane do modelu."""
        pass


class GradCAM(BaseHeatmapGenerator):
    """Implementacja klasycznej metody Grad-CAM."""

    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _find_target_layer(self) -> torch.nn.Module:
        """Wyszukuje warstwę docelową w modelu po nazwie."""
        module = self.model
        try:
            for part in self.target_layer.split('.'):
                if '[' in part:
                    name = part.split('[')[0]
                    idx = int(part[part.find('[') + 1: part.find(']')])
                    module = getattr(module, name)[idx]
                else:
                    module = getattr(module, part)
            return module
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Nie znaleziono warstwy '{self.target_layer}': {e}")

    def _register_hooks(self):
        """Rejestruje hooki forward i backward na wybranej warstwie."""
        target_module = self._find_target_layer()

        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = target_module.register_forward_hook(forward_hook)
        self.backward_handle = target_module.register_full_backward_hook(backward_hook)

    def clear_hooks(self):
        """Usuwa hooki z modelu i czyści bufory."""
        for handle in [self.forward_handle, self.backward_handle]:
            if handle:
                handle.remove()
        self.activations = None
        self.gradients = None

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generuje klasyczną mapę Grad-CAM."""
        output = self.model(input_tensor)

        if class_idx >= output.shape[1]:
            raise ValueError(f"Nieprawidłowy indeks klasy: {class_idx}")

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Obliczenie wag i mapy
        weights = torch.mean(self.gradients, dim=(2, 3))
        cam = torch.sum(weights[:, :, None, None] * self.activations, dim=1)[0]
        cam = torch.relu(cam).cpu().numpy()
        cam = cv2.normalize(cam, None, 0, 1, cv2.NORM_MINMAX)

        return cam

    def __del__(self):
        self.clear_hooks()


class GradCAMPP(GradCAM):
    """Rozszerzenie metody Grad-CAM o Grad-CAM++."""

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        output = self.model(input_tensor)

        if class_idx >= output.shape[1]:
            raise ValueError(f"Nieprawidłowy indeks klasy: {class_idx}")

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients[0].cpu().numpy()
        acts = self.activations[0].cpu().numpy()

        # Wagi uwzględniające kwadratowanie gradientów
        weights = np.mean(np.maximum(grads, 0) ** 2, axis=(1, 2), keepdims=True)
        cam = np.sum(acts * weights, axis=0)
        cam = np.maximum(cam, 0)
        cam = cv2.normalize(cam, None, 0, 1, cv2.NORM_MINMAX)

        return cam


class GuidedBackprop(BaseHeatmapGenerator):
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles = []
        self._replace_relu(self.model)  # wymuszenie ReLU(inplace=False)
        self._register_hooks()

    def _replace_relu(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.ReLU):
                setattr(module, name, torch.nn.ReLU(inplace=False))
            else:
                self._replace_relu(child)

    def _register_hooks(self):
        def relu_backward_hook(module, grad_input, grad_output):
            # Sklonuj, by uniknąć modyfikacji widoku (view)
            grad_input = grad_input[0].clone()
            grad_input[grad_input < 0] = 0
            return (grad_input,)

        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                handle = module.register_full_backward_hook(relu_backward_hook)
                self.handles.append(handle)

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        if class_idx >= output.shape[1]:
            raise ValueError(f"Class index {class_idx} out of range")

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        saliency = input_tensor.grad[0].cpu().numpy()
        saliency = np.max(np.abs(saliency), axis=0)
        saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)

        return saliency

    def clear_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __del__(self):
        self.clear_hooks()
