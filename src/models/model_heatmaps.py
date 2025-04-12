import torch
import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Optional, List


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

    @abstractmethod
    def verify_hooks(self) -> bool:
        """Weryfikuje czy hooki są prawidłowo zarejestrowane."""
        pass


class GradCAM(BaseHeatmapGenerator):
    """Implementacja metody Grad-CAM z rozszerzoną weryfikacją hooków."""

    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.forward_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.backward_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._register_hooks()

        if not self.verify_hooks():
            raise RuntimeError("Hooki nie zostały poprawnie zarejestrowane")

    def _find_target_layer(self) -> torch.nn.Module:
        """Wyszukuje i weryfikuje warstwę docelową."""
        module = self.model
        try:
            for part in self.target_layer.split('.'):
                if '[' in part:
                    name = part.split('[')[0]
                    idx = int(part[part.find('[') + 1:part.find(']')])
                    module = getattr(module, name)[idx]
                else:
                    module = getattr(module, part)
            if not isinstance(module, torch.nn.Module):
                raise TypeError(f"Znaleziony obiekt nie jest modułem: {type(module)}")
            return module
        except (AttributeError, IndexError, TypeError) as e:
            raise ValueError(f"Błąd wyszukiwania warstwy '{self.target_layer}': {str(e)}")

    def _register_hooks(self):
        """Rejestruje hooki z dodatkowym loggingiem."""
        target_module = self._find_target_layer()

        def forward_hook(module, input, output):
            self.activations = output.detach().clone()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach().clone()
            return None

        # Usuń istniejące hooki przed rejestracją nowych
        self.clear_hooks()

        self.forward_handle = target_module.register_forward_hook(forward_hook)
        self.backward_handle = target_module.register_full_backward_hook(backward_hook)

    def verify_hooks(self) -> bool:
        """Sprawdza czy hooki są aktywne."""
        return (self.forward_handle is not None and
                self.backward_handle is not None and
                not self.forward_handle.id in self._forward_hooks() and
                not self.backward_handle.id in self._backward_hooks())

    def _forward_hooks(self):
        """Pobiera ID aktywnych forward hooków."""
        return [h.id for h in self.model._forward_hooks.values()]

    def _backward_hooks(self):
        """Pobiera ID aktywnych backward hooków."""
        return [h.id for h in self.model._backward_hooks.values()]

    def clear_hooks(self):
        """Bezpiecznie usuwa hooki."""
        for handle in [self.forward_handle, self.backward_handle]:
            if handle is not None and handle.id in self._forward_hooks() + self._backward_hooks():
                handle.remove()
        self.activations = None
        self.gradients = None

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generuje heatmapę z walidacją danych."""
        if not self.verify_hooks():
            raise RuntimeError("Hooki nie są aktywne - nie można generować heatmapy")

        output = self.model(input_tensor)

        if class_idx >= output.shape[1]:
            raise ValueError(f"Nieprawidłowy indeks klasy: {class_idx}")

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1

        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Brak danych z hooków - sprawdź rejestrację")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam.squeeze()).cpu().numpy()
        cam = cv2.normalize(cam, None, 0, 1, cv2.NORM_MINMAX)

        return cam

    def __del__(self):
        self.clear_hooks()


class GradCAMPP(GradCAM):
    """Rozszerzenie Grad-CAM++ z dodatkową walidacją."""

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        if not self.verify_hooks():
            raise RuntimeError("Hooki nie są aktywne")

        output = self.model(input_tensor)

        if class_idx >= output.shape[1]:
            raise ValueError(f"Nieprawidłowy indeks klasy: {class_idx}")

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Brak danych z hooków")

        grads = self.gradients[0].cpu().numpy()
        acts = self.activations[0].cpu().numpy()

        weights = np.mean(np.maximum(grads, 0) ** 2, axis=(1, 2), keepdims=True)
        cam = np.sum(acts * weights, axis=0)
        cam = np.maximum(cam, 0)
        cam = cv2.normalize(cam, None, 0, 1, cv2.NORM_MINMAX)

        return cam


class GuidedBackprop(BaseHeatmapGenerator):
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles = []

        # Najpierw zmień wszystkie ReLU na inplace=False
        self._disable_inplace_relu()

        # Następnie zarejestruj hooki
        self._register_hooks()

        # Weryfikacja powinna teraz przejść
        if not self.verify_hooks():
            raise RuntimeError("Hooki ReLU nie zostały poprawnie zarejestrowane")

    def _disable_inplace_relu(self):
        """Rekurencyjnie wyłącza inplace w wszystkich modułach ReLU"""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.ReLU) and module.inplace:
                # Tworzymy nowy moduł ReLU z inplace=False
                new_relu = torch.nn.ReLU(inplace=False)

                # Znajdź rodzicowski moduł i nazwę atrybutu
                parent = self._get_parent_module(name)
                child_name = name.split('.')[-1]

                # Zamień moduł
                setattr(parent, child_name, new_relu)

    def _get_parent_module(self, full_name: str) -> torch.nn.Module:
        """Pomocnicza funkcja do znalezienia modułu rodzicielskiego"""
        names = full_name.split('.')
        current = self.model
        for name in names[:-1]:  # Pomijamy ostatnią nazwę (bieżący moduł)
            current = getattr(current, name)
        return current

    def _register_hooks(self):
        """Rejestruje hooki dla wszystkich modułów ReLU"""

        def relu_backward_hook(module, grad_input, grad_output):
            # Pobierz gradient wejściowy i ustaw wartości ujemne na 0
            grad = grad_input[0].clone()
            grad[grad < 0] = 0
            return (grad,)

        # Wyczyść istniejące hooki
        self.clear_hooks()

        # Zarejestruj nowe hooki dla wszystkich ReLU
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                handle = module.register_full_backward_hook(relu_backward_hook)
                self.handles.append(handle)

    def verify_hooks(self) -> bool:
        """Sprawdza czy liczba hooków odpowiada liczbie modułów ReLU"""
        # Policz wszystkie moduły ReLU w modelu
        relu_count = sum(1 for _ in self.model.modules() if isinstance(_, torch.nn.ReLU))

        # Sprawdź czy liczba zarejestrowanych hooków się zgadza
        return len(self.handles) == relu_count

    def clear_hooks(self):
        """Usuwa wszystkie zarejestrowane hooki"""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generuje mapę aktywacji z walidacją"""
        if not self.verify_hooks():
            raise RuntimeError("Nie wszystkie moduły ReLU mają zarejestrowane hooki")

        # Przygotuj tensor wejściowy
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        if class_idx >= output.shape[1]:
            raise ValueError(f"Nieprawidłowy indeks klasy: {class_idx}")

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        # Sprawdź czy gradienty zostały obliczone
        if input_tensor.grad is None:
            raise RuntimeError("Brak gradientów - hooki mogły nie zadziałać")

        # Przygotuj mapę saliency
        saliency = input_tensor.grad[0].cpu().numpy()
        saliency = np.max(np.abs(saliency), axis=0)
        saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)

        return saliency

    def __del__(self):
        self.clear_hooks()