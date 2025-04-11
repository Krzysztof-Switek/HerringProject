import torch
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer: str):
        """
        Inicjalizacja Grad-CAM dla określonej warstwy modelu

        Args:
            model: Wytrenowany model
            target_layer: Nazwa warstwy docelowej (np. "layer4[-1].conv3")
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Rejestracja hooków
        self._register_hooks()

    def _register_hooks(self):
        """Rejestracja hooków dla przechwytywania aktywacji i gradientów"""
        target_layer = self._find_target_layer()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Zmieniamy backward hook na pełny backward hook
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Zmiana: użycie register_full_backward_hook zamiast register_backward_hook
        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(backward_hook)

    def clear_hooks(self):
        """Jawne usuwanie hooków"""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()
        self.activations = None
        self.gradients = None

    def _find_target_layer(self):
        """Znajdowanie warstwy docelowej na podstawie nazwy"""
        module = self.model
        try:
            for layer in self.target_layer.split('.'):
                if '[' in layer:
                    layer_name = layer[:layer.find('[')]
                    index = int(layer[layer.find('[') + 1:layer.find(']')])
                    module = getattr(module, layer_name)[index]
                else:
                    module = getattr(module, layer)
            return module
        except AttributeError:
            raise ValueError(f"Could not find target layer: {self.target_layer}")

    def generate(self, input_tensor, class_idx: int):
        """
        Generowanie heatmapy Grad-CAM

        Args:
            input_tensor: Tensor wejściowy (1, C, H, W)
            class_idx: Indeks klasy docelowej

        Returns:
            heatmap: Normalizowana mapa aktywacji (H, W)
        """
        # 1. Forward pass
        output = self.model(input_tensor)
        if class_idx >= output.size(1):
            raise ValueError(f"Class index {class_idx} out of range")

        # 2. Zerowanie gradientów i backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 3. Obliczanie ważonych aktywacji
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

        return heatmap

    def __del__(self):
        """Usuwanie hooków przy niszczeniu obiektu"""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()
