import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    @staticmethod
    def denormalize(tensor):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = tensor.numpy().transpose((1, 2, 0))
        image = std * image + mean
        return np.clip(image, 0, 1)

    @staticmethod
    def plot_gradcam(image, heatmap, pred_class, true_class=None):
        image = Visualizer.denormalize(image)
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(np.uint8(255 * image), 0.6, heatmap, 0.4, 0)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image)
        ax[0].set_title(f"Original\n{true_class if true_class else ''}")
        ax[1].imshow(heatmap)
        ax[1].set_title("Grad-CAM")
        ax[2].imshow(superimposed)
        ax[2].set_title(f"Predicted: {pred_class}")
        plt.show()