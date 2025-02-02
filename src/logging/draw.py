import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from PIL import ImageDraw
import torchvision.transforms as T
import math


class Drawer:
    def __init__(self):
        pass

    def _draw_cm(self, cm: dict[str, int]) -> plt.Figure:
        """Usa o matplotlib para desenhar uma matriz de confusão.

        Args:
            cm (dict[str, int]): Confusion matrix.

        Returns:
            plt.Figure: Figura da matriz de confusão.
        """
        matrix = np.array([[cm["tp"], cm["fp"]], [cm["fn"], cm["tn"]]])
        labels = [["TP", "FP"], ["FN", "TN"]]

        fig, ax = plt.subplots()
        cax = ax.matshow(matrix, cmap="Blues")
        plt.colorbar(cax)

        for (i, j), val in np.ndenumerate(matrix):
            ax.text(
                j, i, f"{labels[i][j]}\n{val}",
                ha='center',
                va='center',
                fontsize=12,
                color="black"
            )

        ax.set_xlabel("Predição")
        ax.set_ylabel("Real")

        return fig

    def _draw_grid(self,
                   samples: list
                   ) -> torch.Tensor:

        if not samples:
            return torch.empty(0)

        label_mapping = {0.0: "Round Smooth", 1.0: "Barred Spiral"}

        to_pil = T.ToPILImage()
        to_tensor = T.ToTensor()
        img_tensors = []

        for img, true_label, pred_label in samples:
            pil_img = to_pil(img)
            draw = ImageDraw.Draw(pil_img)

            true_text = label_mapping.get(true_label, str(true_label))
            pred_text = label_mapping.get(pred_label, str(pred_label))
            text = f"True: {true_text} | Pred: {pred_text}"

            draw.text((10, 10), text, fill="red")
            img_tensor = to_tensor(pil_img)
            img_tensors.append(img_tensor)

        # fazendo um grid quadrado
        n_images = len(img_tensors)
        nrow = math.ceil(math.sqrt(n_images))
        grid = make_grid(img_tensors, nrow=nrow)
        return grid
