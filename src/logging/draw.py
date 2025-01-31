import matplotlib.pyplot as plt
import numpy as np


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
