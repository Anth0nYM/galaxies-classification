import torch
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)


class Metrics:
    def __init__(self,
                 device: torch.device
                 ) -> None:
        """
        Inicializa a classe Metrics com o dispositivo especificado.

        Args:
            device (torch.device): O dispositivo (CPU ou GPU) onde as métricas
            serão calculadas.
        """
        self.__device = device
        self.funcs = {
            "accuracy": BinaryAccuracy().to(device=self.__device),
            "precision": BinaryPrecision().to(device=self.__device),
            "recall": BinaryRecall().to(device=self.__device),
            "f1": BinaryF1Score().to(device=self.__device),
        }

    def report(self,
               yt: torch.Tensor,
               yp: torch.Tensor,
               threshold: float = 0.5
               ) -> dict[str, float]:
        """
        Calcula e retorna as métricas de avaliação.

        Args:
            yt (``torch.Tensor``): Rótulos verdadeiros (binários).
            yp (``torch.Tensor``): Rótulos previstos (binários).

        Returns:
            dict: Dicionário com os nomes das métricas e seus valores.
        """
        reports = {}
        yp_bin = (yp > threshold).int()
        for metric_name, metric_func in self.funcs.items():
            reports[metric_name] = metric_func(yp_bin, yt).item()

        return reports
