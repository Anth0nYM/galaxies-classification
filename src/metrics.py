from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import numpy as np


class Metrics:
    def __init__(self) -> None:
        self.__metrics = {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
        }

    def report(self,
               yt: np.ndarray,
               yp: np.ndarray
               ) -> dict[str, float]:
        """
        Calcula e retorna as métricas de avaliação.

        Args:
            yt (``np.ndarray``): Rótulos verdadeiros.
            yp (``np.ndarray``): Rótulos previstos.

        Returns:
            dict: Dicionário com os nomes das métricas e seus valores.
        """
        reports = {}
        for metric_name, metric_func in self.__metrics.items():
            reports[metric_name] = metric_func(yt, yp)

        return reports
