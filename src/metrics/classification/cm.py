import torch


class ConfusionMatrix:
    def __init__(self,
                 y_true: torch.Tensor,
                 y_pred: torch.Tensor
                 ) -> None:

        self.__yt = y_true
        self.__yp = y_pred
        self._components = self._compute_metrics()

    def get_full_matrix(self) -> dict[str, int]:
        """
        Retorna todas as métricas calculadas.
        """
        return self._components

    def get_component(self, metric: str) -> int:
        """
        Retorna um componente específico (TP, FP, FN, TN).
        """
        return self._components.get(metric, 0)

    def _compute_metrics(self) -> dict[str, int]:
        conditions = {
            "tp": (self.__yt == 1) & (self.__yp == 1),
            "fp": (self.__yt == 0) & (self.__yp == 1),
            "fn": (self.__yt == 1) & (self.__yp == 0),
            "tn": (self.__yt == 0) & (self.__yp == 0),
        }
        return {
            key: int(torch.sum(condition).item())
            for key, condition in conditions.items()
            }
