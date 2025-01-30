from src.metrics.classification.cm import ConfusionMatrix


class Precision:
    def __init__(self, cm: ConfusionMatrix) -> None:
        """
        Classe para calcular a precisão (Precision).
        """
        self.__cm = cm
        self.__precision = self._compute_precision()

    def _compute_precision(self) -> float:
        """
        Calcula a precisão a partir da matriz de confusão.
        Fórmula: TP / (TP + FP)
        """
        metrics = self.__cm.get_full_matrix()
        tp, fp = metrics["tp"], metrics["fp"]
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def get_precision(self) -> float:
        """
        Retorna a precisão calculada.
        """
        return self.__precision
