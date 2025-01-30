from src.metrics.classification.cm import ConfusionMatrix


class Recall:
    def __init__(self, cm: ConfusionMatrix) -> None:
        """
        Classe para calcular o recall (Sensibilidade).
        """
        self.__cm = cm
        self.__recall = self._compute_recall()

    def _compute_recall(self) -> float:
        """
        Calcula o recall a partir da matriz de confusão.
        Fórmula: TP / (TP + FN)
        """
        metrics = self.__cm.get_full_matrix()
        tp, fn = metrics["tp"], metrics["fn"]
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def get_recall(self) -> float:
        """
        Retorna o recall calculado.
        """
        return self.__recall
