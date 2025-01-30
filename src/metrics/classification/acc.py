from src.metrics.classification.cm import ConfusionMatrix


class Accuracy:
    def __init__(self, cm: ConfusionMatrix) -> None:
        """
        Classe para calcular a acurácia (Accuracy).
        """
        self.__cm = cm
        self.__accuracy = self._compute_accuracy()

    def _compute_accuracy(self) -> float:
        """
        Calcula a acurácia a partir da matriz de confusão.
        Fórmula: (TP + TN) / (TP + FP + FN + TN)
        """
        metrics = self.__cm.get_full_matrix()
        tp = metrics["tp"]
        tn = metrics["tn"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        total = tp + tn + fp + fn
        return (tp + tn) / total if total > 0 else 0.0

    def get_accuracy(self) -> float:
        """
        Retorna a acurácia calculada.
        """
        return self.__accuracy
