import src.metrics.classification as c
import torch


# TODO clf report sklearn implments here
class ClassificationReport:
    def __init__(self,
                 yt: torch.Tensor,
                 yp: torch.Tensor
                 ) -> None:
        """
        Relatório para tarefas de classificação.
        """
        self.__cm = c.ConfusionMatrix(y_true=yt, y_pred=yp)
        self.__compute_metrics()

    def get_report(self) -> dict[str, float]:
        """
        Retorna um dicionário com todas as métricas calculadas.
        """
        return {
            "accuracy": self.__accuracy,
            "precision": self.__precision,
            "recall": self.__recall,
            "f1_score": self.__f1_score,
        }

    def get_cm(self) -> dict[str, int]:
        """
        Retorna a matriz de confusão calculada.
        """
        return self.__cm.get_full_matrix()

    def __compute_metrics(self) -> None:
        """
        Calcula todas as métricas apenas uma vez e as armazena como atributos.
        """
        self.__accuracy = c.Accuracy(self.__cm).get_accuracy()
        self.__precision = c.Precision(self.__cm).get_precision()
        self.__recall = c.Recall(self.__cm).get_recall()
        self.__f1_score = c.F1Score(precision=self.__precision,
                                    recall=self.__recall).get_f1_score()
