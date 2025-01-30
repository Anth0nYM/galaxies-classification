class F1Score:
    def __init__(self,
                 precision: float,
                 recall: float
                 ) -> None:
        """
        Classe para calcular o F1-score.
        """
        self.__precision = precision
        self.__recall = recall
        self.__f1_score = self._compute_f1()

    def _compute_f1(self) -> float:
        """
        Calcula o F1-score.
        Fórmula: (2 * Precisão * Recall) / (Precisão + Recall)
        """
        if (self.__precision + self.__recall) == 0:
            return 0.0

        return (2 * self.__precision * self.__recall) / (
            self.__precision + self.__recall
        )

    def get_f1_score(self) -> float:
        """
        Retorna o F1-score calculado.
        """
        return self.__f1_score
