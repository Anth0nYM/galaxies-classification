from typing import Optional


class EsMonitor:
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0
                 ) -> None:
        """
        Monitor de Early Stopping.

        Args:
            patience (``int``): Número de épocas sem melhora
            para parar o treinamento.
            min_delta (``float``): Valor mínimo de diferença
            para considerar uma melhora.
        """
        self.__patience = patience
        self.__min_delta = min_delta
        self.wait = 0
        self.__best_loss: Optional[float] = None
        self.__stop_training = False

    def __call__(self, loss: float) -> int:
        """
        Avalia se o treinamento deve continuar com base no loss atual.

        Args:
            loss (``float``): Loss da época atual.

        Returns:
            ``int``: Número de épocas consecutivas sem melhora.
        """
        if self.__best_loss is None:
            self.__best_loss = loss
        elif loss > self.__best_loss - self.__min_delta:
            self.wait += 1
            if self.wait >= self.__patience:
                self.__stop_training = True
        else:
            self.__best_loss = loss
            self.wait = 0

        return self.wait

    def must_stop(self) -> bool:
        """
        Retorna se o treinamento deve parar.

        Returns:
            bool: True se o treinamento deve parar, False caso contrário.
        """
        return self.__stop_training
