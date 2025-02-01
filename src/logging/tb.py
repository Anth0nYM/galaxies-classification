from src.logging.git import Git
from torch.utils.tensorboard import SummaryWriter
from src.logging.draw import Drawer


class TbLog:
    def __init__(self,
                 comment: str,
                 log_dir: str = "logs/",
                 ) -> None:
        """Classe responsável por logar os resultados de um experimento no TensorBoard.

        Args:
            comment (str): Comentário a ser adicionado ao nome do diretório de logs.
            log_dir (str, optional): Caminho base para armazenar os logs. 
                                     Padrão é "logs/".
        """
        self.__git = Git(repo_path=".")
        self.__log_dir = f"{log_dir}{self.__git._get_hex()}{comment}\{self.__git._get_timestamp()}"

        self.__writer = SummaryWriter(log_dir=self.__log_dir)
        self.__drawer = Drawer()

    def log_text(self,
                 text: str
                 ) -> None:
        """
        Registra um texto no TensorBoard.

        Args:
            text (str): Texto a ser registrado.
        """
        self.__writer.add_text("Detalhes do experimento", text)

    def log_cm(self,
               cm: dict[str, int],
               epoch: int
               ) -> None:
        """
        Registra uma matriz de confusão no TensorBoard.

        Args:
            cm (dict[str, int]): Matriz de confusão a ser registrada.
            epoch (int): Época em que a matriz de confusão foi gerada.
        """
        cm_fig = self.__drawer._draw_cm(cm)
        self.__writer.add_figure(
            tag="Matriz de confusão",
            figure=cm_fig,
            global_step=epoch
        )

    def log_metircs(self,
                    split: str,
                    epoch: int,
                    loss: float,
                    metrics: dict[str, float]
                    ) -> None:
        """
        Registra as métricas de classificação e a perda no TensorBoard.

        Args:
            split (str): Nome do conjunto de dados (ex: "train", "val", "test").
            epoch (int): Número da época em que os dados foram registrados.
            loss (float): Valor da perda a ser registrado.
            metrics (dict[str, float]): Dicionário contendo métricas de classificação.
        """
        self.__log_loss(loss_split=split, loss_epoch=epoch, loss_value=loss)
        self.__log_classificatrion_metrics(
            metrics_split=split, metrics_epoch=epoch, metrics_dict=metrics
            )
    
    def close(self
              ) -> None:
        """
        Fecha o escritor do TensorBoard.
        """       
        self.__writer.close()

    def __log_loss(self,
                   loss_split: str,
                   loss_epoch: int,
                   loss_value: float
                   ) -> None:
        """
        Registra o valor da perda no TensorBoard.

        Args:
            loss_split (str): Nome do conjunto de dados (ex: "train", "val", "test").
            loss_epoch (int): Época em que a perda foi registrada.
            loss_value (float): Valor da perda.
        """
        self.__writer.add_scalar(f"{loss_split}/loss", loss_value, loss_epoch)

    def __log_classificatrion_metrics(self,
                                      metrics_split: str,
                                      metrics_epoch: int,
                                      metrics_dict: dict[str, float]
                                      ) -> None:
        """
        Registra métricas de classificação no TensorBoard.

        Args:
            metrics_split (str): Nome do conjunto de dados (ex: "train", "val", "test").
            metrics_epoch (int): Época em que as métricas foram registradas.
            metrics_dict (dict[str, float]): Dicionário contendo métricas de classificação.
        """
        for name, value in metrics_dict.items():
            self.__writer.add_scalar(
                f"{metrics_split}/{name}", value, metrics_epoch
                )
