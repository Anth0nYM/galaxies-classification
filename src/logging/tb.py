from src.logging.git import Git
from torch.utils.tensorboard import SummaryWriter
from src.logging.draw import Drawer


class TbLog:
    def __init__(self,
                 comment: str,
                 log_dir: str = "logs/",
                 ) -> None:

        self.__git = Git(repo_path=".")
        self.__log_dir = (
            f"{log_dir}\
                {self.__git._get_hex()}\
                    _{comment}_\
                        {self.__git._get_timestamp()}"
        )

        self.__writer = SummaryWriter(log_dir=self.__log_dir)
        self.__drawer = Drawer()

    def log_cm(self, cm: dict[str, int]) -> None:
        """
        Loga a matriz de confusão no tensorboard.
        """
        fig = self.__drawer._draw_cm(cm)
        self.__writer.add_figure(
            tag="Confusion Matrix",
            figure=fig,
            global_step=0
        )
