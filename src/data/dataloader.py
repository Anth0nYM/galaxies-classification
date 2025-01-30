from torch.utils.data import DataLoader
from typing import Optional, Callable
from .dataset import Galaxies
from src.data.pp import GrayConverter, Augment


class GalaxiesDataLoader:
    def __init__(self,
                 path: str,
                 batch_size: int,
                 as_gray: bool,
                 denoise: Optional[Callable] = None,
                 augment: bool = False,
                 size: float = 1.0,
                 ) -> None:
        """
        Inicializa a classe GalaxiesDataLoader com os parâmetros especificados.

        Args:
            path (str): Caminho para o arquivo do conjunto de dados.
            batch_size (int): Tamanho do lote (batch) para o DataLoader.
            as_gray (bool): Se True, converte as imagens para escala de cinza.
            denoise (Optional[Callable], optional): Função para aplicar
                desnoisificação nas imagens. Defaults to None.
            augment (bool, optional): Se True, aplica aumentos nas imagens.
                Defaults to False.
            size (float, optional): Proporção do conjunto de dados a ser usada.
                Defaults to 1.0.
        """
        self.__path = path
        self.__batch_size = batch_size
        self.__as_gray = as_gray
        self.__denoise = denoise
        self.__augment = augment
        self.__size = size

        self.is_denoised = True if denoise is not None else False
        self.is_gray = True if as_gray else False
        self.is_augmented = True if augment else False
        self.is_full = True if size == 1.0 else False

        self.__g_conv = GrayConverter().luma if self.__as_gray else None

    def get_loader(self,
                   split: Optional[str] = None,
                   shuffle: bool = True
                   ) -> DataLoader:
        """
        Gera um DataLoader para um conjunto de dados específico.

        Args:
            split (Optional[str], optional): Divisão do conjunto de dados
            ('train', 'val', 'test'). Defaults to None.
            shuffle (bool, optional): Se o DataLoader deve embaralhar os dados.
            Defaults to True.

        Returns:
            DataLoader: DataLoader configurado para a divisão especificada.
        """

        dataset = Galaxies(
            path=self.__path,
            size=self.__size,
            gray=self.__g_conv if self.__as_gray else None,
            denoise=self.__denoise,
            augment=Augment() if self.__augment else None,
            split=split
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.__batch_size,
            shuffle=shuffle
        )

    def split(self):
        """
        Retorna DataLoaders para treino, validação e teste.

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]:
            DataLoaders para as divisões.
        """
        train_loader = self.get_loader(split='train', shuffle=True)
        val_loader = self.get_loader(split='val', shuffle=False)
        test_loader = self.get_loader(split='test', shuffle=False)

        return train_loader, val_loader, test_loader
