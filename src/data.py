from typing import Optional, Callable
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np
from . import preprocessing
import torch


class Galaxies(Dataset):
    def __init__(self,
                 path: str,
                 size: float = 1.0,
                 split: Optional[str] = None,
                 gray: Optional[Callable] = None,
                 denoise: Optional[Callable] = None,
                 augment: Optional[Callable] = None,
                 normalize: Optional[Callable] = None
                 ) -> None:
        """
        Inicializa a classe Galaxies com os parâmetros especificados.

        Args:
            path (str): Caminho para o arquivo HDF5
            contendo o conjunto de dados.
            size (float, optional): Proporção do conjunto de dados a ser usada.
                Defaults to 1.0.
            split (Optional[str], optional): Divisão do conjunto de dados
                ('train', 'val', 'test'). Defaults to None.
            gray (Optional[Callable], optional): Função para
            converter as imagens
                para escala de cinza. Defaults to None.
            denoise (Optional[Callable], optional): Função para aplicar
                desnoisificação nas imagens. Defaults to None.
            augment (Optional[Callable], optional): Função
            para aplicar aumentos
                nas imagens. Defaults to None.
            normalize (Optional[Callable], optional): Função para normalizar
                as imagens e rótulos. Defaults to None.

        Raises:
            FileNotFoundError: Se o arquivo do conjunto de dados especificado
                não for encontrado.
            ValueError: Se o valor de split for inválido.
        """
        self.__path = path
        self.__size = size
        self.__split = split.lower() if split is not None else None
        self.__gray = gray
        self.__denoise = denoise
        self.__augment = augment
        self.__normalize = normalize

        try:
            with h5py.File(self.__path, 'r') as f:
                total_samples = len(f['images'])
                num_samples = int(total_samples * self.__size)

                if self.__split:
                    split_map = {"train": 0, "val": 1, "test": 2}
                    if split not in split_map:
                        raise ValueError(f"Invalid split: {split}")

                    split_filter = split_map[split]
                    split_indices = np.where(
                        f['split'][:num_samples] == split_filter
                    )[0]

                    self.__imgs = f['images'][split_indices]
                    self.__labels = f['labels'][split_indices]
                else:
                    self.__imgs = f['images'][:num_samples]
                    self.__labels = f['labels'][:num_samples]

        except FileNotFoundError:
            raise FileNotFoundError('Missing dataset')

    def __len__(self) -> int:
        return len(self.__imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.__imgs[idx]
        label = self.__labels[idx]

        if self.__gray:
            img = self.__gray(img)

        if self.__denoise:
            img = self.__denoise(img)

        if self.__augment and self.__split == 'train':
            img = self.__augment(img)

        if self.__normalize:
            img = self.__normalize(img)
            label = self.__normalize(label)

        return img, label


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
            gray=preprocessing.luma if self.__as_gray else None,
            denoise=self.__denoise,
            augment=preprocessing.augment() if self.__augment else None,
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
