from typing import Optional, Callable
import h5py
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from . import preprocessing
import torch
import albumentations as A


class Galaxies(Dataset):
    def __init__(self,
                 path: str,
                 gray: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 denoise: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 augment: Optional[A.Compose] = None,
                 normalize: Optional[A.Compose] = None
                 ) -> None:
        """Classe que representa o dataset de imagens de galáxias.

        Args:
            path (``str``): Caminho para o arquivo base do dataset.
            gray (``Optional[Callable]``, optional): Função de conversão para
            cinza.
            denoise (``Optional[Callable]``, optional): Função para aplicação
            da remoção de ruído.
        """
        self.__path = path
        self.__gray = gray
        self.__denoise = denoise
        self.__augment = augment
        self.__normalize = normalize

        try:
            with h5py.File(self.__path, 'r') as f:
                self.__imgs = f['images'][:]
                self.__labels = f['labels'][:]
        except FileNotFoundError:
            raise FileNotFoundError('Missing dataset')

    def __len__(self) -> int:
        return len(self.__imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self):
            raise IndexError('Index out of range')

        img = self.__imgs[idx]
        label = self.__labels[idx]

        if self.__gray:
            img = self.__gray(img)

        if self.__denoise:
            img = self.__denoise(img)

        if self.__augment:
            img = self.__augment(image=img)['image']

        if self.__normalize:
            img = self.__normalize(image=img)['image']

        return img, label


class GalaxiesDataLoader:
    def __init__(self,
                 path: str,
                 batch_size: int,
                 size: float = 1.0,
                 as_gray: bool = True,
                 denoise: bool = False,
                 augment: bool = False
                 ) -> None:
        """Classe responsável por carregar e dividir o dataset de galáxias em
        conjuntos de treino, validação e teste.

        Args:
            path (``str``): Caminho para o arquivo base do dataset.
            batch_size (``int``): Tamanho do lote (batch) para o DataLoader.
            as_gray (``bool``): Se True, converte as imagens para escala de
            cinza.
            denoise (``bool``): Se True, aplica remoção de ruído.
            augment (``bool``): Se True, aplica aumento de dados.
            size (``float``): Proporção do dataset a ser carregada.
            Defaults to 1.0.

        """
        self.__path = path
        self.__batch_size = batch_size
        self.__size = size

        self.__gray = as_gray
        self.__denoise = denoise
        self.__augment = augment

        self.__gray_converter = preprocessing.Make_gray()
        self.__denoiser = preprocessing.Denoiser()
        self.__augmenter = preprocessing.Augmenter()

        self.is_gray = True if as_gray else False
        self.is_denoised = True if denoise else False
        self.is_augmented = True if augment else False
        self.is_full = True if size == 1.0 else False

    def get_dataloader(self):
        gray = self.__gray_converter.luma if self.__gray else None
        denoise = self.__denoiser.max if self.__denoise else None
        augment_pipe = self.__augmenter.augment() if self.__augment else None
        # normalize_pipe = self.__augmenter.normalize()

        dataset = Galaxies(path=self.__path,
                           gray=gray,
                           denoise=denoise,
                           augment=augment_pipe,
                           normalize=None)
        if not self.is_full:
            total_len = len(dataset)
            subset_len = int(total_len * self.__size)

            np.random.seed(0)
            indices = np.random.choice(total_len, subset_len, replace=False)
            dataset = Subset(dataset, indices)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.__batch_size,
                                shuffle=True)
        return dataloader
