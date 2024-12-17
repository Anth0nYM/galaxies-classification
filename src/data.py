from typing import Optional, Callable
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np
from . import preprocessing
import torch


class Galaxies(Dataset):
    def __init__(self,
                 path: str,
                 gray: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 denoise: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 augment: Optional[Callable[[np.ndarray], np.ndarray]] = None
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
            img = self.__augment(img)

        img = img.astype(np.float32) / 255.0  # Normaliza para [0, 1]

        # Converte a imagem e o rótulo para torch.Tensor
        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor


class GalaxiesDataLoader:
    def __init__(self,
                 path: str,
                 batch_size: int,
                 as_gray: bool = True,
                 denoise: bool = False,
                 augment: bool = False,
                 seed: int = 0
                 ) -> None:
        """Classe responsável por carregar e dividir o dataset de galáxias em
        conjuntos de treino, validação e teste.

        Args:
            path (``str``): Caminho para o arquivo base do dataset.
            batch_size (``int``): Tamanho do lote (batch) para o DataLoader.
            as_gray (``bool``): Se True, converte as imagens para escala de
            cinza.
            augment (``bool``): Se True, aplica aumento de dados.
            denoise (``bool``): Se True, aplica remoção de ruído.
            img_size (``tuple[int, int]``, optional): Tamanho das imagens.
                Defaults to (``256``, ``256``).
            seed (``int``, optional): Semente para geração de números
            aleatórios. Defaults to ``0``.
        """
        self.__path = path
        self.__batch_size = batch_size

        self.__gray = as_gray
        self.__denoise = denoise
        self.__augment = augment
        self.__seed = seed

        self.__gray_converter = preprocessing.Make_gray()
        self.__denoiser = preprocessing.Denoiser()

        self.is_gray = True if as_gray else False
        self.is_denoised = True if denoise else False
        self.is_augmented = True if augment else False

    def get_dataloader(self):
        gray = self.__gray_converter.luma if self.__gray else None
        denoise = self.__denoiser.median if self.__denoise else None
        augment_pipeline = preprocessing.augment()

        dataset = Galaxies(path=self.__path,
                           gray=gray,
                           denoise=denoise,
                           augment=augment_pipeline)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.__batch_size,
                                shuffle=True)
        return dataloader
