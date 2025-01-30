from typing import Optional, Callable
import h5py
from torch.utils.data import Dataset
import numpy as np
import torch
from src.data.pp import Normalize


class Galaxies(Dataset):
    def __init__(self,
                 path: str,
                 size: float = 1.0,
                 split: Optional[str] = None,
                 gray: Optional[Callable] = None,
                 denoise: Optional[Callable] = None,
                 augment: Optional[Callable] = None,
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

        self.__is_gray = bool(self.__gray)

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

        norm = Normalize(is_gray=self.__is_gray)
        img, label = norm(img, label)
        return img, label
