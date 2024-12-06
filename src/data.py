from random import shuffle
from typing import Optional, Callable
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
# import albumentations as A


class Galaxies:
    def __init__(self,
                 path: str,
                 transform: Optional[Callable],
                 gray: Optional[Callable] = None,
                 classes: list[int] = [2, 5]
                 ) -> None:
        """Classe que representa o dataset de imagens de galáxias.

        Args:
            path (str): Caminho para o arquivo base do dataset
            transform (Optional[Callable]): Função que aplica transformações
            gray (Optional[Callable], optional): Função de conversão para cinza
        """
        self.__path = path
        self.__transform = transform
        self.__gray = gray
        self.__classes = classes
        self.__saved_path = "data/Binary_2_5_dataset.h5"
        self.__saved = False

        try:
            with h5py.File(self.__saved_path, 'r') as f:
                self.__imgs = f['images'][:]
                self.__labels = f['labels'][:]
                self.__saved = True
        except FileNotFoundError:
            self.__saved = False

        if not self.__saved:
            with h5py.File(self.__path, 'r') as f:
                self.imgs = f['images'][:]
                self.labels = f['ans'][:]
                all_labels = f['ans']
                selected_indices = np.where(
                    (all_labels[:] == self.__classes[0]) |
                    (all_labels[:] == self.__classes[1])
                )[0]

                self.__imgs = f['images'][selected_indices]

                # Reencodar os rótulos: 2 -> 0 e 5 -> 1
                self.__labels = np.where(
                    all_labels[selected_indices] == 2, 0, 1
                )

            with h5py.File(self.__saved_path, 'w') as f:
                f.create_dataset('images', data=self.__imgs)
                f.create_dataset('labels', data=self.__labels)

    def __len__(self) -> int:
        """Retorna o tamanho do dataset.

        Returns:
            int: Tamanho (em quantidade de imagens) do dataset.
        """
        return len(self.__imgs)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retorna uma tupla contendo a imagem e o rótulo no índice fornecido.
        """
        if idx >= len(self):
            raise IndexError('Index out of range')

        img = self.__imgs[idx]
        label = self.__labels[idx]

        if self.__gray:
            img = self.__gray(img)

        if self.__transform:
            img = self.__transform(img)

        return img, label


class GalaxiesDataLoader:
    def __init__(self,
                 path: str,
                 batch_size: int,
                 subset_size: Optional[int],
                 as_gray: int,
                 augment: bool,
                 img_size: tuple[int, int] = (256, 256),
                 seed: int = 0
                 ) -> None:

        self.__path = path
        self.__batch_size = batch_size
        self.__subset_size = subset_size
        self.__as_gray = as_gray
        self.__augment = augment
        self.__img_size = img_size
        self.__seed = seed

    def __gray(self) -> Callable:
        ''' Converte a imagem para escala de cinza usando a função luma.

        Returns:
            Callable: Função que converte a imagem para escala de cinza.
        '''
        return lambda img: (
            0.2126 * img[..., 0] +
            0.7152 * img[..., 1] +
            0.0722 * img[..., 2]
        ).astype(np.uint8)

    def __compose(self,
                  split: str
                  ) -> Callable:
        ''' Aplica transformações e aumentos nas imagens.

        Args:
            split (str): Qual conjunto de dados está sendo transformado.

        Returns:
            Callable: Função que aplica as transformações e aumentos.
        '''
        return

    def split(self,
              sizes: tuple[int, int, int]
              ) -> tuple[Galaxies, Galaxies, Galaxies]:
        """Realiza a divisão do dataset em conjuntos de treino,
        validação e teste

        Args:
            sizes (tuple[int, int, int]): Percentagem (arredondada) do tamanho
            de cada subconjunto em relação ao original.

        Returns:
            tuple[DataSet, DataSet, DataSet]: Conjuntos de treino,
            validação e teste.
        """
        return