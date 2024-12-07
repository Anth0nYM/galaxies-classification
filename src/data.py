from typing import Optional, Callable
import h5py
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import src.gray as gray


class Galaxies(Dataset):
    def __init__(self,
                 path: str,
                 gray: Optional[Callable] = None,
                 denoise: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 ) -> None:
        """Classe que representa o dataset de imagens de galáxias.

        Args:
            path (str): Caminho para o arquivo base do dataset
            transform (Optional[Callable]): Função que aplica transformações
            gray (Optional[Callable], optional): Função de conversão para cinza
        """
        self.__path = path
        self.__gray = gray
        self.__denoise = denoise
        self.__transform = transform

        try:
            with h5py.File(self.__path, 'r') as f:
                self.__imgs = f['images'][:]
                self.__labels = f['labels'][:]
        except FileNotFoundError:
            raise FileNotFoundError('Missing dataset')

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

        if self.__denoise:
            img = self.__denoise(img)

        if self.__transform:
            img = self.__transform(img)

        return img, label


class GalaxiesDataLoader:
    def __init__(self,
                 path: str,
                 batch_size: int,
                 as_gray: bool,
                 augment: bool,
                 denoise: Optional[Callable] = None,
                 img_size: tuple[int, int] = (256, 256),
                 seed: int = 0
                 ) -> None:

        self.__path = path
        self.__batch_size = batch_size
        self.__gray = gray.luma() if as_gray else None
        self.__denoise = denoise
        self.__augment = augment
        self.__img_size = img_size
        self.__seed = seed

    def is_gray(self) -> bool:
        return self.__gray is not None

    # TODO: Compose

    def __compose(self) -> None:
        ''' Aplica transformações e aumentos nas imagens.

        Args:
            split (str): Qual conjunto de dados está sendo transformado.

        Returns:
            Callable: Função que aplica as transformações e aumentos.
        '''
        return None

    def split(self,
              sizes: tuple[int, int, int]
              ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Realiza a divisão do dataset em conjuntos de treino, validação e teste.

        Args:
            sizes (tuple[int, int, int]): Percentagem do tamanho de cada
            subconjunto (treinamento, validação, teste).

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]: Conjuntos de treino,
            validação e teste.
        """
        if sum(sizes) != 100:
            raise ValueError('A soma das porcentagens deve ser 100')

        dataset = Galaxies(path=self.__path,
                           gray=self.__gray,
                           denoise=self.__denoise,
                           transform=None)

        n = len(dataset)
        train_size = int(n * sizes[0] / 100)
        val_size = int(n * sizes[1] / 100)
        test_size = n - train_size - val_size

        generator = torch.Generator()
        generator.manual_seed(self.__seed)

        train_dataset, val_dataset, test_dataset = random_split(
            dataset=dataset,
            lengths=[train_size, val_size, test_size],
            generator=generator)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.__batch_size,
            shuffle=True)

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.__batch_size,
            shuffle=False)

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.__batch_size,
            shuffle=False)

        return train_loader, val_loader, test_loader
