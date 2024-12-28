from typing import Optional, Callable
import h5py
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
from . import preprocessing
import torch


class Galaxies(Dataset):
    def __init__(self,
                 path: str,
                 gray: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 size: float = 1.0,
                 ) -> None:
        """Um Dataset customizado do PyTorch para lidar
        com conjuntos de dados de imagens
        de galáxias armazenados no formato HDF5.

        Args:
            path (str): Caminho para o arquivo HDF5
            contendo o conjunto de dados.

            gray (Optional[Callable[[np.ndarray], np.ndarray]]):
            Função para converter imagenspara escala de cinza.
            Defaults to None.

            size (float, optional): Proporção do conjunto de dados a ser usada.
            Defaults to 1.0.

        Raises:
            FileNotFoundError: Se o arquivo do conjunto de dados especificado
            não for encontrado.
        """
        self.__path = path
        self.__gray = gray
        self.__size = size

        try:
            with h5py.File(self.__path, 'r') as f:
                total_samples = len(f['images'])
                num_samples = int(total_samples * self.__size)

                self.__imgs = f['images'][:num_samples]
                self.__labels = f['labels'][:num_samples]

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

        return img, label


class GalaxiesDataLoader:
    def __init__(self,
                 path: str,
                 batch_size: int,
                 size: float = 1.0,
                 seed: int = 0,
                 as_gray: bool = True,
                 ) -> None:
        """    Classe para gerenciar o carregamento de dados
        do conjunto de dados Galaxies.

        Args:
            path (str): path (str): Caminho para o arquivo HDF5
            contendo o conjunto de dados.

            batch_size (int): Tamanho do lote (batch)

            size (float, optional): Fração do conjunto de dados a ser usada.
            Defaults to 1.0.

            seed (int, optional): Semente para reprodução dos splits de dados.
            Defaults to 0.

            as_gray (bool, optional): Define se as imagens serão convertidas
            para escala de cinza. Defaults to True.
        """
        self.__path = path
        self.__batch_size = batch_size
        self.__size = size
        self.__seed = seed
        self.__gray = as_gray
        self.__gray_converter = preprocessing.Make_gray()
        self.is_gray = True if as_gray else False
        self.is_full = True if size == 1.0 else False

    def __split_dataset(self,
                        dataset: Galaxies,
                        split_size: tuple[float, float, float]
                        ) -> tuple[Subset, Subset, Subset]:
        """Divide o conjunto de dados em
        treino, validação e teste.

        Args:
            dataset (Galaxies): Conjunto de dados a ser dividido.
            split_size (tuple[float, float, float]): Proporção de divisão.

        Returns:
            tuple[Subset, Subset, Subset]: Os 3 conjuntos criados.
        """
        total_size = len(dataset)
        train_size = int(total_size * split_size[0])
        val_size = int(total_size * split_size[1])
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(self.__seed)

        train_dataset, val_dataset, test_dataset = random_split(
            dataset=dataset,
            lengths=[train_size, val_size, test_size],
            generator=generator)

        return train_dataset, val_dataset, test_dataset

    def split(self,
              sizes: tuple[float, float, float] = (0.8, 0.1, 0.1)
              ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Divide o dataloader em treino, validação e teste.

        Args:
            split_size (tuple[float, float, float]): Proporção para cada split.
            Defaults to (0.8, 0.1, 0.1).

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]: Dataloaders criados.
        """
        gray = self.__gray_converter.luma if self.__gray else None
        dataset = Galaxies(path=self.__path, gray=gray, size=self.__size)

        train, val, test = self.__split_dataset(
            dataset=dataset, split_size=sizes)

        train_loader = DataLoader(dataset=train, batch_size=self.__batch_size)
        val_loader = DataLoader(dataset=val, batch_size=self.__batch_size)
        test_loader = DataLoader(dataset=test, batch_size=self.__batch_size)

        return train_loader, val_loader, test_loader
