from typing import Optional, Callable
import h5py
import numpy as np
# import albumentations as A


class DataSet:
    def __init__(self,
                 path: str,
                 transform: Optional[Callable],
                 gray: Optional[Callable] = None,
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

        with h5py.File(self.__path, 'r') as f:
            self.imgs = f['images'][:]
            self.labels = f['ans'][:]
            all_labels = f['ans']
            # Apenas as classes 2 e 5 são consideradas
            selected_indices = np.where(
                (all_labels[:] == 2) | (all_labels[:] == 5)
            )[0]

            self.__imgs = f['images'][selected_indices]

            # Reencodar os rótulos: 2 -> 0 e 5 -> 1
            self.__labels = np.where(all_labels[selected_indices] == 2, 0, 1)

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
        if idx < 0 or idx >= len(self):
            raise IndexError('Index out of range')

        img = self.__imgs[idx]
        label = self.__labels[idx]

        if self.__gray:
            img = self.__gray(img)

        if self.__transform:
            img = self.__transform(img)

        return img, label


class DataLoader:
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

    def __compose(self,
                  split: str
                  ) -> Callable:
        ''' Aplica transformações e aumentos nas imagens.

        Args:
            split (str): Qual conjunto de dados está sendo transformado.

        Returns:
            Callable: Função que aplica as transformações e aumentos.
        '''
        return max

    def __get_dataloader(self,
                         split_and_size: dict[str, int],
                         seed: int = 0
                         ) -> DataSet:
        """Obtém o dataset para cada subconjunto e seu tamanho.

        Args:
            split_and_size (dict[str, int]): Nome de cada conjunto
            e seu tamanho.

        Returns:
            DataSet: O DataLoader criado.
        """
        split = list(split_and_size.keys())[0]
        print(split)
        return DataSet(path=self.__path,
                       transform=None,
                       gray=None)

    def split(self,
              sizes: tuple[int, int, int]
              ) -> tuple[DataSet, DataSet, DataSet]:
        """Realiza a divisão do dataset em conjuntos de treino,
        validação e teste

        Args:
            sizes (tuple[int, int, int]): Percentagem (arredondada) do tamanho
            de cada subconjunto em relação ao original.

        Returns:
            tuple[DataSet, DataSet, DataSet]: Conjuntos de treino,
            validação e teste.
        """
        assert len(sizes) == 3
        assert 100 == sum(sizes)
        train = self.__get_dataloader({'train': sizes[0]}, seed=self.__seed)
        val = self.__get_dataloader({'val': sizes[1]}, seed=self.__seed)
        test = self.__get_dataloader({'test': sizes[2]}, seed=self.__seed)
        return train, val, test


if __name__ == '__main__':
    dataset = DataSet(
        path="Galaxy10_DECals.h5",
        transform=None,
        gray=None)

    print(f"Tamanho do dataset: {len(dataset)}")

    img, label = dataset[0]
    print("Imagem shape:", img.shape)
    print("Label:", label)
