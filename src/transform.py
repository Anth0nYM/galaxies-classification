import albumentations as A
import numpy as np


class Transformer:
    def __init__(self):
        pass
        """Classe responsável por aplicar
        transformações nas imagens do dataset.
        """
    def augment(self, img: np.ndarray) -> np.ndarray:
        """Aplica o aumento de dados nas imagens.

        Args:
            img (´´np.ndarray´´): Imagem em array numpy.

        Returns:
            ´´np.ndarray´´: Imagem com aumento de dados.
        """
        augmentation_pipeline = A.Compose([
            A.ElasticTransform(alpha=500, sigma=1000 * 0.05, p=1.0),
        ])
        return augmentation_pipeline(image=img)['image']
