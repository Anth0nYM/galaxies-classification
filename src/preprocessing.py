import cv2
import numpy as np
from scipy import ndimage
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Make_gray:
    def __init__(self):
        pass

    def luma(self, img: np.ndarray) -> np.ndarray:
        """Aplica a transformação Luma.

        Args:
            img (np.ndarray): imagem colorida.
        Returns:
            np.ndarray: imagem em escala de cinza.
        """
        img_gray = (
            0.2126 * img[..., 0] +
            0.7152 * img[..., 1] +
            0.0722 * img[..., 2]
        ).astype(np.uint8)

        return img_gray


class Denoiser:
    def __init__(self, kernel_size=3) -> None:
        self.__kernel_size = kernel_size

    def mean(self, image: np.ndarray) -> np.ndarray:
        """Filtro de média."""
        return cv2.blur(image, (self.__kernel_size, self.__kernel_size))

    def gaussian(self, image: np.ndarray, sigma=0) -> np.ndarray:
        """Filtro Gaussiano."""
        return cv2.GaussianBlur(
            image, (self.__kernel_size, self.__kernel_size), sigma
        )

    def laplace(
        self, image: np.ndarray, ddepth=cv2.CV_64F
    ) -> np.ndarray:
        """Filtro Laplaciano."""
        return cv2.Laplacian(image, ddepth)

    def sobel_x(
        self, image: np.ndarray, ddepth=cv2.CV_64F
    ) -> np.ndarray:
        """Filtro Sobel na direção X."""
        return cv2.Sobel(
            image, ddepth, 1, 0, ksize=self.__kernel_size
        )

    def sobel_y(
        self, image: np.ndarray, ddepth=cv2.CV_64F
    ) -> np.ndarray:
        """Filtro Sobel na direção Y."""
        return cv2.Sobel(
            image, ddepth, 0, 1, ksize=self.__kernel_size)

    def median(self, image: np.ndarray) -> np.ndarray:
        """Filtro da mediana."""
        return cv2.medianBlur(image, self.__kernel_size)

    def min(self, image: np.ndarray) -> np.ndarray:
        """Filtro mínimo."""
        return ndimage.minimum_filter(
            image, size=self.__kernel_size)

    def max(self, image: np.ndarray) -> np.ndarray:
        """Filtro máximo."""
        return ndimage.maximum_filter(
            image, size=self.__kernel_size)


class Augmenter:
    def __init__(self):
        pass

    def normalize(self) -> A.Compose:
        return A.Compose([

             A.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),

             ToTensorV2()
        ])

    def augment(self, p: float = 0.15) -> A.Compose:
        return A.Compose([
            A.RandomResizedCrop(height=256, width=256, scale=(0.75, 1.0)),
            A.Rotate(limit=45, p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.ShiftScaleRotate(shift_limit=0.05,
                               scale_limit=0,
                               rotate_limit=0,
                               p=p),
        ])
