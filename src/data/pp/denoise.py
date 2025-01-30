import cv2
import numpy as np
from scipy import ndimage


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
