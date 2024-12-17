import cv2
import numpy as np
from scipy import ndimage, stats


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
        return cv2.Laplacian(image, ddepth).astype(np.ndarray)

    def sobel_x(
        self, image: np.ndarray, ddepth=cv2.CV_64F
    ) -> np.ndarray:
        """Filtro Sobel na direção X."""
        return cv2.Sobel(
            image, ddepth, 1, 0, ksize=self.__kernel_size
        ).astype(np.ndarray)

    def sobel_y(
        self, image: np.ndarray, ddepth=cv2.CV_64F
    ) -> np.ndarray:
        """Filtro Sobel na direção Y."""
        return cv2.Sobel(
            image, ddepth, 0, 1, ksize=self.__kernel_size
        ).astype(np.ndarray)

    def median(self, image: np.ndarray) -> np.ndarray:
        """Filtro da mediana."""
        return cv2.medianBlur(image, self.__kernel_size).astype(np.ndarray)

    def mode(self, image: np.ndarray) -> np.ndarray:
        """Filtro de moda."""

        img = np.asarray(image)
        pad_size = self.__kernel_size // 2
        padded_img = np.pad(img, pad_size, mode='reflect')

        output = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded_img[i:i+self.__kernel_size,
                                    j:j+self.__kernel_size]

                mode_value = stats.mode(region, axis=None, keepdims=False)[0]
                output[i, j] = mode_value
        return output.astype(image.dtype)

    def min(self, image: np.ndarray) -> np.ndarray:
        """Filtro mínimo."""
        return ndimage.minimum_filter(
            image, size=self.__kernel_size).astype(np.ndarray)

    def max(self, image: np.ndarray) -> np.ndarray:
        """Filtro máximo."""
        return ndimage.maximum_filter(
            image, size=self.__kernel_size).astype(np.ndarray)


def augment():
    # TODO: augment pipeline
    return
