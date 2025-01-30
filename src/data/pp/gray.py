import numpy as np


class GrayConverter:
    def __init__(self):
        pass

    def luma(self,
             img: np.ndarray
             ) -> np.ndarray:

        img_gray = (
            0.2126 * img[..., 0] +
            0.7152 * img[..., 1] +
            0.0722 * img[..., 2])
        return img_gray
