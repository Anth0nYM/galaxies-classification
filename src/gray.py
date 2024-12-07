from typing import Callable
import numpy as np


def luma() -> Callable:
    ''' Converte a imagem para escala de cinza usando a função luma.

    Returns:
        Callable: Função que converte a imagem para escala de cinza.
    '''
    return lambda img: (
        0.2126 * img[..., 0] +
        0.7152 * img[..., 1] +
        0.0722 * img[..., 2]
    ).astype(np.uint8)
