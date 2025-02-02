import torch
import numpy as np


class Normalize:
    def __init__(self, is_gray: bool) -> None:
        """
        Classe para normalização de imagens.

        Args:
            is_gray (bool): Indica se a imagem é em escala de cinza.
        """
        self.__is_gray = is_gray

    def __call__(self,
                 img: np.ndarray,
                 label: np.uint8
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normaliza a ima-gem e o rótulo.

        Args:
            img (np.ndarray): Imagem a ser normalizada.
            label (np.uint8): Rótulo associado à imagem.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Imagem e rótulo normalizados.
        """
        return self.__norm_img(img), self.__norm_label(label)

    def __norm_img(self, img_to_norm: np.ndarray) -> torch.Tensor:
        """
        Normaliza a imagem.

        Args:
            img_to_norm (np.ndarray): Imagem a ser normalizada.

        Returns:
            torch.Tensor: Imagem normalizada.
        """
        img_normalized = torch.from_numpy(img_to_norm).float()
        if self.__is_gray:
            img_normalized = img_normalized.unsqueeze(0)
            img_normalized = img_normalized.repeat(3, 1, 1)
        else:
            img_normalized = img_normalized.permute(2, 0, 1)

        img_normalized /= 255.0

        return img_normalized

    def __norm_label(self, label_to_norm: np.uint8) -> torch.Tensor:
        """
        Normaliza o rótulo.

        Args:
            label_to_norm (np.uint8): Rótulo a ser normalizado.

        Returns:
            torch.Tensor: Rótulo normalizado.
        """
        label_normalized = torch.tensor(label_to_norm, dtype=torch.float32)
        return label_normalized.view(1)
