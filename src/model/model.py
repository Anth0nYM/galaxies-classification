import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch


class GalaxyClassifier(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 pre_trained: bool = True
                 ) -> None:
        """
        Inicializa o modelo GalaxyClassifier baseado na ResNet-50.

        Args:
            num_classes (int, optional): Número de classes.
                                         Padrão é 1.
            pre_trained (bool, optional): Se True, utiliza pesos pré-treinados.
                                           Padrão é True.
        """
        super(GalaxyClassifier, self).__init__()

        self.__num_classes = num_classes
        self.__pre_trained = pre_trained
        self.__weights = ResNet50_Weights.DEFAULT \
            if self.__pre_trained else None

        self.__model = resnet50(weights=self.__weights)

        in_features = self.__model.fc.in_features
        self.__model.fc = nn.Linear(in_features, self.__num_classes)

    def forward(self, x):
        """
        Executa a passagem direta (forward) do modelo.

        Args:
            x (torch.Tensor): Tensor de entrada
            contendo as imagens a serem classificadas.

        Returns:
            torch.Tensor: Saída do modelo com valores sigmoid.
        """
        logits = self.__model(x)
        return torch.sigmoid(logits)
