import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch


class GalaxyClassifier(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 pre_trained: bool = True):

        super(GalaxyClassifier, self).__init__()

        self.__num_classes = num_classes
        self.__pre_trained = pre_trained
        self.__weights = ResNet50_Weights.DEFAULT \
            if self.__pre_trained else None

        self.__model = resnet50(weights=self.__weights)

        in_features = self.__model.fc.in_features
        self.__model.fc = nn.Linear(in_features, self.__num_classes)

    def forward(self, x):
        logits = self.__model(x)
        return torch.sigmoid(logits)
