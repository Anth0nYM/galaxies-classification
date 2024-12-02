# Classificação de Galáxias

Pipeline para a classificação de galáxias do tipo "Barred Spiral" e "Round Smooth"

## Pré-processamento de dados

* Conversão para escala de cinza,
* Filtros para remoção de ruído (mediana, media, maximo, minimo),
* Aumento de dados (Modificações de luminosidade, espaciais, etc).

## Hipóteses

Espera-se que o modelo de cnn escolhido performe melhor que o artigo base, além de uma melhora com o uso de filtros removedores de ruído e aumento de dados.

## Métricas

Acurácia, precisão, recall, F1-score.

## Experimentos

Avaliação binária de modelo de cores (RGB vs Grayscale), avaliação Binária de cada filtro escolhido (Com ou sem filtro), comparação binária entre modelo do artigo base com o escolhido e ajuste de Hiperparâmetros (Número de épocas, Taxa de aprendizado,Funções de ativação, Otimizadores)
