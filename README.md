# Galaxies Classification (Classificação de Galáxias)

## How to Run

1. ⬇️**Download the Dataset:**  
   Download the [dataset file](https://www.kaggle.com/datasets/anth0nym/galaxy10-decals-fork-dataset/data).

2. ⬇️**Install Dependencies:**
   Install the project dependencies using the following command: `pip install -r requirements.txt`. Most dependencies will be installed automatically. However, [PyTorch](https://pytorch.org/) installation depends on your hardware configuration (GPU/CPU). It is recommended to install PyTorch directly from the [official website](https://pytorch.org/get-started/locally/), following the specific instructions for your environment.

3. ⚙️**Configure the Project:**
   - Create a folder named `data/` in the project root directory and move the downloaded file into this folder.

---

## Task Overview

This project implements a pipeline to classify galaxies into two categories:

- **"Round Smooth"** (round and smooth galaxies).
- **"Barred Spiral"** (spiral galaxies with bars).

### Data Preprocessing Steps

- Convert images to grayscale.
- Apply noise removal filters (median, mean, max, min).
- Perform data augmentation (brightness adjustments, spatial transformations, etc.).

### Hypotheses

- The chosen CNN model is expected to outperform the baseline model presented in the original article.
- Applying noise removal filters and data augmentation will enhance performance.

### Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### Experiments

- Binary evaluation of the impact of selected filters (with or without filter).
- Comparison between the baseline model and the proposed model.
- Hyperparameter tuning, including:
  - Number of epochs.
  - Learning rate.
  - Activation functions.
  - Optimizers.

---

## Como executar

1. ⬇️**Baixe o Dataset:**  
   Faça o download do [arquivo do dataset](https://www.kaggle.com/datasets/anth0nym/galaxy10-decals-fork-dataset/data).

2. ⬇️**Instale as dependências:**  
   Instale as dependências do projeto utilizando o comando: `pip install -r requirements.txt`, a maioria das dependências será instalada automaticamente. No entanto, a instalação do [PyTorch](https://pytorch.org/) depende da sua configuração de hardware (GPU/CPU). Recomenda-se instalá-lo diretamente do [site oficial](https://pytorch.org/get-started/locally/), seguindo as instruções específicas para o seu ambiente.

3. ⚙️**Configure o Projeto:**  
   - Crie uma pasta chamada `data/` na raiz do projeto e mova o arquivo baixado para essa pasta.

---

## Resumo da Tarefa

Este projeto implementa um pipeline para classificação de galáxias em duas categorias:

- **"Round Smooth"** (galáxias arredondadas e suaves).
- **"Barred Spiral"** (galáxias espirais barradas).

### Etapas de Pré-processamento de Dados

- Conversão de imagens para escala de cinza.
- Aplicação de filtros para remoção de ruído (mediana, média, máximo, mínimo).
- Aumento de dados (alterações de luminosidade, transformações espaciais, entre outros).

### Hipóteses

- O modelo CNN escolhido deve superar o desempenho do modelo apresentado no artigo base.
- A aplicação de filtros de remoção de ruído e aumento de dados deve melhorar os resultados.

### Métricas Utilizadas

- **Acurácia**
- **Precisão**
- **Recall**
- **F1-Score**

### Experimentos

- Avaliação do impacto dos filtros escolhidos (com ou sem filtro).
- Comparação entre o modelo do artigo base e o modelo proposto.
- Ajuste de hiperparâmetros, incluindo:
  - Número de épocas.
  - Taxa de aprendizado.
  - Funções de ativação.
  - Otimizadores.
