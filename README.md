# Galaxies Classification (Classificação de Galáxias)

## How to Run

1. [ ] **Download the Dataset:**  
   Download the dataset file from [Galaxy10 Dataset](https://github.com/henrysky/Galaxy10).

2. [ ] **Set Up the Project:**  
   Create a folder named `data/` at the root of the project and move the downloaded file into this folder.

3. [ ] **First Execution:**  
   During the first run, a subset of the original dataset will be created, containing only the "Round Smooth" and "Barred Spiral" classes. This subset will be saved as a new file inside the `data/` folder. This process might take some time depending on your hardware.

4. [ ] **Future Executions:**  
   After the first run, data will be loaded directly from the newly created file, improving processing efficiency.

---

## Task Overview

This project implements a pipeline to classify galaxies into two categories:

- **"Round Smooth"** (round and smooth galaxies).
- **"Barred Spiral"** (spiral galaxies with bars).

### Data Preprocessing Steps

- [ ] Convert images to grayscale.
- [ ] Apply noise removal filters (median, mean, max, min).
- [ ] Perform data augmentation (brightness adjustments, spatial transformations, etc.).

### Hypotheses

- [ ] The chosen CNN model is expected to outperform the baseline model presented in the original article.
- [ ] Applying noise removal filters and data augmentation will enhance performance.

### Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### Experiments

- [ ] Binary evaluation of color models: **RGB vs Grayscale**.
- [ ] Binary evaluation of the impact of selected filters (with or without filter).
- [ ] Comparison between the baseline model and the proposed model.
- [ ] Hyperparameter tuning, including:
  - Number of epochs.
  - Learning rate.
  - Activation functions.
  - Optimizers.

---

## Como executar

1. [ ] **Baixe o Dataset:**  
   Faça o download do arquivo do dataset em [Galaxy10 Dataset](https://github.com/henrysky/Galaxy10).

2. [ ] **Configure o Projeto:**  
   Crie uma pasta chamada `data/` na raiz do projeto e mova o arquivo baixado para essa pasta.

3. [ ] **Primeira Execução:**  
   Na primeira execução, será criado um subconjunto do dataset original contendo apenas as classes "Round Smooth" e "Barred Spiral". Esse subconjunto será salvo em um novo arquivo dentro da pasta `data/`. Este processo pode levar algum tempo, dependendo do hardware disponível.

4. [ ] **Execuções Futuras:**  
   Após a primeira execução, os dados serão carregados diretamente do arquivo criado, otimizando o tempo de processamento.

---

## Resumo da Tarefa

Este projeto implementa um pipeline para classificação de galáxias em duas categorias:

- **"Round Smooth"** (galáxias arredondadas e suaves).
- **"Barred Spiral"** (galáxias espirais barradas).

### Etapas de Pré-processamento de Dados

- [ ] Conversão de imagens para escala de cinza.
- [ ] Aplicação de filtros para remoção de ruído (mediana, média, máximo, mínimo).
- [ ] Aumento de dados (alterações de luminosidade, transformações espaciais, entre outros).

### Hipóteses

- [ ] O modelo CNN escolhido deve superar o desempenho do modelo apresentado no artigo base.
- [ ] A aplicação de filtros de remoção de ruído e aumento de dados deve melhorar os resultados.

### Métricas Utilizadas

- **Acurácia**
- **Precisão**
- **Recall**
- **F1-Score**

### Experimentos

- [ ] Comparação binária entre modelos de cores: **RGB vs Grayscale**.
- [ ] Avaliação do impacto dos filtros escolhidos (com ou sem filtro).
- [ ] Comparação entre o modelo do artigo base e o modelo proposto.
- [ ] Ajuste de hiperparâmetros, incluindo:
  - Número de épocas.
  - Taxa de aprendizado.
  - Funções de ativação.
  - Otimizadores.
