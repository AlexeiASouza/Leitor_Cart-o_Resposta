# Documentação do Código de Detecção e Segmentação de Objetos

### Descrição

Este documento apresenta a documentação completa para o código de detecção e análise de imagens de provas. O objetivo deste código é permitir a detecção automatizada de objetos em imagens de provas, a extração de regiões de interesse (ROIs) correspondentes e a realização de operações de análise nessas ROIs. O código utiliza um modelo pré-treinado para a detecção de objetos, bem como bibliotecas populares como OpenCV, TensorFlow e NumPy para processamento de imagem e análise. Esta documentação fornecerá uma visão geral do código, descreverá suas principais funções, entradas e saídas, além de fornecer instruções detalhadas para a instalação das bibliotecas necessárias.

### Funcionalidades

O código oferece as seguintes funcionalidades principais:

- Detecção de Objetos: Utiliza um modelo de detecção pré-treinado para processar uma imagem e identificar objetos presentes nela.
- Filtragem de Previsões: Filtra as detecções com base em uma pontuação atribuída a cada objeto detectado.
- Obtenção de Coordenadas da Caixa Delimitadora: Localiza a caixa delimitadora de um objeto com base no rótulo atribuído a ele.
- Extração de Região de Interesse (ROI): Extrai a região de interesse correspondente à caixa delimitadora de um objeto na imagem.
- Operações Morfológicas: Aplica operações morfológicas à imagem para melhorar a qualidade da detecção.

Instalação

Para executar este código, as seguintes bibliotecas devem ser instaladas:

OpenCV (versão 3.4.2 ou superior): Biblioteca popular de visão computacional e processamento de imagem.
TensorFlow (versão 2.0 ou superior): Plataforma de aprendizado de máquina de código aberto amplamente utilizada para treinamento e inferência de modelos de aprendizado profundo.
NumPy (versão 1.16.4 ou superior): Biblioteca essencial para cálculos numéricos em Python.

Siga as instruções abaixo para instalar as bibliotecas necessárias usando o gerenciador de pacotes pip:

```sh
pip install opencv-python
pip install tensorflow
pip install numpy
```

Certifique-se de ter privilégios de administrador ou utilize um ambiente virtual Python para evitar conflitos de dependências.

### Utilização

Para utilizar este código, siga as etapas abaixo:

1. Importe as funções necessárias para o seu código.
2. Carregue o modelo de detecção pré-treinado desejado.
3. Carregue uma imagem para realizar a detecção e segmentação de objetos.
4. Chame a função `detect_fn` para obter as detecções na imagem.
5. Chame a função `get_preds` para filtrar as previsões com base em um limite de pontuação.
6. Chame a função `get_box_by_label` para obter as coordenadas da caixa delimitadora de um objeto específico.
7. Chame a função `get_object` para extrair a região de interesse correspondente à caixa delimitadora.
8. Opcionalmente, chame a função `morphology_operations` para aplicar operações morfológicas à imagem.
9. Processe e utilize as detecções e ROIs obtidas conforme necessário.

### Funções Principais

#### `detect_fn(image, model)`

Esta função recebe uma imagem e um modelo de detecção pré-treinado e retorna as detecções dos objetos presentes na imagem.

- Entrada:
    - `image` (array NumPy): Array que representa a imagem a ser processada.
    - `model` (objeto TensorFlow): Objeto do modelo de detecção pré-treinado.
- Saída:
    - `detections` (array NumPy):

 Array contendo as detecções dos objetos na imagem.

#### `get_preds(detections, score)`

Esta função recebe as detecções de objetos e um limite de pontuação e retorna as previsões filtradas com base na pontuação.

- Entrada:
    - `detections` (array NumPy): Array contendo as detecções dos objetos na imagem.
    - `score` (float): Limite de pontuação para filtrar as previsões.
- Saída:
    - `preds` (array NumPy): Array contendo as previsões filtradas.

#### `get_box_by_label(preds, label)`

Esta função recebe as previsões de objetos e um rótulo específico e retorna as coordenadas da caixa delimitadora desse objeto.

- Entrada:
    - `preds` (array NumPy): Array contendo as previsões filtradas.
    - `label` (str): Rótulo do objeto desejado.
- Saída:
    - `box` (tuple): Coordenadas (xmin, ymin, xmax, ymax) da caixa delimitadora do objeto.

#### `get_object(image, box, ofs)`

Esta função recebe uma imagem, as coordenadas da caixa delimitadora de um objeto e um deslocamento (ofs) e retorna a região de interesse correspondente à caixa delimitadora.

- Entrada:
    - `image` (array NumPy): Array que representa a imagem original.
    - `box` (tuple): Coordenadas (xmin, ymin, xmax, ymax) da caixa delimitadora do objeto.
    - `ofs` (int): Valor de deslocamento para aumentar a região de interesse.
- Saída:
    - `object_roi` (array NumPy): Array contendo a região de interesse (ROI) do objeto.

#### `morphology_operations(image)`

Esta função recebe uma imagem e aplica operações morfológicas para melhorar a qualidade da detecção.

- Entrada:
    - `image` (array NumPy): Array que representa a imagem original.
- Saída:
    - `processed_img` (array NumPy): Array contendo a imagem processada.

Exemplo de Uso

A seguir, um exemplo de como usar as funções em um código:


```python
import cv2
import tensorflow as tf
import numpy as np

# Carregar o modelo de detecção pré-treinado
model = load_detection_model("modelo_de_deteccao")

# Carregar a imagem
image = cv2.imread("image.jpg")

# Realizar deteccção de objetos
detections = detect_fn(image, model)

# Filtrar as previsões com a base na pontuação 
preds =  get_preds(detections, score = 0.5)

# Obter a caixa delimitadora de um objeto específico
box = get_box_by_label(preds, label = "cachorro")

# Extrai a região de interesse correspondente à caixa delimitadora
objetc_roi = get_object(image, box, ofs = 10)

# Aplicar operações morfológicas à imagem
processed_img = morphology_operations(image)
```


Certifique-se de substituir as chamadas de função `load_detection_model("modelo_de_deteccao")`, `cv2.imread("imagem.jpg")` e as etapas finais do exemplo com seu próprio código, de acordo com o contexto em que está utilizando as funções.


### Fluxo de Execução do Código
- Carregar a imagem de prova.
- Chamar a função `detect_objects()` para obter as detecções dos objetos presentes na imagem.
- Chamar a função `filter_predictions()` para filtrar as previsões com base em uma pontuação de limite.
- Para cada objeto detectado:
    - Chamar a função `get_object_roi()` para extrair a região de interesse correspondente ao objeto.
    - Aplicar operações morfológicas na ROI chamando a função `apply_morphological_operations()`.
    - Realizar operações de análise na ROI conforme necessário.
    - Apresentar os resultados da análise.

### Considerações Finais

Este código pode ser usado

 como ponto de partida para a detecção e segmentação de objetos em imagens de provas. No entanto, ele pode exigir adaptações adicionais, como o treinamento de um modelo específico para detectar os objetos de interesse em suas imagens de prova. Além disso, outros pré-processamentos e pós-processamentos podem ser necessários dependendo do domínio do problema e dos requisitos específicos.

