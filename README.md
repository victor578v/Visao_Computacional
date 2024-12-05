# Projeto de Visão Computacional

## Descrição
Este projeto foi desenvolvido para a disciplina Fundamentos de Inteligência Artificial (FIA) - Graduação, do Prof. Pablo De Chiaro. 
O projeto tem como objetivo aplicar técnicas de Visão Computacional para resolver problemas práticos relacionados ao reconhecimento de frutas e vegetais, bem como à detecção de condições específicas, como frutas podres, em vídeos. Utilizando de modelos pré-treinados, para realizar a detecção de frutas e vegetais, além de implementar técnicas de detecção de objetos em tempo real para analisar vídeos. O objetivo final é construir uma aplicação que consiga identificar e rastrear frutas em vídeos e, caso alguma delas não atenda aos critérios de qualidade, como ser identificada como "podre" (Considerando que seu estado tenha se degradado o suficiente), o sistema irá alertar sobre a condição da fruta.

Integrante do Grupo: Victor Andrei

## Modelo Pre Treinado
O modelo FOOD-INGREDIENT-CLASSIFICATION-MODEL que foi usado para a detecção de frutas pode ser baixado através do seguinte link:

- [Download FOOD-INGREDIENT-CLASSIFICATION-MODEL.pth](https://www.kaggle.com/models/sunnyagarwal427444/food-ingredient-classification-model)

Extraia o arquivo `fruits_vegetables_51.pth` do arquivo tar.gz baixado e coloque-o no diretório `detecta-maturidade` do projeto.

## Instalação de Dependências

Certifique-se de que seu ambiente virtual esteja ativado. Instale as dependências listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Conteúdo do arquivo `requirements.txt`:

```text
numpy==2.0.0
opencv-python==4.10.0.84
torch==2.5.0
torchvision==0.20.0
Pillow==9.5.0
```

## Verificação da Instalação

Para verificar se as bibliotecas foram instaladas corretamente, você pode executar o seguinte comando em um terminal Python:

```python
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
```

## Executando o Projeto

Para executar a deteccao de frutas, simplesmente execute o script `main.py` com Python. Certifique-se de que todos os arquivos necessários estão na mesma pasta que o script.

## Controles

Durante a execução do projeto, você pode:

- Pressionar 'p' para pausar/continuar o vídeo.
- Pressionar 'q' para sair do aplicativo.
