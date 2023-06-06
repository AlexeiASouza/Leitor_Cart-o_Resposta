[![GitHub contributors](https://img.shields.io/github/contributors/AlexeiASouza/Leitor_Cartao_Resposta?color=green)](https://github.com/AlexeiASouza/Leitor_Cartao_Resposta/graphs/contributors)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django?color=green)
[![GitHub forks](https://img.shields.io/github/forks/AlexeiASouza/Leitor_Cartao_Resposta?logoColor=green&style=social)](https://github.com/AlexeiASouza/Leitor_Cartao-Resposta/network/members)


# Algoritmo para leitura de cartões resposta
## Tutorial de instalação de requerimentos e dependencias

```python
-pip install -r requirements.txt
```

### -Baixar [protocolbuffer](https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip) 
- Adicionar pasta bin do protocol buffers ao path (variáveis de sistema)
    Baixar [tensoflow model garden](https://github.com/tensorflow/models/archive/refs/heads/master.zip)


>   cd models-master/research && protoc object_detection/protos/*.proto --python_out=. && copy >   object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py >   install
cd /models-master/research/slim && pip install -e . 

```python
pip install tensorflow`
pip install protobuf==3.20.*
```
