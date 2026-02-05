# Doom Visualizer - Deep Learning

## Introducción
Este repositorio contiene toda información relacionada a la implementación de un agente entrenado por aprendizaje profundo que sea capaz de progresar en un simulador del videojuego **Doom (1993)**

## Quick Start
Para poder jugar el visualizador se puede correr el siguiente comando: 
```bash
python doom_play.py --config game_config.yaml --keymap keymap.yaml
```
Para jugarlo en modo grabación (una vez terminado de jugar, se guardará en una carpeta llamada `recordings`):
```bash
python doom_play.py --config game_config.yaml --keymap keymap.yaml --record
```
Para poder ejecutar el modelo entrenado en el mapa configurado en `doom_test.py` se usa:
```bash
python doom_test.py
```
\* Para esto es necesario cargar el archivo del modelo `doom_model_gpu.pth`. Para el presente proyecto este fue entrenado usando los recursos de Google Colaboratory. Los detalles se encuentran en esta [carpeta](https://drive.google.com/drive/folders/1-juNDtYpMphmfPsBDAeyeJJujALwfqNF) (junto con el dataset y código de entrenamiento).