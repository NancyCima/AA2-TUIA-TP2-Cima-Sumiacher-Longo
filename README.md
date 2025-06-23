# Trabajo Práctico 2 - Aprendizaje Automático II

**Universidad Nacional de Rosario - FCEIA**

**Carrera:** Tecnicatura Universitaria en Inteligencia Artificial

**Materia:** Aprendizaje Automático II

## Integrantes

* **Cima, Nancy Lucía**
  Email: [nancy.cima.bertoni@hotmail.com](mailto:nancy.cima.bertoni@hotmail.com)
* **Longo, Gonzalo**
  Email: [longogonzalo.g@gmail.com](mailto:longogonzalo.g@gmail.com)
* **Sumiacher, Julia**
  Email: [jsumiacher@gmail.com](mailto:jsumiacher@gmail.com)

## Descripción general

Este repositorio contiene la solución al **Trabajo Práctico 2**, el cual abarca dos problemas principales relacionados con el aprendizaje automático:

### Problema 1: Clasificación de Audio - Audio MNIST

* Se trabaja con el dataset [`spoken_digit`](https://www.tensorflow.org/datasets/catalog/spoken_digit) de TensorFlow Datasets.
* El objetivo es entrenar modelos que clasifiquen clips de audio (0 al 9) en base a sus espectrogramas.
* Se implementan dos modelos:

  * **Modelo convolucional (CNN)** sobre espectrogramas.
  * **Modelo recurrente (RNN)** sobre espectrogramas.
* La solución incluye:

  * Análisis y preprocesamiento de datos.
  * Entrenamiento y evaluación de modelos.
  * Métricas y visualizaciones relevantes.

### Problema 2: Aprendizaje por Refuerzo - Flappy Bird

* Se entrena un agente para jugar Flappy Bird utilizando la librería [PyGame Learning Environment](https://pygame-learning-environment.readthedocs.io/en/latest/).
* Se implementan dos enfoques:

  * **Agente basado en Q-Learning** (discretización de estados).
  * **Agente basado en Deep Q-Learning** (con red neuronal).
* Se provee soporte para elegir el tipo de agente mediante el parámetro `--agent` al ejecutar `test_agent.py`.
