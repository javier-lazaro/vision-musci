# Detector de cartas en tiempo real #

El objetivo de este proyecto para la asignatura de Visión por Computador del MUSCI 24-25 es la creación de un sistema de detección de cartas de la baraja de naipes en tiempo real mediante el uso de métodos de visión artificial. La aplicación desarrollada utiliza la librería CV2 con la que se aplican diversas técnicas de procesamiento de imagen que extraen información característica de las cartas presentes en los fotogramas capturados mediante una cámara. La captura, procesamiento y muestra de resultados ocurre en tiempo real y busca tener la mayor robustez posible de cara a la extracción de las caracterísicas de las cartas. Esta robustez es un requisito importante del proyecto, ya que esta aplicación será reutilizada como módulo de entrada para el proyecto de la asigantura de Robótica y Automatización del MUSCI 24-25: https://github.com/motri/mucsi_robotica 

![pngegg](https://github.com/user-attachments/assets/d21d665c-ce31-4c22-b2cf-6b8f05720705)

## Carcaterísticas principales ##

La aplicación desarrollada para este proyecto permite extraer diversas características de las cartas de la baraja de naipes en tiempo real: 

1. Detección de número de cartas que se encuentran sobre el tablero
2. Cálculo de la posición de las cartas detectadas
3. Extracción del color para cada una de las cartas (Rojo/Negro)
4. Obtención del número que reprentan cada una de las cartas (1-10, J, Q, K)
5. Obtención del palo al que pertenece cada carta (Diamantes, Picas, Tréboles, Corazones)
6. Detección de la carta especial sin palo ni número (Joker)


## Pasos seguidos para el desarrollo ##

En esta sección se detallan los distintos métodos aplicados para obtener cada una de las características previamente mencionadas.

### Detección de número de cartas que existen sobre el tablero ###

1. Uso de Threshold para resaltar elementos claros (cartas) frente a un fondo oscuro (tablero)
2. Aplicación de filtro Gausiano para reducir la detección de contornos pequeños
3. Obtención de contornos frente al umbral y reordenado de la salida
4. Cálculo del area media y desviación estándar de la muestra de todos los contornos
5. Descarte de contornos que muestren valores atípicos en la distribución --> Contornos muy grandes y muy pequeños
6. Obtención del rectángulo con mínima área que comprende a los contornos resultado con minAreaRect() y uso de boxPoints() para mostrar el resultado por pantalla 

### Cálculo de la posición de las cartas detectadas ###

1. Obtención del centroide de cada uno de los contornos localizados en la fase anterior mediante moments()
2. Cálculo de las coordenadas X e Y de los centroides
3. Dibujado por pantalla mostrando el punto central y los valores de las coordenadas X e Y

## Siguientes pasos ##

1. Extracción de regiones de interés (ROI) para cada una de las cartas comprendidas dentro de los contornos:
    - Obtención de las zonas superior izquierda e inferior derecha (ROI 1) --> Información sobre el color, palo y número
    - Obtención de la zona intermedia (ROI 2) --> Especialmente útil para J, Q, K y Joker

<p align="center">
    <img src="https://github.com/user-attachments/assets/fc2b9640-ade1-4de7-9a15-069e18d44a03" alt="07 of hearts" width="300"/>
</p>

2. Cálculo del color de la carta mediante la obtención del color predominante (que no sea blanco) de la ROI 1
3. Obtención del palo mediante la distinción de los contornos de la ROI 1 (Posiblemente mediante filtro Canny y cálculo de áreas u otros métodos)
4. Obtención del número de la carta mediante una doble validación de carcaterísticas de la ROI 1 y 2 --> Detectar el número en la ROI 1 y contar contornos en la ROI 2
5. Muestra del color, palo y número en centroide de cada carta (en lugar del círculo y coordenada)

## Posibles mejoras ##

1. Gestión del reconocimiento cuando existen varias cartas posicionadas parcialmente una encima de la otra
2. Entrenamiento de una clasificador Haar para detectar las cartas que carecen de características representativas --> JOKER
