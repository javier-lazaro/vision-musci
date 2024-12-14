# Detector de cartas en tiempo real #

<p align="justify">
El objetivo de este proyecto para la asignatura de Visión por Computador del MUSCI 24-25 es la creación de un sistema de detección de cartas de la baraja de naipes en tiempo real mediante el uso de métodos de visión artificial. La aplicación desarrollada utiliza la librería CV2 con la que se aplican diversas técnicas de procesamiento de imagen que extraen información característica de las cartas presentes en los fotogramas capturados mediante una cámara. La captura, procesamiento y muestra de resultados ocurre en tiempo real y busca tener la mayor robustez posible de cara a la extracción de las caracterísicas de las cartas. 

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

### Calibración de la cámara ###

Se ha realizado una calibración de la cámara utilizada para las pruebas utilizando la [implementación de OpenCV](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) (cv2.calibrateCamera). La calibración es particular para cada cámara, por lo que su uso puede no ser conveniente en todos los casos. 

El código implementado para la calibración se encunetra en [src/utils/static/calibracion_camara.py](https://github.com/javier-lazaro/vision-musci/blob/main/src/utils/static/calibracion_camara.py) y el resultado de misma está guardado en [static/npz/calibration_data.npz](https://github.com/javier-lazaro/vision-musci/blob/main/static/npz/calibration_data.npz).

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

La lógica descrita en estas dos secciones anteriores se encuentra localizada en el bucle principal dentro de [src/main.py](https://github.com/javier-lazaro/vision-musci/blob/main/src/main.py)

### Extracción de colores de las cartas ###

Esta sección engloba dos pasos importantes. Por un lado, el postprocesado de las cartas detectadas en los pasos anteriores mediante una rotación de perspectiva y, por otro lado, a extracción del color más representativo para cada una de las cartas postprocesadas. Se han seguido los siguientes pasos para cada una de estas partes: 

1. Obtención y ordenación según ubicación de las 4 esquinas de cada rectángulo de carta detectado
2. Cálculo de largura de los lados del rectángulo para tener información sobre la orientación de la carta frente a la cámara
3. Rotación de los píxeles dentro del box mediante una transformación de perspectiva.

<p align="center">
    <img src="https://github.com/user-attachments/assets/7c876051-e7c3-4ff4-9e66-9ebaf6eed9c9" alt="Transformation" width="400"/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    
    <img src="https://github.com/user-attachments/assets/c62210ca-842e-469c-9060-63e421966c8e" alt="Transformation v2" width="400"/>
</p>

El resultado es una versión "enderezada" de cada una de las cartas detectadas en la sección inicial. A cada una de estas cartas coreguidas se les han aplicado las siguientes operaciones para extraer su color:

1. Transformación del rectángulo de píxeles a HSV
2. Creación de umbrales para extracción de colores: Rojo, Negro y Amarillo
3. Aplicación de máscaras usando estos umbrales para obtener un porcentaje de coincidencia frente a cada carta

El uso del umbral paa el color amarillo, permite que este módulo también sea capaz de determinar si una carta es una carta común (carta entre el 1 y 10) o una figura (J, Q, K), ya que estas últimas son las únicas que tienen un porcentaje significativo de color amarillo. 

La lógica descrita para la transformación de las cartas y la extracción del color se encuentra en [src/utils/real_time/detector_color.py](https://github.com/javier-lazaro/vision-musci/blob/main/src/utils/real_time/detector_color.py)

### Extracción de números de las cartas ###

La extracción de números de carta se realiza de forma distinta si la carta se trata de una carta común o una figura. Para las cartas comunes: 

1. Creación de un rectángulo para analizar región central de interés usándo el centroide de cada carta detectada como punto central
2. Aplicación de un filtro Canny en base al threshold, junto con cerrado automático de contornos abiertos, sobre la región central de interés de cada carta
3. Recuento de contornos obtenidos con lógica adicional para detectar conotornos inusualmente pequeños y obviarlos para el conteo

La implementación de la lógica anterior se encuentra en [src/utils/real_time/detector_numero.py](https://github.com/javier-lazaro/vision-musci/blob/main/src/utils/real_time/detector_numero.py)

La extracción del número de carta para las figuras se hace de la siguiente manera:

1. Extracción de información sobre si la carta es común o figura mediante el módulo de detección de color
2. Aplicación de un modelo YOLO entrenado para detectar las clases J, Q y K sólo sobre las cartas que son figuras

La implementación de la lógica anterior se encuentra en [src/utils/real_time/detector_figuras.py](https://github.com/javier-lazaro/vision-musci/blob/main/src/utils/real_time/detector_figuras.py)

### Extracción de palos de las cartas ###

Al igual que para la detección de figuras (J, Q y K), para la detección de los distintos palos de cada carta se ha utilizado un modelo YOLO entrenado manualmente. El entrenamiento ha sido realizado utilizando un dataset de 342 imágenes de nuestra baraja de cartas. El dataset está compuesto de imágenes de cartas individuales con distintos fondos, rotaciones e iluminaciones. La anotación del conjutnto de datos se ha realizado de forma manual utilizando [Roboflow](https://roboflow.com/). Se ha aplicado también un data augmentation para generar imágenes con _blur_, incrementando el número de elementos del dataset a 800, las imágenes de salida tienen un tamaño de 640x640 píxeles. 

El modelo ha sido entrenado usando YOLO Nano como punto de partida durante 100 epochs. El modelo es capaz de detectar las siguientes clases: **Rombo, Pica, Trébol, Corazón, J, Q K y Joker**. Las métricas obtenidas frente al conjunto de datos de validación se recogen en la siguiente tabla:

| Precision (B) | Recall (B)   | mAP50 (B) | mAP50-95 (B) |
| ------------- |:------------:| ---------:| ------------:|
| 0.93918       | 0.94923      | 0.95645   | 0.74899      |

Utilizando el modelo YOLO entrenado, el proceso de exracción del palo se realiza de la siguiente manera:

1. Carga del modelo YOLO utilizando los pesos tras el entrenamiento
2. Obtención de las clases pedecidas para los píxeles transformados tras la extracción del color
3. Cribado inicial de los resultados para obtener la clase con mayor número de coincidencias
4. Operación de post procesado adicional para comparar la clase detectada (palo) con el color extraído de la carta.
   - Si el resultado es favorable se devuelve el palo detectado
   - Si la predición es imposible dado el color, se aplica una correción manual en base al vecino más próximo. Se devuelve el palo corregido.
   - Si el palo es detectado como Joker, no se realiza ningún postprocesado

La lógica descrita se encuentra implementada en [src/utils/real_time/detector_figura.py](https://github.com/javier-lazaro/vision-musci/blob/main/src/utils/real_time/detector_figuras.py)

## Uso de la aplicación ##

La aplicación final en OpenCV permite utilizar todas las funciones descritas anteriormente mediante una serie de controles por teclado. Inicialmente, la aplicación muestra una ventana principal con las imágenes capturadas en tiempo real desde la cámara, así como una ventana secundaria donde se muestra el resultado del threshold a modo de referencia. Por defecto, cada vez que una carta aparece en un frame, se detecta su presencia. Esto se simboliza mediante un rectágulo rojo que engloba a la carta, así como un punto verde dibujado sobre su centroide con sus coordenadas. Un texto en la parte superior izquierda de la pantalla muestra el número total de cartas detectadas en cada momento.   

Un menú en la zona superior derecha muestra los controles para activar o desactivar las funciones adicionales de la aplicación:

- Tecla "p" : Activa o desactiva la calibración de la cámara
- Tecla "c" : Activa o desactiva la detección de color para las cartas
- Tecla "n" : Activa o desactiva la detección de los números de las cartas normales (cartas del 1 al 10)
- Tecla "y" : Activa o desactiva el uso de YOLO para la detección del palo y las figuras 


<img width="1146" alt="image" src="https://github.com/user-attachments/assets/13963a90-c819-4c44-9fd8-42602b09f7fd" />

## Problemas conocidos ##

1. La detección de los números para las cartas normales es muy dependiente de la calidad del threshold aplicado. Requiere de una iluminación estable y de buena calibración del dispositivo de captura.
2. El modelo YOLO tiene una peor precisión para la detección de figuras (J, Q, K) que para los palos (Rombo, Pica, Corazón, Trébol).
3. La naturaleza de la captura en tiempo real hace que las detecciones de las características tengan imperfecciones puntuales durante la captura. 

</p>
