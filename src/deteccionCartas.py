import cv2
import numpy as np

# Lectura de imagen en tiempo real
cap = cv2.VideoCapture(0)

# Crea una ventana llamada 'VentanaCartas'.
cv2.namedWindow('VentanaCartas')
cv2.namedWindow('VentanaCartas2')

# Lee el primer fotograma de la cámara.
success, frame = cap.read() # Succes indica si la lectura fue exitosa.

# Bucle para mostrar el video en tiempo real.
while success and cv2.waitKey(1) == -1: 
    
    cv2.imshow('VentanaCartas', frame)  # Muestra el fotograma actual en la ventana.

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro Gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar un umbral binario para segmentar la imagen
    ret, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # Encontrar los contornos en la imagen umbralizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    # Ordenar los contornos por área de mayor a menor
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Eliminamos los contornos pequeños
    contours = [contour for contour in contours if cv2.contourArea(contour) > 1000 ] 

    # Eliminamos el contorno más grande (puede que no lo necesitemos)
    if contours:
        contours.pop(0)

    contours = [contour for contour in contours if cv2.contourArea(contour) > 130000 ] 
    print(len(contours))

    # Calculamos cuantas cartas hay en la mesa
    # Imprimimos el area del ultimo contorn

    # Dibujar los contornos en la imagen original
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2) 

    cv2.imshow('VentanaCartas2', frame)  # Muestra el fotograma actual en la ventana.

    success, frame = cap.read()  # Lee el siguiente fotograma de la cámara. 

# Imprimimos el area de los cuatro contornos más grandes
if contours:
    for contour in contours[:4]:
        print(cv2.contourArea(contour))

# Paso 1, saber cuantas cartas hay sobre la mesa
# Paso 2, saber el color de las distitnas cartas que hay en la mesa
# Paso 3, saber el número de las distintas cartas que hay sobre la mesa
# Paso 4, el palo de las distintas cartas que hay sobre la mesa
# Paso 5, saber la posición de las cartas sobre la mesa


