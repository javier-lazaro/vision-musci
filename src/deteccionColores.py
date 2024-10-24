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

    # Aplicar la función threshold
    ret, thresh_1 = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

    # Aplicar un segundo threshold
    ret, thresh_2 = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

    # Combinar los resultados de los dos thresholds
    thres_combi  = cv2.bitwise_and(thresh_1, thresh_2)

    # Convertir la imagen a color
    color = cv2.cvtColor(thres_combi, cv2.COLOR_GRAY2BGR)

    # Aplicar la máscara a la imagen original
    frame = cv2.bitwise_and(frame, color) # Al cambiar de color, los 0 (negro) se quedan a 0, y cambia unicamente lo que es blanco

    # Encontrar contontornos en la imagen umbralizada
    contours, _ = cv2.findContours(thres_combi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    cv2.imshow('VentanaCartas2', frame)  # Muestra el fotograma actual en la ventana.

    # Recuperamos el color de las cartas



    #cv2.imshow('VentanaCartas2', thresh)  # Muestra el fotograma actual en la ventana.

    success, frame = cap.read()  # Lee el siguiente fotograma de la cámara. 


# Paso 1, saber cuantas cartas hay sobre la mesa
# Paso 2, saber el color de las distitnas cartas que hay en la mesa
# Paso 3, saber el número de las distintas cartas que hay sobre la mesa
# Paso 4, el palo de las distintas cartas que hay sobre la mesa
# Paso 5, saber la posición de las cartas sobre la mesa


