import cv2
import numpy as np

# Cargar la imagen de la carta
image = cv2.imread('static/images/J_treboles.jpg')

# Crea una ventana llamada 'VentanaCartas'.
cv2.namedWindow('VentanaCartas2')

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro Gaussiano para reducir el ruido
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Aplicar un umbral binario para segmentar la imagen
ret, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# Encontrar los contornos en la imagen umbralizada
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

# Ordenar los contornos por área de mayor a menor
if contours:
    contours = sorted(contours, key=cv2.contourArea, reverse=True)


if contours:
    # Calculamos la media de aera de los contornos
    media = 0
    for contour in contours:
        media += cv2.contourArea(contour)
    media = media / len(contours)

    # Calculamos la desviación estandar
    desviacion = 0
    for contour in contours:
        desviacion += (cv2.contourArea(contour) - media) ** 2
    desviacion = (desviacion / len(contours)) ** 0.5

    # Eliminamos los contornos que no sean cartas
    contours = [contour for contour in contours if cv2.contourArea(contour) > media + 2 * desviacion]

# TODO: Buscar una forma de determinar la forma que tiene el contorno, es decir que sea un elemento rectangular
# TODO: Calcaulo de proporción


# Dibujar los contornos en la imagen original
cv2.drawContours(image, contours, -1, (0, 255, 0), 2) 

# Mostrar el resultado en la ventana 'VentanaCartas2'
cv2.imshow('VentanaCartas2', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Imprimimos el area de los cuatro contornos más grandes
if contours:
    for contour in contours[:4]:
        print(cv2.contourArea(contour))

# Paso 1, saber cuantas cartas hay sobre la mesa
# Paso 2, saber el color de las distitnas cartas que hay en la mesa
# Paso 3, saber el número de las distintas cartas que hay sobre la mesa
# Paso 4, el palo de las distintas cartas que hay sobre la mesa
# Paso 5, saber la posición de las cartas sobre la mesa
# Paso Extra, detección del joker


