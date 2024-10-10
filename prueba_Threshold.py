import cv2
import numpy as np

# Ruta de la imagen
image_path = "proyecto/images/baraja.jpg"

# Leer la imagen
img = cv2.imread(image_path)

# Verificar si la imagen se ha cargado correctamente
if img is None:
    print("Error al cargar la imagen.")
    exit()

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro Gaussiano para reducir el ruido
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Aplicar un umbral binario para segmentar la imagen
ret, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# Encontrar los contornos en la imagen umbralizada
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Verificar si se encontraron contornos
if len(contours) > 0:
    # Eliminar el mayor contorno que corresponde a la imagen completa
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Dibujar los contornos en la imagen original
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Mostrar la imagen con los contornos detectados
cv2.imshow("Contornos detectados", img)

# Esperar a que se presione una tecla para cerrar la ventana
cv2.waitKey(0)
cv2.destroyAllWindows()