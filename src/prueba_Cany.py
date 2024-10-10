import cv2
import numpy as np

# Ruta de la imagen
image_path = "proyecto/images/As_corazones.PNG"

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

# Aplicar el algoritmo de Canny para detectar bordes
edges = cv2.Canny(blurred, 50, 150)

# Encontrar los contornos en la imagen de bordes
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos en la imagen original
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Mostrar la imagen con los contornos detectados
cv2.imshow("Contornos detectados", img)

# Esperar a que se presione una tecla para cerrar la ventana
cv2.waitKey(0)
cv2.destroyAllWindows()