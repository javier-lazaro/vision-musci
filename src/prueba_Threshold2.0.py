import cv2
import numpy as np

# Ruta de la imagen de la carta
image_path = "vision-musci/static/images/J_treboles.jpg"

# Leer la imagen de la carta
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
ret, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# Encontrar los contornos en la imagen umbralizada
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos en la imagen original
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Mostrar la imagen con los contornos detectados
cv2.imshow("Contornos detectados", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Cargar las plantillas de los palos
templates = {
    "corazones": cv2.imread("proyecto/templates/corazones.png", 0),
    "treboles": cv2.imread("proyecto/templates/treboles.png", 0),
    "diamantes": cv2.imread("proyecto/templates/diamantes.png", 0),
    "picas": cv2.imread("proyecto/templates/picas.png", 0)
}

# Verificar si las plantillas se han cargado correctamente
for name, template in templates.items():
    if template is None:
        print(f"Error al cargar la plantilla: {name}")
        exit()

# Función para comparar la ROI con las plantillas
def match_template(roi, templates):
    best_match = None
    best_score = float('inf')
    for name, template in templates.items():
        res = cv2.matchTemplate(roi, template, cv2.TM_SQDIFF_NORMED)
        min_val, _, _, _ = cv2.minMaxLoc(res)
        if min_val < best_score:
            best_score = min_val
            best_match = name
    return best_match

# Extraer la región de interés (ROI) que contiene el símbolo del palo
# Aquí asumimos que el símbolo del palo está en la esquina superior izquierda
x, y, w, h = 0, 0, 50, 50  # Ajusta estos valores según sea necesario
roi = thresh[y:y+h, x:x+w]

# Mostrar la ROI para verificar que contiene el símbolo del palo
cv2.imshow("ROI", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Verificar si la ROI es más grande que las plantillas
for name, template in templates.items():
    if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
        print(f"La ROI es más pequeña que la plantilla: {name}")
        exit()

# Identificar el palo de la carta
palo = match_template(roi, templates)
print(f"El palo de la carta es: {palo}")