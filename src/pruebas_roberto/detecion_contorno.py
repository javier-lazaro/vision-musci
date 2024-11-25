import cv2
import numpy as np

# Procesar las imágenes de referencia para Q, J, y K
reference_image_Q = cv2.imread("static/images/figura_Q.png", cv2.IMREAD_GRAYSCALE)
#reference_image_J = cv2.imread("static/images/figura_J.jpg", cv2.IMREAD_GRAYSCALE)
#reference_image_K = cv2.imread("static/images/figura_K.jpg", cv2.IMREAD_GRAYSCALE)

# Verificar si las imágenes se cargaron correctamente
if reference_image_Q is None: #or reference_image_J is None or reference_image_K is None:
    raise FileNotFoundError("Una o más imágenes de referencia no se pudieron cargar. Verifica las rutas.")

# Preprocesar las imágenes (suavizado y detección de bordes)
reference_Q_blurred = cv2.GaussianBlur(reference_image_Q, (5, 5), 0)
#reference_J_blurred = cv2.GaussianBlur(reference_image_J, (5, 5), 0)
#reference_K_blurred = cv2.GaussianBlur(reference_image_K, (5, 5), 0)

reference_Q_edges = cv2.Canny(reference_Q_blurred, 50, 150)
#reference_J_edges = cv2.Canny(reference_J_blurred, 50, 150)
#reference_K_edges = cv2.Canny(reference_K_blurred, 50, 150)

# Encontrar contornos en las imágenes procesadas
contours_Q, _ = cv2.findContours(reference_Q_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#contours_J, _ = cv2.findContours(reference_J_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#contours_K, _ = cv2.findContours(reference_K_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Seleccionar el contorno más grande como referencia para cada figura
reference_Q = max(contours_Q, key=cv2.contourArea) if contours_Q else None
#reference_J = max(contours_J, key=cv2.contourArea) if contours_J else None
#reference_K = max(contours_K, key=cv2.contourArea) if contours_K else None

# Verificar si se encontraron contornos válidos
if reference_Q is None: #or reference_J is None or reference_K is None:
    raise ValueError("No se encontraron contornos válidos en una o más imágenes de referencia.")

# Dibujar y mostrar los contornos detectados para cada figura
def show_contour(image, contour, title):
    # Crear una copia de la imagen original para dibujar el contorno
    image_with_contour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contour, [contour], -1, (0, 255, 0), 2)  # Contorno en verde
    cv2.imshow(title, image_with_contour)

# Mostrar los contornos detectados
show_contour(reference_image_Q, reference_Q, "Contorno Detectado - Q")

# Guardar los contornos en un archivo
np.savez("contornos_referencias.npz", Q=reference_Q)#, J=reference_J, K=reference_K)
print("Contornos guardados exitosamente en 'contornos_referencias.npz'")


