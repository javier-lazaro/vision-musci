import cv2
import numpy as np

# Función para extraer contornos de referencia desde una imagen de carta
def process_reference_card(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Suavizar la imagen con un filtro Gaussiano
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Aplicar umbral binario inverso para separar blanco y negro
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # Encontrar los contornos en la imagen umbralizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar los contornos por área (de mayor a menor)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Eliminar el contorno más grande si representa el borde de la imagen
    if contours:
        contours.pop(0)

    # Filtrar los contornos para obtener solo cartas
    if len(contours) > 1:
        # Calcular la media y desviación estándar del área de los contornos
        areas = [cv2.contourArea(c) for c in contours]
        media = np.mean(areas)
        desviacion = np.std(areas)

        # Mantener contornos que sean significativamente más grandes
        contours = [c for c in contours if cv2.contourArea(c) > media + 2 * desviacion]

    # Si no se encontraron contornos válidos, detener
    if not contours:
        return None

    # Procesar la esquina superior izquierda (ROI) de la primera carta detectada
    largest_contour = contours[0]
    x, y, w, h = cv2.boundingRect(largest_contour)
    box_region = gray[y:y + h, x:x + w]

    # Extraer la esquina superior izquierda para identificar la figura
    roi_corner = box_region[0:h // 6, 0:w // 6]
    
    # Suavizar y detectar bordes en el ROI
    blurred = cv2.GaussianBlur(roi_corner, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Encontrar contornos en la ROI procesada
    roi_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Seleccionar el contorno más grande como referencia
    if roi_contours:
        return max(roi_contours, key=cv2.contourArea)
    return None


# Cargar imágenes de referencia
reference_image_Q = cv2.imread("static/images/figura_Q.jpg")
reference_image_K = cv2.imread("static/images/figura_K.jpg")
reference_image_J = cv2.imread("static/images/figura_J.jpg")

# Verificar si las imágenes de referencia se cargaron correctamente
if reference_image_Q is None or reference_image_K is None or reference_image_J is None:
    raise FileNotFoundError("Una o más imágenes de referencia no se pudieron cargar. Verifica las rutas.")

# Procesar cada imagen de referencia
reference_Q = process_reference_card(reference_image_Q)
reference_K = process_reference_card(reference_image_K)
reference_J = process_reference_card(reference_image_J)

# Verificar que se hayan encontrado contornos válidos
if reference_Q is None or reference_K is None or reference_J is None:
    raise ValueError("No se encontraron contornos válidos en una o más imágenes de referencia.")

# Guardar los contornos en un archivo para su uso posterior
np.savez("contornos_referencias.npz", Q=reference_Q, K=reference_K, J=reference_J)
print("Contornos de referencia guardados exitosamente en 'contornos_referencias.npz'.")

# Mostrar el contorno detectado tanto en la imagen completa como en la ROI
def show_contour(image, contour, title, position_x=0, position_y=0):
    # Crear una copia de la imagen original para dibujar el contorno completo
    image_with_contour = image.copy()
    if len(image_with_contour.shape) == 2:  # Si es en escala de grises, convertir a BGR
        image_with_contour = cv2.cvtColor(image_with_contour, cv2.COLOR_GRAY2BGR)

    # Dibujar el contorno completo en la imagen original
    cv2.drawContours(image_with_contour, [contour], -1, (0, 255, 0), 2)  # Contorno en verde

    # Calcular la región delimitada por el contorno (ROI)
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y + h, x:x + w]  # Recortar la ROI de la imagen original

    # Dibujar un rectángulo alrededor de la ROI en la imagen completa
    cv2.rectangle(image_with_contour, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Rectángulo en azul

    # Mostrar la imagen completa con el contorno
    cv2.imshow(f"{title} - Imagen Completa", image_with_contour)
    cv2.moveWindow(f"{title} - Imagen Completa", position_x, position_y)

    # Verificar si la ROI es válida
    if len(roi.shape) == 2:  # Si la ROI es en escala de grises, convertir a BGR
        roi_with_contour = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    else:
        roi_with_contour = roi.copy()

    # Dibujar el contorno ajustado a la ROI
    adjusted_contour = contour - contour.min(axis=0)  # Ajustar coordenadas relativas al ROI
    cv2.drawContours(roi_with_contour, [adjusted_contour], -1, (0, 0, 255), 2)  # Contorno en rojo

    # Mostrar la ROI con el contorno
    cv2.imshow(f"{title} - ROI", roi_with_contour)
    cv2.moveWindow(f"{title} - ROI", position_x, position_y + 300)  # Mostrar debajo de la imagen completa


# Mostrar contornos detectados en la imagen completa y en la ROI
show_contour(reference_image_Q, reference_Q, "Contorno Detectado - Q", position_x=100, position_y=100)
show_contour(reference_image_K, reference_K, "Contorno Detectado - K", position_x=500, position_y=100)
show_contour(reference_image_J, reference_J, "Contorno Detectado - J", position_x=900, position_y=100)

# Esperar a que se cierren las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()
