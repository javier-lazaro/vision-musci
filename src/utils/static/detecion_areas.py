import cv2
import numpy as np

# Función para extraer la ROI de referencia y calcular el área del contorno
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
        return None, None, None

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

    # Seleccionar el contorno más grande como referencia y calcular el área
    if roi_contours:
        largest_roi_contour = max(roi_contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_roi_contour)
        return roi_corner, contour_area, (x, y, w, h)
    return None, None, None

# Cargar imágenes de referencia
reference_image_Q = cv2.imread("static/images/figura_Q.jpg")
reference_image_K = cv2.imread("static/images/figura_K.jpg")
reference_image_J = cv2.imread("static/images/figura_J.jpg")

# Verificar si las imágenes de referencia se cargaron correctamente
if reference_image_Q is None or reference_image_K is None or reference_image_J is None:
    raise FileNotFoundError("Una o más imágenes de referencia no se pudieron cargar. Verifica las rutas.")

# Procesar cada imagen de referencia para obtener la ROI específica y el área del contorno
roi_corner_Q, area_Q, bounding_box_Q = process_reference_card(reference_image_Q)
roi_corner_K, area_K, bounding_box_K = process_reference_card(reference_image_K)
roi_corner_J, area_J, bounding_box_J = process_reference_card(reference_image_J)

# Verificar que se hayan encontrado ROIs y áreas válidas
if roi_corner_Q is None or roi_corner_K is None or roi_corner_J is None:
    raise ValueError("No se encontraron ROIs válidas en una o más imágenes de referencia.")

# Guardar las ROIs de referencia y las áreas de los contornos en un archivo para su uso posterior
# Almacena las ROIs de referencia Q, K, J y las áreas en un archivo npz
np.savez("./static/npz/areas_referencias.npz", Q=roi_corner_Q, K=roi_corner_K, J=roi_corner_J,
         area_Q=area_Q, area_K=area_K, area_J=area_J)
print("ROIs y áreas de referencia guardadas exitosamente en 'areas_referencias.npz'.")

# Mostrar las ROIs detectadas y el área del contorno
def show_roi(roi_corner, area, title, position_x=0, position_y=0):
    # Mostrar la ROI específica (esquina superior izquierda)
    cv2.imshow(f"{title} - ROI Esquina Superior Izquierda", roi_corner)
    cv2.moveWindow(f"{title} - ROI Esquina Superior Izquierda", position_x, position_y)

    # Imprimir el área del contorno
    print(f"Área del contorno de {title}: {area} píxeles")

# Mostrar las ROIs específicas para Q, K y J
show_roi(roi_corner_Q, area_Q, "ROI Detectada - Q", position_x=100, position_y=100)
show_roi(roi_corner_K, area_K, "ROI Detectada - K", position_x=600, position_y=100)
show_roi(roi_corner_J, area_J, "ROI Detectada - J", position_x=1100, position_y=100)

# Esperar a que se cierren las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()
