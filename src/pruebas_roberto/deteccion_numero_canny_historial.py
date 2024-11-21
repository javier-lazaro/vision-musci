import cv2
import numpy as np
from colorDetection import ColorDetection
from roiExtractor import ROIExtractor


# Cargar los valores desde el archivo npz
with np.load('./static/npz/calibration_data.npz') as data:
    loaded_mtx = data['camera_matrix']
    loaded_dist = data['dist_coeffs']

# Función para obtener la línea que pasa por dos puntos
def calculate_line(p1, p2):
    m = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else None
    if m is not None:
        b = p1[1] - m * p1[0]
    else:
        b = p1[0]  # Vertical line case
    return m, b

# Función para desplazar dos puntos una distancia N de forma paralela
def parallel_shift(p1, p2, distance):
    direction = np.array([p2[1] - p1[1], -(p2[0] - p1[0])])  # Perpendicular vector
    unit_direction = direction / np.linalg.norm(direction)
    shift_vector = unit_direction * distance
    return p1 + shift_vector, p2 + shift_vector

# Función para extraer los puntos en los que intersectan dos líneas
def find_intersection(m1, b1, m2, b2):
    # Check if lines are parallel (same slope)
    if m1 == m2:
        return None
    
    # Calculate intersection
    if m1 is not None and m2 is not None:
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
    elif m1 is None:
        x = b1
        y = m2 * x + b2
    elif m2 is None:
        x = b2
        y = m1 * x + b1
    return (int(np.round(x)), int(np.round(y)))

# Lectura de imagen en tiempo real
cap = cv2.VideoCapture(0)

# Crea una ventana llamada 'VentanaCartas'.
cv2.namedWindow('VentanaCartas')
cv2.namedWindow('VentanaThresh')
cv2.namedWindow('VentanaCanny')

# Lee el primer fotograma de la cámara.
success, frame = cap.read() # Succes indica si la lectura fue exitosa.

# Diccionario para almacenar las ventanas activas por carta
active_windows = {}

# Bucle para mostrar el video en tiempo real.
while success and cv2.waitKey(1) == -1: 

    # Corregir la distorsión del fotograma usando los parámetros cargados
    undistorted_frame = cv2.undistort(frame, loaded_mtx, loaded_dist, None, loaded_mtx)

    frame_with_lines = frame.copy()
    # Creamos una mascara basandonos en el frame original
    roi_frame = frame.copy()
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro Gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Aplicar un umbral binario para segmentar la imagen, para separar blanco de negro
    ret, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # Encontrar los contornos en la imagen umbralizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    # Ordenar los contornos por área de mayor a menor
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Eliminamos el contorno más grande, que representa el contorno de la propia ventana
    if contours:
        contours.pop(0)

    if len(contours) > 1:
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

        # Indicamos en la ventana el número de cartas que hay en la mesa
        cv2.putText(frame, "Numero de cartas: " + str(len(contours)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Si solo hay un contorno, no es necesario calcular la media y la desviación estándar, es la única carta
    else:
        cv2.putText(frame, "Numero de cartas: " + str(len(contours)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Obtener la cantidad de cartas actuales y actualizar las ventanas
    current_contour_ids = set(range(len(contours)))
    existing_contour_ids = set(active_windows.keys())

    # Cerrar ventanas de contornos que ya no están presentes
    for contour_id in existing_contour_ids - current_contour_ids:
        cv2.destroyWindow(active_windows[contour_id])
        del active_windows[contour_id]
    
    # Dibujar y mostrar la región de interés de cada carta detectada
    for idx, c in enumerate(contours):
        # Encontrar el área mínima
        rect = cv2.minAreaRect(c)
        # Calcular las coordenadas del rectángulo de área mínima
        box = cv2.boxPoints(rect)
        # Normalizar las coordenadas a enteros
        box = np.int32(box)
        #print("Box: ", box)
        # dibujar contornos
        cv2.drawContours(thresh, [box], 0, (0,0, 255), 3)
        cv2.drawContours(frame, [box], 0, (0,0, 255), 3)

        # Extraer la región del box y mostrarla en una ventana separada
        x, y, w, h = cv2.boundingRect(box)
        box_region = frame[y:y+h, x:x+w]
        if box_region.size > 0:  # Verificar que el tamaño del contorno es válido
            window_name = f'Carta_{idx}'  # Nombre único para cada carta detectada
            active_windows[idx] = window_name
            #cv2.imshow(window_name, box_region)

            # Mostrar la carta detectada usando la clase ColorDetection
            #color_detection = ColorDetection(frame, box)
            #color_detection.draw_line_between_corners()  # Dibujar la línea entre las esquinas
            #color_detection.show_detected_card()

            # Posicionar la ventana en una ubicación diferente
            window_x = 100 + (idx % 5) * 600  # Espaciado horizontal entre ventanas aumentado
            window_y = 100 + (idx // 5) * 600  # Espaciado vertical entre filas aumentado
            cv2.moveWindow(window_name, window_x, window_y)

############################################################################################
# Extraer la región del box y mostrarla en una ventana separada
        # Extraer la región del box y mostrarla en una ventana separada
        x, y, w, h = cv2.boundingRect(box)
        box_region = thresh[y:y + h, x:x + w]

        # Historial para estabilizar el número de contornos
        contour_history = []  # Lista para almacenar los números detectados
        history_length = 5  # Longitud del historial para estabilizar

        if box_region.size > 0:  # Verificar que el tamaño del contorno es válido
            
            # Verificar el número de canales de la imagen
            if len(box_region.shape) == 3 and box_region.shape[2] == 3:
                # Convertir la imagen a escala de grises si tiene 3 canales (BGR)
                box_gray = cv2.cvtColor(box_region, cv2.COLOR_BGR2GRAY)
            else:
                # Si la imagen ya está en escala de grises, no es necesario convertirla
                box_gray = box_region

            # Aplicar un filtro Gaussiano para suavizar la imagen
            box_blurred = cv2.GaussianBlur(box_gray, (5, 5), 0)

            # Aplicar el detector de bordes Canny
            canny_edges = cv2.Canny(box_blurred, threshold1=50, threshold2=150)

            # Aplicar una operación de cerrado morfológico para asegurar que los bordes estén bien cerrados
            kernel = np.ones((5, 5), np.uint8)
            closed = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)

            # Mostramos la ventana Canny
            cv2.imshow('VentanaCanny', closed)

            # Encontrar los contornos en la imagen de la región del box
            inner_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calcular el centro de la región del box y definir un rectángulo
            height, width = box_region.shape[:2]
            rect_width = int(width * 0.70)  # Ajustar el tamaño del rectángulo al 75% del ancho
            rect_height = int(height * 0.85)  # Ajustar el tamaño del rectángulo al 75% de la altura

            # Definir el rectángulo central (top-left y bottom-right)
            top_left = (width // 2 - rect_width // 2, height // 2 - rect_height // 2)
            bottom_right = (width // 2 + rect_width // 2, height // 2 + rect_height // 2)

            # Dibujar el rectángulo en la misma imagen donde se dibujarán los contornos
            #cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.rectangle(frame, (x + top_left[0], y + top_left[1]), (x + bottom_right[0], y + bottom_right[1]), (0, 255, 0), 2)

            # Contar cuántos contornos están completamente dentro del rectángulo
            count = 0
            for contour in inner_contours:
                inside_rectangle = True
                for point in contour:
                    if not (top_left[0] <= point[0][0] <= bottom_right[0] and top_left[1] <= point[0][1] <= bottom_right[1]):
                        inside_rectangle = False
                        break
                if inside_rectangle:
                    count += 1
                    contour[:, 0, 0] += x  # Desplazamiento en X
                    contour[:, 0, 1] += y  # Desplazamiento en Y
                    # Dibujar los contornos que están dentro del rectángulo
                    cv2.drawContours(frame, [contour], -1, (0, 0, 255), 3)
            
            # Actualizar el historial de contornos
            contour_history.append(count)
            if len(contour_history) > history_length:
                contour_history.pop(0)  # Eliminar el valor más antiguo si excede el tamaño del historial

            # Calcular un valor estable (por ejemplo, la mediana del historial)
            stable_count = int(np.median(contour_history))

            # Dibujar texto con el valor estable
            text = f'Numero: {stable_count}'
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_width, text_height = text_size

            # Coordenadas para el texto y el punto verde
            text_x = x + top_left[0] + (rect_width // 2) - (text_width // 2)
            text_y = y + top_left[1] - 10  # Espacio por encima del rectángulo

            # Dibujar el texto
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA)

################################################################################################################

        # Lo que hacemos en las siguientes lineas es identificar el punto del centro del área del objeto identificado
        # Función que utilizamos
        M = cv2.moments(c) 

        # Se ha puesto este if ya que se hace una división y si el denominador fuera 0 sería infinito lo cual daría error
        # Por lo tanto si es 0 se le asigna el valor de 1
        if(M['m00'] == 0): M['m00'] = 1 
        x = int(M['m10']/M['m00']) # Coordenada x del punto
        y = int(M['m01']/M['m00']) # Coordenada y del punto

        # Para dibujar el círculo del punto central conforme se mueve en la imagen utilizamos la siguiente función
        cv2.circle(frame, # Imagen que se va a dibujar
                    (x,y), # Coordenadas donde se va a dibujar
                    15, # Radio del circulo
                    (0,255,0), # Color con el que se va a dibujar, verde.
                    -1)
        
        # Para indicar las coordenadas que tiene el objeto en la imagen conforme se mueve: 
        font = cv2.FONT_HERSHEY_SIMPLEX # Con esto declaramos la fuente / tipografía del texto
        cv2.putText(frame, # Imagen que se va a dibujar
                    '{}, {}'.format(x,y), # Texto de las coordendas que se van a indicar, 'x' y 'y' entre las llaves. 
                    (x+10, y), # Ubicación con respecto al punto, mas  a la derecha obviamente para que no se solape
                    font, # Fuente que se había definido antes
                    1.10, # Tamaño del texto
                    (255,0,0), # Color del texto
                    2, # Grosor del texto (Negrita)
                    cv2.LINE_AA)
        
        ### EXTRACION DE ROIs ###

        roi_extractor = ROIExtractor()
        roi_list = roi_extractor.extract_rois(box)
        for roi in roi_list:
            cv2.polylines(frame_with_lines, [np.array(roi, np.int32).reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)

        if len(roi_list) > 0:
            for roi in roi_list:                   
                # Fill the polygon in the mask
                roi_np = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [roi_np], color=255)
                # Extract the ROI using the mask
                roi_masked = cv2.bitwise_and(roi_frame, roi_frame, mask=mask)
                # Optional: Crop the bounding rectangle for simpler processing (if needed)
                #x, y, w, h = cv2.boundingRect(roi_np)
                #cropped_roi = roi_masked[y:y+h, x:x+w]
        else:
            roi_masked = cv2.bitwise_and(roi_frame, roi_frame, mask=mask)


    # Display the results
    #cv2.imshow("Mask", mask)
    #cv2.imshow("ROI", roi_masked)
    #cv2.imshow("Cropped ROI", cropped_roi)

    # Next steps: 
    # 1. Extraer tambien la ROI central, no solo las laterales
    # 2. Hacer un crop extra de las ROIs laterales (igual no hace falta)
    # 3. Crear el método de busqueda de color más común (probs con un threshold) para aplicar a cada ROI extraida
    # 4. Aplicar un filtro (Canny??) para buscar los contornos en cada ROI y sacar numero y palo

        
    # Display the frame with lines and points
    #cv2.imshow("Frame with Lines and Points", frame_with_lines)

    
        
    cv2.imshow('VentanaCartas', frame)  # Muestra el fotograma actual en la ventana.
    cv2.imshow('VentanaThresh', thresh)  # Muestra el fotograma actual en la ventana.

    success, frame = cap.read()  # Lee el siguiente fotograma de la cámara. 

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()