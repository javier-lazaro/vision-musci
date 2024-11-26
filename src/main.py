import cv2
import numpy as np
from colorDetection import ColorDetection
from roiExtractor import ROIExtractor
import os

# Cargar los valores desde el archivo npz
current_dir = os.path.dirname(os.path.abspath(__file__))
calibration_path = os.path.join(current_dir, '../static/npz/calibration_data.npz')

# Verificar si el archivo existe antes de intentar cargarlo
if os.path.exists(calibration_path):
    with np.load(calibration_path) as data:
        loaded_mtx = data['camera_matrix']
        loaded_dist = data['dist_coeffs']
else:
    raise FileNotFoundError(f"El archivo de calibración no se encuentra en la ruta especificada: {calibration_path}")

# Función para desplazar dos puntos una distancia N de forma paralela
def parallel_shift(p1, p2, distance):
    direction = np.array([p2[1] - p1[1], -(p2[0] - p1[0])])  # Perpendicular vector
    unit_direction = direction / np.linalg.norm(direction)
    shift_vector = unit_direction * distance
    return p1 + shift_vector, p2 + shift_vector

def ordenar_puntos(box):
    # Convertimos a un array numpy con 4 filas y 2 columnas
    puntos = np.array(box)

    # Calculamos el centroide (promedio de las coordenadas X y Y)
    centroide = np.mean(puntos, axis=0)

    # Clasificamos los puntos con respecto al centroide (orden en el sentido de las agujas del reloj)
    puntos_ordenados = sorted(puntos, key=lambda p: (np.arctan2(p[1] - centroide[1], p[0] - centroide[0])))

    # Convertimos a numpy array en el orden adecuado
    puntos_ordenados = np.array(puntos_ordenados, dtype="float32")

    # Identificar cuál es el lado largo y cuál es el corto
    d1 = np.linalg.norm(puntos_ordenados[0] - puntos_ordenados[1])  # Distancia entre punto 0 y 1
    d2 = np.linalg.norm(puntos_ordenados[1] - puntos_ordenados[2])  # Distancia entre punto 1 y 2

    if d1 > d2:
        # Si el lado entre los puntos 0 y 1 es el más largo, debemos reorganizar los puntos
        # para asegurarnos de que el lado largo siempre esté representado como la "altura"
        puntos_ordenados = np.roll(puntos_ordenados, -1, axis=0)  # Rotar los puntos en sentido antihorario

    return puntos_ordenados

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
cv2.namedWindow('Frame with Lines and Points')

# Lee el primer fotograma de la cámara.
success, frame = cap.read() # Succes indica si la lectura fue exitosa.

# Diccionario para almacenar las ventanas activas por carta
active_windows = {}

# Bucle para mostrar el video en tiempo real.
while success and cv2.waitKey(1) == -1: 

    # Corregir la distorsión del fotograma usando los parámetros cargados
    undistorted_frame = cv2.undistort(frame, loaded_mtx, loaded_dist, None, loaded_mtx)

    frame_with_lines = undistorted_frame.copy()
    # Creamos una mascara basandonos en el frame original
    roi_frame = undistorted_frame.copy()
    #mask = np.zeros(undistorted_frame.shape[:2], dtype=np.uint8)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

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
        cv2.putText(undistorted_frame, "Numero de cartas: " + str(len(contours)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Si solo hay un contorno, no es necesario calcular la media y la desviación estándar, es la única carta
    else:
        cv2.putText(undistorted_frame, "Numero de cartas: " + str(len(contours)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Obtener la cantidad de cartas actuales y actualizar las ventanas
    current_contour_ids = set(range(len(contours)))
    existing_contour_ids = set(active_windows.keys())

    # Cerrar ventanas de contornos que ya no están presentes
    for contour_id in existing_contour_ids - current_contour_ids:
        if contour_id in active_windows:
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

        # Utilizar esta función para ordenar los puntos antes de definir `src_pts`
        box = ordenar_puntos(box)

        # Dibujar contornos
        #cv2.drawContours(thresh, [box], 0, (0,0, 255), 3)
        #cv2.drawContours(undistorted_frame, [box], 0, (0,0, 255), 3)

        # Obtener dimensiones reales del rectángulo rotado
        width = int(rect[1][0])  # Ancho del rectángulo
        height = int(rect[1][1])  # Alto del rectángulo

        # Asegurarse de que height sea el lado mayor y width el lado menor
        if height < width:
            width, height = height, width
            
        # Definir puntos destino con el tamaño exacto del rectángulo
        dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")


        # Crear los puntos del rectángulo detectado
        src_pts = np.array(box, dtype="float32")
        
        # Calcular la transformación de perspectiva
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Aplicar la transformación a la imagen original
        warped = cv2.warpPerspective(undistorted_frame, M, (width, height))

        # Rotar la imagen resultante si es necesario para que siempre esté en orientación vertical
        if warped.shape[1] > warped.shape[0]:  # Si el ancho es mayor que la altura, significa que está horizontal
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        # Escalar la ventana para que se ajuste mejor al tamaño original
        escala = 1.5  # Factor de escala para aumentar el tamaño de la ventana
        scaled_width = int(width * escala)
        scaled_height = int(height * escala)
        warped_resized = cv2.resize(warped, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
        
        # Aplicar un threshold al área recortada (con fondo negro)
        #gray_card = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)  # Convertir la carta a escala de grises
        #_, card_thresh = cv2.threshold(gray_card, 127, 255, cv2.THRESH_BINARY)  # Aplicar threshold binario

        # Mantener el fondo negro
        #warped[np.where(card_thresh == 0)] = [0, 0, 0]

        # Definir una sub-región central de la carta para evitar los bordes
        h, w = warped.shape[:2]
        offset = 0  # Porcentaje para reducir el ROI a un 60% del área original
        x_start = int(w * offset)
        y_start = int(h * offset)
        x_end = int(w * (1 - offset))
        y_end = int(h * (1 - offset))
        warped_central = warped[y_start:y_end, x_start:x_end]

        # Convertir la sub-región de la carta a espacio de color HSV
        hsv_card = cv2.cvtColor(warped_central, cv2.COLOR_BGR2HSV)
        
        # Detectar si la carta es roja o negra
        # Convertir la carta a espacio de color HSV
        hsv_card = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

        # Definir los límites del color rojo en HSV
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        # Definir los límites del color amarillo en HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Crear máscara para detectar rojos
        mask_red1 = cv2.inRange(hsv_card, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_card, lower_red2, upper_red2)
        mask_red = mask_red1 + mask_red2
        mask_yellow = cv2.inRange(hsv_card, lower_yellow, upper_yellow)

        # Calcular la cantidad de píxeles rojos
        red_pixels = np.sum(mask_red > 0)

        # Calcular la cantidad de píxeles oscuros (negros) en escala de grises
        gray_card_central = cv2.cvtColor(warped_central, cv2.COLOR_BGR2GRAY)
        mask_black = cv2.inRange(gray_card_central, 0, 80)
        #black_pixels = np.sum(gray_card_central < 80)

        # Calcular la cantidad de píxeles para cada color
        total_pixels = hsv_card.shape[0] * hsv_card.shape[1]
        red_pixels = np.sum(mask_red > 0)
        yellow_pixels = np.sum(mask_yellow > 0)
        black_pixels = np.sum(mask_black > 0)

        # Calcular el porcentaje de cada color
        red_percentage = (red_pixels / total_pixels) * 100
        yellow_percentage = (yellow_pixels / total_pixels) * 100
        black_percentage = (black_pixels / total_pixels) * 100

        # Determinar el color predominante
        color_label = "Negra"
        if red_percentage > yellow_percentage and red_percentage > black_percentage:
            color_label = "Roja"
        elif yellow_percentage > red_percentage and yellow_percentage > black_percentage:
            color_label = "Amarilla"


        # Añadir la etiqueta de porcentajes y el color sobre la carta detectada
        cv2.putText(warped_resized, f"Color: {color_label}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(warped_resized, f"Roja: {red_percentage:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(warped_resized, f"Amarilla: {yellow_percentage:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(warped_resized, f"Negra: {black_percentage:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Mostrar la región recortada en una ventana con tamaño ajustado
        window_name = f'Carta_Rotada_{idx}'
        cv2.imshow(window_name, warped_resized)

        # Actualizar el diccionario de ventanas activas
        active_windows[idx] = window_name
        
        # Mover las ventanas de las cartas a la parte inferior derecha
        cv2.moveWindow(window_name, 650 + (idx % 5) * (scaled_width + 10), 500 + (idx // 5) * (scaled_height + 10))

        
        # Extraer la región del box y mostrarla en una ventana separada
        #x, y, w, h = cv2.boundingRect(box)
        #box_region = undistorted_frame[y:y+h, x:x+w]
        #if box_region.size > 0:  # Verificar que el tamaño del contorno es válido
        #    window_name = f'Carta_{idx}'  # Nombre único para cada carta detectada
        #    active_windows[idx] = window_name
        #    cv2.imshow(window_name, box_region)

            # Mostrar la carta detectada usando la clase ColorDetection
            #color_detection = ColorDetection(frame, box)
            #color_detection.draw_line_between_corners()  # Dibujar la línea entre las esquinas
            #color_detection.show_detected_card()

            # Posicionar la ventana en una ubicación diferente
            #window_x = 100 + (idx % 5) * 600  # Espaciado horizontal entre ventanas aumentado
            #window_y = 100 + (idx // 5) * 600  # Espaciado vertical entre filas aumentado
            #cv2.moveWindow(window_name, window_x, window_y)

        # Lo que hacemos en las siguientes lineas es identificar el punto del centro del área del objeto identificado
        # Función que utilizamos
        M = cv2.moments(c) 

        # Se ha puesto este if ya que se hace una división y si el denominador fuera 0 sería infinito lo cual daría error
        # Por lo tanto si es 0 se le asigna el valor de 1
        if(M['m00'] == 0): M['m00'] = 1 
        x = int(M['m10']/M['m00']) # Coordenada x del punto
        y = int(M['m01']/M['m00']) # Coordenada y del punto

        # Para dibujar el círculo del punto central conforme se mueve en la imagen utilizamos la siguiente función
        cv2.circle(undistorted_frame, # Imagen que se va a dibujar
                    (x,y), # Coordenadas donde se va a dibujar
                    15, # Radio del circulo
                    (0,255,0), # Color con el que se va a dibujar, verde.
                    -1)
        
        # Para indicar las coordenadas que tiene el objeto en la imagen conforme se mueve: 
        font = cv2.FONT_HERSHEY_SIMPLEX # Con esto declaramos la fuente / tipografía del texto
        cv2.putText(undistorted_frame, # Imagen que se va a dibujar
                    '{}, {}'.format(x,y), # Texto de las coordendas que se van a indicar, 'x' y 'y' entre las llaves. 
                    (x+10, y), # Ubicación con respecto al punto, mas  a la derecha obviamente para que no se solape
                    font, # Fuente que se había definido antes
                    1.10, # Tamaño del texto
                    (255,0,0), # Color del texto
                    2, # Grosor del texto (Negrita)
                    cv2.LINE_AA)
        
        ### EXTRACION DE ROIs ###

        #roi_extractor = ROIExtractor()
        #roi_list = roi_extractor.extract_rois(box)
        #for roi in roi_list:
        #    cv2.polylines(frame_with_lines, [np.array(roi, np.int32).reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
        #
        #if len(roi_list) > 0:
        #    for roi in roi_list:                   
        #        # Fill the polygon in the mask
        #        roi_np = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
        #        cv2.fillPoly(mask, [roi_np], color=255)
        #        # Extract the ROI using the mask
        #        roi_masked = cv2.bitwise_and(roi_frame, roi_frame, mask=mask)
        #        # Optional: Crop the bounding rectangle for simpler processing (if needed)
        #        #x, y, w, h = cv2.boundingRect(roi_np)
        #        #cropped_roi = roi_masked[y:y+h, x:x+w]
        #else:
        #    roi_masked = cv2.bitwise_and(roi_frame, roi_frame, mask=mask)


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
    cv2.imshow('VentanaCartas', undistorted_frame)  # Muestra el fotograma actual en la ventana.
    cv2.moveWindow('VentanaCartas', 0, 0) # Parte superior izquierda

    cv2.imshow("Frame with Lines and Points", frame_with_lines)
    cv2.moveWindow("Frame with Lines and Points", 650, 0) # Parte superior derecha

    cv2.imshow('VentanaThresh', thresh)  # Muestra el fotograma actual en la ventana.
    cv2.moveWindow('VentanaThresh', 0,500) # Parte inferior izquierda

    success, frame = cap.read()  # Lee el siguiente fotograma de la cámara. 

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()