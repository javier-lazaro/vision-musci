import cv2
import numpy as np
from utils.real_time import detector_numero as dn
from utils.real_time import detector_color as dc
from utils.real_time.detector_figuras import FigureDetector

# Cargar los valores de calibración de la cámara
with np.load('../static/npz/calibration_data.npz') as data:
    loaded_mtx = data['camera_matrix']
    loaded_dist = data['dist_coeffs']

# Lectura de imagen en tiempo real
cap = cv2.VideoCapture(0)

# Lectura del primer fotograma de la cámara
success, frame = cap.read()

# Diccionario para almacenar las ventanas activas por carta
active_windows = {}

# Variables para activar/desactivar la detección de figuras, colores, números y calibración de la cámara
yolo_detector,color_detector, number_detector, calibracion = False, False, False, False

# Bucle para mostrar el video en tiempo real.
while success:

    # Lógica de la detección de teclas
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):  # Presiona 'ñ' para activar la calibración
        calibracion = not calibracion
    if key == ord('c'):  # Presiona 'c' para activar el detector de color
        color_detector = not color_detector
    if key == ord('n'):  # Presiona 'n' para activar el detector de número
        number_detector = not number_detector
    if key == ord('y'):  # Presiona 'y' para activar el detector de texto
        yolo_detector = not yolo_detector
    if key == ord('q'):  # Presiona 'q' para salir del bucle
        break

    # Creación de una ventana negra para los mensajes
    message_frame = np.zeros((frame.shape[0], int(frame.shape[1] * 1.5), 3), dtype=np.uint8)

    calibration_text = (0, 255, 0) if calibracion else (0, 0, 255)
    cv2.putText(message_frame, "P: Calibracion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, calibration_text, 2, cv2.LINE_AA)

    # Mostrar cada opción con color dinámico según su estado
    color_text = (0, 255, 0) if color_detector else (0, 0, 255)
    cv2.putText(message_frame, "C: Detector de color", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv2.LINE_AA)

    number_text = (0, 255, 0) if number_detector else (0, 0, 255)
    cv2.putText(message_frame, "N: Detector de numero", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, number_text, 2, cv2.LINE_AA)

    yolo_text = (0, 255, 0) if yolo_detector else (0, 0, 255)
    cv2.putText(message_frame, "Y: Detector de figuras", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, yolo_text, 2, cv2.LINE_AA)

    # Activar la calibración en función de si se pulso o no la tecla correspondiente
    if calibracion:
        # Corregir la distorsión del fotograma usando los parámetros cargados
        frame = cv2.undistort(frame, loaded_mtx, loaded_dist, None, loaded_mtx)

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

    # Eliminar el contorno más grande, que representa el contorno de la propia ventana
    if contours:
        contours.pop(0)

    # Comprobae que haya más de un contorno
    if len(contours) > 1:

        # Calcular la media de aera de los contornos
        media = 0
        for contour in contours:
            media += cv2.contourArea(contour)
        media = media / len(contours)

        # Calcular la desviación estandar
        desviacion = 0
        for contour in contours:
            desviacion += (cv2.contourArea(contour) - media) ** 2
        desviacion = (desviacion / len(contours)) ** 0.5

        # Eliminar los contornos que no sean cartas
        contours = [contour for contour in contours if cv2.contourArea(contour) > media + 2 * desviacion]

        # Indicar en la ventana el número de cartas que hay en la mesa
        cv2.putText(frame, "Numero de cartas: " + str(len(contours)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Si solo hay un contorno, no es necesario calcular la media y la desviación estándar, es la única carta
    else:
        cv2.putText(frame, "Numero de cartas: " + str(len(contours)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Obtener la cantidad de cartas actuales y actualizamos las ventanas
    current_contour_ids = set(range(len(contours)))
    existing_contour_ids = set(active_windows.keys())

    # Cerrar ventanas de contornos que ya no están presentes
    for contour_id in existing_contour_ids - current_contour_ids:
        if contour_id in active_windows:  # Verifica que la ventana exista en el diccionario
            try:
                cv2.destroyWindow(active_windows[contour_id])  # Intenta destruir la ventana
            except cv2.error as e:
                print(f"Error al cerrar la ventana {active_windows[contour_id]}: {e}")
            del active_windows[contour_id]
    
    # Dibujar y mostrar la región de interés de cada carta detectada
    for idx, c in enumerate(contours):

        # Crear una copia del frame para no modificar el original
        color_detection_frame = frame.copy()

        # Encontrar el área mínima
        rect = cv2.minAreaRect(c)

        # Calcular las coordenadas del rectángulo de área mínima
        box = cv2.boxPoints(rect)

        # Normalizar las coordenadas a enteros
        box = np.int32(box)

        # Dibujar contornos
        cv2.drawContours(thresh, [box], 0, (0,0, 255), 3)
        cv2.drawContours(frame, [box], 0, (0,0, 255), 3)

        # Extraer la región del box y mostrarla en una ventana separada
        x, y, w, h = cv2.boundingRect(box)
        box_region = frame[y:y+h, x:x+w]

        # Si la región del box es válida
        if box_region.size > 0: 

            # Si activamos el detector de color
            if color_detector:

                # Detectar el color de la carta
                figura, color_label, warped_resized = dc.detectar_color_carta(box, idx, color_detection_frame, active_windows)
                
                # Si activamos el detector de número
                if number_detector:

                    # Detectar el número de la carta
                    dn.process_card_box(box, frame, thresh, figura)

                    # Si activamos el Yolo
                    if yolo_detector:

                        # Activar la detección por Yolo
                        df = FigureDetector()

                        # Si figura es True entonces devolvemos ambos labels, si no solo la figura
                        if figura:
                            figure_label, letter_label = df.detectar_figuras(warped_resized, color=color_label, figura=figura)
                            text = f"Letra: {letter_label} Figura: {figure_label}"
                        else: 
                            figure_label = df.detectar_figuras(warped_resized, color=color_label)
                            text = f"Figura: {figure_label}"
                    
                        # Obtener el tamaño del texto
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                        # Calcular la posición en la parte superior derecha
                        text_x = 10
                        text_y = 70  # Mantiene la altura del texto

                        # Dibujar el texto
                        cv2.putText(
                            warped_resized, text,
                            (text_x, text_y), font, font_scale, 
                            (191, 4, 255), thickness, cv2.LINE_AA
                        )

                        # Mostrar la región recortada en una ventana con tamaño ajustado
                        window_name = f'Carta_Rotada_{idx}'
                        cv2.imshow(window_name, warped_resized)

        # Se calculan los momentos de la imagen
        M = cv2.moments(c) 

        # Se calculan las coordenadas del punto central del objeto 
        if(M['m00'] == 0): M['m00'] = 1  # Si el denominador es 0, se le asigna 1, para evitar errores de división
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00']) 
        
        # Dibujar un círculo en el centro del objeto
        cv2.circle(frame, (x,y), 15, (0,255,0), -1)
        
        # Dibujar las coordenadas del objeto
        cv2.putText(frame, '{}, {}'.format(x,y), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (235,0,27), 2, cv2.LINE_AA)

    # Escalar las ventanas
    scale_percent = 100  # Escala en porcentaje
    frame_width = int(frame.shape[1] * scale_percent / 100)
    frame_height = int(frame.shape[0] * scale_percent / 100)
    dim = (frame_width, frame_height)

    # Redimensionar los frames
    rescaled_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    rescaled_thresh = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)


    # Crear un canvas para acomodar todos los frames
    canvas_height = rescaled_frame.shape[0] + rescaled_thresh.shape[0]  # Altura máxima considerando message_frame
    canvas_width = rescaled_frame.shape[1] + message_frame.shape[1]  # Ancho total
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # Canvas blanco

    # Colocar rescaled_frame en la parte izquierda
    canvas[:rescaled_frame.shape[0], :rescaled_frame.shape[1]] = rescaled_frame

    # Convertir rescaled_thresh a BGR y colocarlo en la parte inferior izquierda
    rescaled_thresh_bgr = cv2.cvtColor(rescaled_thresh, cv2.COLOR_GRAY2BGR)
    thresh_y = canvas_height - rescaled_thresh_bgr.shape[0]
    canvas[thresh_y:, :rescaled_thresh_bgr.shape[1]] = rescaled_thresh_bgr

    # Colocar message_frame en la parte superior derecha
    message_x = canvas_width - message_frame.shape[1]
    canvas[:message_frame.shape[0], message_x:] = message_frame

    # Mostrar el canvas en una sola ventana
    cv2.imshow('Ventana Principal', canvas)

    # Lectura el siguiente fotograma de la cámara
    success, frame = cap.read() 

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()