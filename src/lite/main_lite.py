import cv2
import numpy as np
import detector_numero as dn
import os

# Obtener la ruta absoluta al archivo
base_dir = os.path.dirname(os.path.abspath(__file__))
calibration_data_path = os.path.join(base_dir, "static/calibration_data.npz")

# Cargar los valores de calibración de la cámara
with np.load(calibration_data_path) as data:
    loaded_mtx = data['camera_matrix']
    loaded_dist = data['dist_coeffs']

# Lectura de imagen en tiempo real
cap = cv2.VideoCapture(0)

# Creación de las ventanas necesarias
cv2.namedWindow('VentanaCartas')
cv2.namedWindow('VentanaThresh')

# Lectura del primer fotograma de la cámara
success, frame = cap.read()

# Diccionario para almacenar las ventanas activas por carta
active_windows = {}

# Variables para activar/desactivar la calibración de la cámara
calibracion = False

# Bucle para mostrar el video en tiempo real.
while success:

    # Lógica de la detección de teclas
    key = cv2.waitKey(1) & 0xFF
    if key == ord('ñ'):  # Presiona 'ñ' para activar la calibración
        calibracion = not calibracion
    if key == ord('q'):  # Presiona 'q' para salir del bucle
        break

    # Mostrar mensaje en la ventana adicional, en función de si la calibración está activada
    if calibracion:
        print(f"Calibración {'activada' if calibracion else 'desactivada'}")     

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
        cv2.putText(frame, "Numero de cartas: " + str(len(contours)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Si solo hay un contorno, no es necesario calcular la media y la desviación estándar, es la única carta
    else:
        cv2.putText(frame, "Numero de cartas: " + str(len(contours)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Obtener la cantidad de cartas actuales y actualizamos las ventanas
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

        # Dibujar contornos
        cv2.drawContours(thresh, [box], 0, (0,0, 255), 3)
        cv2.drawContours(frame, [box], 0, (0,0, 255), 3)

        # Extraer la región del box y mostrarla en una ventana separada
        x, y, w, h = cv2.boundingRect(box)
        box_region = frame[y:y+h, x:x+w]

        # Si la región del box es válida
        if box_region.size > 0: 

            # Crear una ventana con el nombre de la carta
            window_name = f'Carta_{idx}'  
            active_windows[idx] = window_name

            # Posicionar la ventana en una ubicación diferente
            window_x = 100 + (idx % 5) * 600  # Espaciado horizontal entre ventanas aumentado
            window_y = 100 + (idx // 5) * 600  # Espaciado vertical entre filas aumentado
            cv2.moveWindow(window_name, window_x, window_y)

            # Crear una copia del frame para no modificar el original
            color_detection_frame = frame.copy()

            # Detectar el número de la carta
            dn.process_card_box(box, frame, thresh, figura=False)

        # Se calculan los momentos de la imagen
        M = cv2.moments(c) 

        # Se calculan las coordenadas del punto central del objeto 
        if(M['m00'] == 0): M['m00'] = 1  # Si el denominador es 0, se le asigna 1, para evitar errores de división
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00']) 
        
        # Dibujar un círculo en el centro del objeto
        cv2.circle(frame, (x,y), 15, (0,255,0), -1)
        
        # Dibujar las coordenadas del objeto
        cv2.putText(frame, '{}, {}'.format(x,y), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.10, (255,0,0), 2, cv2.LINE_AA)

    # Mostrar las ventanas 
    cv2.imshow('VentanaCartas', frame)
    cv2.imshow('VentanaThresh', thresh) 

    # Lectura el siguiente fotograma de la cámara
    success, frame = cap.read() 

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()