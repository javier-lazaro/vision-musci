import cv2
import numpy as np
from colorDetection import ColorDetection

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
cap = cv2.VideoCapture(1)

# Crea una ventana llamada 'VentanaCartas'.
cv2.namedWindow('VentanaCartas')
cv2.namedWindow('VentanaThresh')

# Lee el primer fotograma de la cámara.
success, frame = cap.read() # Succes indica si la lectura fue exitosa.

# Diccionario para almacenar las ventanas activas por carta
active_windows = {}

# Bucle para mostrar el video en tiempo real.
while success and cv2.waitKey(1) == -1: 

    frame_with_lines = frame.copy()

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
            cv2.imshow(window_name, box_region)

            # Mostrar la carta detectada usando la clase ColorDetection
            #color_detection = ColorDetection(frame, box)
            #color_detection.draw_line_between_corners()  # Dibujar la línea entre las esquinas
            #color_detection.show_detected_card()

            # Posicionar la ventana en una ubicación diferente
            window_x = 100 + (idx % 5) * 600  # Espaciado horizontal entre ventanas aumentado
            window_y = 100 + (idx // 5) * 600  # Espaciado vertical entre filas aumentado
            cv2.moveWindow(window_name, window_x, window_y)

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
        


        #### Obtencion de las ROI ####
        # Calculate the lengths of all four edges
        line_lengths = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
        shortest_length = min(line_lengths)
        longest_length = max(line_lengths)

        # Define separate distances for long and short lines
        distance_long = -round((shortest_length*0.2))  # Distance for long lines
        distance_short = -round((shortest_length*0.4))   # Distance for short lines

        # Initialize a copy of the frame for visualization
        
        parallel_edges = []

        # Calculate parallel lines for each edge and classify them by length ratio
        for i in range(4):
            p1, p2 = box[i], box[(i + 1) % 4]
            line_length = line_lengths[i]
            distance = distance_long if line_length >= 1.2 * shortest_length else distance_short
            shifted_p1, shifted_p2 = parallel_shift(p1, p2, distance)
            parallel_edges.append((shifted_p1, shifted_p2))

            # Draw original and parallel lines
            cv2.line(frame_with_lines, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 0), 2)
            cv2.line(frame_with_lines, tuple(shifted_p1.astype(int)), tuple(shifted_p2.astype(int)), (255, 0, 0), 2)

        # Find intersection points of the parallel lines with the original rectangle edges
        # Calculate intersections of the parallel lines with the original rectangle edges
        # Loop through each parallel edge and calculate intersections with both the rectangle and other parallel lines
        # Loop through each parallel edge and calculate intersections with both the rectangle and other parallel lines
        for i, (p1, p2) in enumerate(parallel_edges):
            m1, b1 = calculate_line(p1, p2)
            
            # Intersect with rectangle edges
            for j in range(4):
                rect_p1, rect_p2 = box[j], box[(j + 1) % 4]
                m2, b2 = calculate_line(rect_p1, rect_p2)
                
                # Calculate intersection point between parallel line and rectangle edge
                intersection_point = find_intersection(m1, b1, m2, b2)
                
                # Skip if no intersection (parallel lines) or point is too far
                if intersection_point is None or not (0 <= intersection_point[0] < frame_with_lines.shape[1] and 0 <= intersection_point[1] < frame_with_lines.shape[0]):
                    continue

                # Categorize and color the points based on position (e.g., top-left, bottom-right)
                if intersection_point[0] < np.mean(box[:, 0]) and intersection_point[1] < np.mean(box[:, 1]):
                    cv2.circle(frame_with_lines, intersection_point, 5, (0, 255, 255), -1)  # Yellow for top-left
                elif intersection_point[0] > np.mean(box[:, 0]) and intersection_point[1] > np.mean(box[:, 1]):
                    cv2.circle(frame_with_lines, intersection_point, 5, (0, 255, 0), -1)  # Green for bottom-right
                else:
                    cv2.circle(frame_with_lines, intersection_point, 5, (255, 0, 255), -1)  # Pink for other points

            # Intersect with other parallel lines (wrap last with first to complete the loop)
            p3, p4 = parallel_edges[(i + 1) % len(parallel_edges)]
            m2, b2 = calculate_line(p3, p4)
            
            # Calculate intersection point between consecutive parallel lines
            intersection_point = find_intersection(m1, b1, m2, b2)
            
            # Skip if no intersection (parallel lines) or point is too far
            if intersection_point is None or not (0 <= intersection_point[0] < frame_with_lines.shape[1] and 0 <= intersection_point[1] < frame_with_lines.shape[0]):
                continue
            
            # Categorize and color the points based on position (e.g., top-left, bottom-right)
            if intersection_point[0] < np.mean(box[:, 0]) and intersection_point[1] < np.mean(box[:, 1]):
                cv2.circle(frame_with_lines, intersection_point, 5, (0, 255, 255), -1)  # Yellow for top-left
            elif intersection_point[0] > np.mean(box[:, 0]) and intersection_point[1] > np.mean(box[:, 1]):
                cv2.circle(frame_with_lines, intersection_point, 5, (0, 255, 0), -1)  # Green for bottom-right
            else:
                cv2.circle(frame_with_lines, intersection_point, 5, (255, 0, 255), -1)  # Pink for other points

    # Display the frame with lines and points
    cv2.imshow("Frame with Lines and Points", frame_with_lines)


        #print(p1, p2)

        #frame = cv2.line(frame, p1, p2, (255, 0, 0), thickness=5)
        
    cv2.imshow('VentanaCartas', frame)  # Muestra el fotograma actual en la ventana.
    cv2.imshow('VentanaThresh', thresh)  # Muestra el fotograma actual en la ventana.

    success, frame = cap.read()  # Lee el siguiente fotograma de la cámara. 

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()