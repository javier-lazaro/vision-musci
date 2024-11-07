import cv2
import numpy as np

# Lectura de imagen en tiempo real
cap = cv2.VideoCapture(0)

# Crea una ventana llamada 'VentanaCartas'.
cv2.namedWindow('VentanaCartas')
cv2.namedWindow('VentanaCartas2')

# Lee el primer fotograma de la cámara.
success, frame = cap.read() # Succes indica si la lectura fue exitosa.

# Contador para las imágenes
contour_image_counter = 0

"""
# Diccionario para almacenar las ventanas de cada contorno
contour_windows = {}
"""

# Diccionario para almacenar las ventanas activas por carta
active_windows = {}

# Bucle para mostrar el video en tiempo real.
while success and cv2.waitKey(1) == -1: 
    
    #cv2.imshow('VentanaCartas', frame)  # Muestra el fotograma actual en la ventana.

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro Gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Aplicar un umbral binario para segmentar la imagen
    ret, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # Encontrar los contornos en la imagen umbralizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    # Ordenar los contornos por área de mayor a menor
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Eliminamos los contornos pequeños
    #contours = [contour for contour in contours if cv2.contourArea(contour) > 1000 ] 

    # Eliminamos el contorno más grande (puede que no lo necesitemos)
    if contours:
        contours.pop(0)

    if contours:
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

    # TODO: Buscar una forma de determinar la forma que tiene el contorno, es decir que sea un elemento rectangular
    # TODO: Calcaulo de proporción
    
    # Creamos una copia del frame
    frame_copy = frame.copy()

    """
    # Obtener los identificadores actuales de contornos
    current_contour_ids = set(range(len(contours)))
    existing_contour_ids = set(contour_windows.keys())

    # Cerrar ventanas de contornos que ya no están presentes
    for contour_id in existing_contour_ids - current_contour_ids:
        cv2.destroyWindow(contour_windows[contour_id])
        del contour_windows[contour_id]
    """

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
        # dibujar contornos
        cv2.drawContours(thresh, [box], 0, (0,0, 255), 3)
        cv2.drawContours(frame, [box], 0, (0,0, 255), 3)

        # Extraer la región del box y mostrarla en una ventana separada
        # Extraer la región del box y mostrarla en una ventana separada
        x, y, w, h = cv2.boundingRect(box)
        box_region = frame[y:y+h, x:x+w]
        if box_region.size > 0:  # Verificar que el tamaño del contorno es válido
            window_name = f'Carta_{idx}'  # Nombre único para cada carta detectada
            active_windows[idx] = window_name
            cv2.imshow(window_name, box_region)

            # Posicionar la ventana en una ubicación diferente
            window_x = 100 + (idx % 5) * 600  # Espaciado horizontal entre ventanas aumentado
            window_y = 100 + (idx // 5) * 600  # Espaciado vertical entre filas aumentado
            cv2.moveWindow(window_name, window_x, window_y)

        ###########################################################################

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
                    7, # Radio del circulo
                    (0,255,0), # Color con el que se va a dibujar, verde.
                    -1)
        
        # Para indicar las coordenadas que tiene el objeto en la imagen conforme se mueve: 
        font = cv2.FONT_HERSHEY_SIMPLEX # Con esto declaramos la fuente / tipografía del texto
        cv2.putText(frame, # Imagen que se va a dibujar
                    '{}, {}'.format(x,y), # Texto de las coordendas que se van a indicar, 'x' y 'y' entre las llaves. 
                    (x+10, y), # Ubicación con respecto al punto, mas  a la derecha obviamente para que no se solape
                    font, # Fuente que se había definido antes
                    0.75, # Grosor del texto
                    (0,255,0), # Color del texto
                    1, # Tamaño del texto
                    cv2.LINE_AA)

        ###########################################################################

        #cv2.drawContours(frame, contours, -1, (0, 255, 0), 2) 
        
    """
    # Dibujar los contornos en la imagen original
    if len(current_contour_ids) > len(existing_contour_ids):  # Solo si hay nuevas cartas
        for contour_id, c in enumerate(contours):
            if contour_id in contour_windows:
                continue  # Si la ventana ya existe, saltar

            # Encontrar el área mínima
            rect = cv2.minAreaRect(c)
            # Calcular las coordenadas del rectángulo de área mínima
            box = cv2.boxPoints(rect)
            # Normalizar las coordenadas a enteros
            box = np.int32(box)

            # Crear una imagen con el contorno actual
            x, y, w, h = cv2.boundingRect(c)
            if w > 0 and h > 0:  # Verificar que el tamaño del contorno es válido
                contour_image = frame[y:y+h, x:x+w]
                
                # Mostrar la imagen del contorno en una ventana
                window_name = f"Contour_{contour_id}"
                contour_windows[contour_id] = window_name
                cv2.imshow(window_name, contour_image)
    """
        
    #cv2.drawContours(frame, contours, -1, (0, 255, 0), 2) 
    cv2.imshow('VentanaCartas', frame)  # Muestra el fotograma actual en la ventana.
    cv2.imshow('VentanaCartas2', thresh)  # Muestra el fotograma actual en la ventana.
    

    success, frame = cap.read()  # Lee el siguiente fotograma de la cámara. 

# Imprimimos el area de los cuatro contornos más grandes
if contours:
    for contour in contours[:4]:
        print(cv2.contourArea(contour))

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()

# Paso 1, saber cuantas cartas hay sobre la mesa
# Paso 2, saber el color de las distitnas cartas que hay en la mesa
# Paso 3, saber el número de las distintas cartas que hay sobre la mesa
# Paso 4, el palo de las distintas cartas que hay sobre la mesa
# Paso 5, saber la posición de las cartas sobre la mesa
# Paso Extra, detección del joker


