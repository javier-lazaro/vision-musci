import cv2
import numpy as np

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

def detectar_color_carta(box, idx, color_detection_frame, active_windows):
    # Usamos una copia del `box` para mantener el estado original
    box = ordenar_puntos(box)

    # Obtener dimensiones reales del rectángulo rotado
    width = int(np.linalg.norm(box[0] - box[1]))
    height = int(np.linalg.norm(box[1] - box[2]))

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
    warped = cv2.warpPerspective(color_detection_frame, M, (width, height))

    # Rotar la imagen resultante si es necesario para que siempre esté en orientación vertical
    if warped.shape[1] > warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    # Escalar la ventana para que se ajuste mejor al tamaño original
    escala = 1.5
    scaled_width = int(width * escala)
    scaled_height = int(height * escala)
    warped_resized = cv2.resize(warped, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

    # Detectar si la carta es roja, amarilla o negra
    hsv_card = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # Definir los límites del color rojo en HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Definir los límites del color amarillo en HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Crear máscara para detectar rojos y amarillos
    mask_red1 = cv2.inRange(hsv_card, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_card, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    mask_yellow = cv2.inRange(hsv_card, lower_yellow, upper_yellow)

    # Calcular la cantidad de píxeles oscuros (negros) en escala de grises
    gray_card_central = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    mask_black = cv2.inRange(gray_card_central, 0, 80)

    # Calcular la cantidad de píxeles para cada color
    total_pixels = hsv_card.shape[0] * hsv_card.shape[1]
    red_pixels = np.sum(mask_red > 0)
    yellow_pixels = np.sum(mask_yellow > 0)
    black_pixels = np.sum(mask_black > 0)

    # Calcular el porcentaje de cada color
    red_percentage = (red_pixels / total_pixels) * 100
    yellow_percentage = (yellow_pixels / total_pixels) * 100
    black_percentage = (black_pixels / total_pixels) * 100

    # Determinar si la carta es una figura
    figura = yellow_percentage > 0.5

    # Determinar el color predominante
    color_label = "Negra"
    if red_percentage > yellow_percentage and red_percentage > black_percentage:
        color_label = "Roja"
    elif yellow_percentage > red_percentage and yellow_percentage > black_percentage:
        color_label = "Amarilla"

    # Añadir la etiqueta de porcentajes y el color sobre la carta detectada
    cv2.putText(warped_resized, f"Color: {color_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (191, 4, 255), 2, cv2.LINE_AA)
    cv2.putText(warped_resized, f"Roja: {red_percentage:.2f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(warped_resized, f"Amarilla: {yellow_percentage:.2f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(warped_resized, f"Negra: {black_percentage:.2f}%", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    # Mostrar la región recortada en una ventana con tamaño ajustado
    window_name = f'Carta_Rotada_{idx}'
    cv2.imshow(window_name, warped_resized)

    # Actualizar el diccionario de ventanas activas
    active_windows[idx] = window_name

    # Mover las ventanas de las cartas a la parte inferior derecha
    cv2.moveWindow(window_name, 650 + (idx % 5) * (scaled_width + 10), 430 + (idx // 5) * (scaled_height + 10))

    return figura, color_label, warped_resized