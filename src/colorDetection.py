import cv2
import numpy as np

class ColorDetection:
    def __init__(self, frame, box):
        self.frame = frame
        self.box = box

    def draw_line_between_corners(self):
        # Dibujar una línea entre la esquina 1 y la esquina 3 del box (línea original)
        corner_1 = tuple(self.box[1])  # Esquina 1 (superior izquierda)
        corner_3 = tuple(self.box[2])  # Esquina 3 (superior derecha)
        cv2.line(self.frame, corner_1, corner_3, (255, 0, 0), 3)  # Dibujar la línea de color azul con grosor 3

        # Calcular la altura del rectángulo delimitador y un desplazamiento del 25%
        _, _, _, height = cv2.boundingRect(self.box)
        offset_1 = int(height * 0.25)  # Primer desplazamiento del 25%

        # Crear los puntos desplazados para la primera línea paralela (25% por debajo)
        corner_1_offset_1 = (corner_1[0], corner_1[1] + offset_1)
        corner_3_offset_1 = (corner_3[0], corner_3[1] + offset_1)

        # Dibujar la primera línea paralela desplazada (línea verde)
        cv2.line(self.frame, corner_1_offset_1, corner_3_offset_1, (0, 255, 0), 3)

        # Dibujar una línea entre las esquinas inferiores del box (segunda línea paralela)
        corner_2 = tuple(self.box[0])  # Esquina 2 (inferior izquierda)
        corner_4 = tuple(self.box[3])  # Esquina 4 (inferior derecha)

        # Crear los puntos desplazados para la primera línea paralela (25% por debajo)
        corner_2_offset_1 = (corner_2[0], corner_2[1] - offset_1)
        corner_4_offset_1 = (corner_4[0], corner_4[1] - offset_1)

        cv2.line(self.frame, corner_2_offset_1, corner_4_offset_1, (0, 255, 0), 3)  # Dibujar la línea roja con grosor 3

        # Calcular la dirección del vector desde corner_2 a corner_4
        direction_vector = (corner_4_offset_1[0] - corner_2_offset_1[0], corner_4_offset_1[1] - corner_2_offset_1[1])

        # Extender la línea hacia la izquierda y hacia la derecha usando un factor grande
        extend_factor = 1000  # Este valor puede ajustarse según el tamaño del frame

        # Calcular puntos extendidos hacia adelante y hacia atrás a lo largo de la dirección del vector
        extended_start = (corner_2_offset_1[0] - extend_factor * direction_vector[0], corner_2_offset_1[1] - extend_factor * direction_vector[1])
        extended_end = (corner_4_offset_1[0] + extend_factor * direction_vector[0], corner_4_offset_1[1] + extend_factor * direction_vector[1])

        # Convertir a enteros para usar en la función cv2.line
        extended_start = tuple(map(int, extended_start))
        extended_end = tuple(map(int, extended_end))

        # Dibujar la línea extendida (línea roja)
        cv2.line(self.frame, extended_start, extended_end, (0, 255, 255), 3)

        # Llamada a la función
        intersection_point = self.line_intersection((corner_2_offset_1, corner_4_offset_1), (extended_start, extended_end))

        # Dibuja la intersección si existe
        if intersection_point:
            cv2.circle(self.frame, intersection_point, 100, (0, 0, 255), -1)  # Dibuja un círculo rojo en el punto de intersección
        else :
            print("Las líneas son paralelas y no se cruzan")

    def line_intersection(self, line1, line2):
        """Calcula el punto de intersección entre dos líneas.
        line1 y line2 deben estar en formato ((x1, y1), (x2, y2)).
        """
        x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
        x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]

        # Cálculo de la intersección usando la fórmula del determinante
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None  # Las líneas son paralelas y no tienen intersección.

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        return int(px), int(py)

    def show_detected_card(self):
        # Extraer la región de la carta usando las coordenadas del box
        x, y, w, h = cv2.boundingRect(self.box)
        card_region = self.frame[y:y + h, x:x + w]

        if card_region.size > 0:
            window_name = 'Carta_detectada'
            cv2.imshow(window_name, card_region)
            cv2.moveWindow(window_name, 300, 300)  # Puedes ajustar la posición de la ventana

