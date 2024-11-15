import cv2
import math
import numpy as np

class ROIExtractor():
    def __init__(self):
        pass

    # Función para obtener la línea que pasa por dos puntos
    def __calculate_line(self, p1, p2):
        m = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else None
        if m is not None:
            b = p1[1] - m * p1[0]
        else:
            b = p1[0]  # Vertical line case
        return m, b

    # Función para obtener 2 puntos que están separados una distancia D de forma paralela a los dos puntos que se le envian como atributo
    def __parallel_shift(self, p1, p2, distance):
        direction = np.array([p2[1] - p1[1], -(p2[0] - p1[0])])  # Perpendicular vector
        norm = np.linalg.norm(direction)
        # Check por si los dos puntos son el mismo punto
        if norm == 0:
            return p1, p2
        else:
            unit_direction = direction / np.linalg.norm(direction)
            shift_vector = unit_direction * distance
            return p1 + shift_vector, p2 + shift_vector

    # Función para extraer los puntos en los que intersectan dos líneas
    def __find_intersection(self, m1, b1, m2, b2):
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

    # Función para obtener las distancias de las líneas del box, sirve para diferenciar entre lado largo y corto
    # Devuelve un diccionario con el tamaño de cada línea y los 2 puntos que la componen
    def __calculate_line_lengths(self, box):
        # Calculate the distances between consecutive points
        lines = []
        for i in range(len(box)):
            # Get the current point and the next point (wrap around using modulo)
            point1 = box[i]
            point2 = box[(i + 1) % len(box)]
            
            # Calculate the Euclidean distance
            distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
            lines.append((distance, (point1, point2)))
        
        # Sort the lines by length
        lines = sorted(lines, key=lambda x: x[0], reverse=True)
        
        # Identify long and short lines
        long_lines = lines[:2]  # Two longest lines
        short_lines = lines[2:]  # Two shortest lines
        
        return {"long": long_lines, "short" : short_lines}

    # Función que devuelve una lista con elementos que contienen las 4 esquinas de cada ROI para las cartas que estén detectadadas dentro del box que recibe
    def extract_rois(self, box):
        lines_dict = self.__calculate_line_lengths(box)
        parallel_edges, roi_list = [], []
        for line in lines_dict["long"]:
            p1, p2 = line[1][0], line[1][1]
            distance = -round((line[0]*0.1)) 
            shifted_p1, shifted_p2 = self.__parallel_shift(p1, p2, distance)
            parallel_edges.append((shifted_p1, shifted_p2))

            roi_list.append([p1, p2, shifted_p2, shifted_p1])

        return roi_list

    """
    def __extras(self, box):
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

    """
