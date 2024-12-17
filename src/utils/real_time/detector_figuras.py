from ultralytics import YOLO  # Import YOLOv8
import torch

class FigureDetector:
    def __init__(self):
        # Cargamos el modelo YOLOv8
        model_path = "../../static/yolov8/weights/best.pt" 
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def detectar_figuras(self, image, color=None, figura=False):
        # Lanzamos la detección con YOLOv8
        results = self.model.predict(image, device=self.device, verbose=False)
        detected_classes = [self.model.names[int(box.cls[0])] for result in results for box in result.boxes]

        # Separamos la detección de J, Q, K y las figuras
        letters = ["J", "Q", "K"]
        figures = [cls for cls in detected_classes if cls not in letters]

        letter = None
        if figura:
            detected_letters = [cls for cls in detected_classes if cls in letters]
            letter_label = max(set(detected_letters), key=detected_letters.count) if detected_letters else "Unknown"

        # Determinamos la figura más detectada (excluyendo J, Q y K)
        if figures:
            figure_label = max(set(figures), key=figures.count)
        else:
            figure_label = "Unknown"

        # Post procesamiento de la figura si se ha recibido un color a comparar
        if color is not None:
            figure_label = self.__post_procesado(figure_label, color)

        # Si figura es True entonces devolvemos ambos labels, si no solo la figura
        if figura:
            return figure_label, letter_label
        else:
            return figure_label

    def __post_procesado(self, label, color):
        label_corrected = label

        # Definimos qué figuras son correctas para cada color
        color_mapping = {
            "Roja": ["corazon", "rombo"],  # Corazones y Diamantes siempre ROJOS
            "Negra": ["pica", "trebol"]   # Picas y Treboles siempre NEGROS
        }

        # Correcciones manuales para casos incorrectos
        corrections = {
            "corazon": "pica",  
            "pica": "corazon",  
            "trebol": "rombo",  
            "rombo": "trebol"  
        }

        if label != "Unknown" and color in color_mapping:
            valid_suits = color_mapping[color]
            if label not in valid_suits:
                # Corregimos el label si es incorrecto y lo imprimimos
                label_corrected = corrections.get(label)
                print(f"Detectado un: {label} || Error dado el color: {color}, corregido a {label_corrected}")
            else:
                # Si es valido, se explicita por pantalla
                print(f"Detectado un: {label} || Validado como correcto para el color: {color}")

        return label_corrected
