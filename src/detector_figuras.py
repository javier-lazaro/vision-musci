from ultralytics import YOLO  # Import YOLOv8
import torch


class FigureDetector:
    def __init__(self):
        # Load YOLOv8 model
        model_path = "../static/yolov8/weights/best.pt"  # Update path if needed
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)

    def detectar_figuras(self, image, color=None, figura=False):
        results = self.model.predict(image, device=self.device, verbose=False)

        # Parse YOLOv8 results
        detected_classes = [self.model.names[int(box.cls[0])] for result in results for box in result.boxes]
        if detected_classes:
            # Determine the most detected figure
            figure_label = max(set(detected_classes), key=detected_classes.count)
        else:
            figure_label = "Unknown"
        
        if color != None:
            figure_label = self.__post_procesado(figure_label, color)
        
        return figure_label


    def __post_procesado(self, label, color):
        label_corrected = label

        color_mapping = {
            "Roja": ["Corazon", "Rombo"],  # Corazones y Diamantes son siempre ROJOS
            "Negra": ["Pica", "Trebol"]   # Treboles y Picas son siempre NEGROS
        }

        # Diccionario con las correcciones realizadas de forma manual
        correcciones = {
            "Corazon": "Pica",  
            "Pica": "Corazon",  
            "Trebol": "Rombo",  
            "Rombo": "Trebol"  
        }

        if label != "Unknown" and color in color_mapping:
            valid_suits = color_mapping[color]
            if label not in valid_suits:
                label_corrected = correcciones.get(label, label)
                print(f"Detectado un: {label} erroneo dato el color: {color}")

        return label_corrected
