#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int32MultiArray
from collections import Counter

class Robot:
    def __init__(self):
        rospy.init_node('robot_node', anonymous=True)
        rospy.Subscriber('/cartas_detectadas', Int32MultiArray, self.vision_callback)
        self.detected_cards = []  # Última lectura de cartas detectadas
        self.readings = []  # Lecturas acumuladas para varias iteraciones

    def vision_callback(self, msg):
        # Guardar las cartas detectadas en el callback
        self.detected_cards = msg.data

    def esperar_deteccion(self):
        # Esperar hasta que se detecten cartas
        rospy.loginfo("Esperando detección de cartas...")
        while not rospy.is_shutdown() and not self.detected_cards:
            rospy.sleep(0.1)
        rospy.loginfo(f"Cartas detectadas: {self.detected_cards}")
        return self.detected_cards

    def realizar_varias_lecturas(self, num_lecturas=5):
        # Realizar varias lecturas y acumular los resultados
        self.readings = []
        rospy.loginfo(f"Realizando {num_lecturas} lecturas para mayor precisión...")
        for i in range(num_lecturas):
            self.detected_cards = []
            self.esperar_deteccion()  # Esperar cada lectura
            if self.detected_cards:  # Solo acumular si hay lecturas válidas
                self.readings.append(self.detected_cards)
            rospy.sleep(0.1)  # Pequeña pausa entre lecturas

        rospy.loginfo(f"Lecturas acumuladas: {self.readings}")
        return self.readings

    def calcular_valores_frecuentes(self):
        # Calcular el número más frecuente para cada posición
        if not self.readings:
            return []

        # Transponer las lecturas acumuladas (para manejar cada posición por separado)
        transposed = list(zip(*self.readings))
        frequent_values = []
        for pos, numbers in enumerate(transposed):
            count = Counter(numbers)
            most_common = count.most_common(1)[0][0]  # Número más frecuente
            frequent_values.append(most_common)
            rospy.loginfo(f"Posición {pos + 1}, valores: {numbers}, más frecuente: {most_common}")

        return frequent_values

    def ordenar_cartas(self, cartas):
        # Simulación de ordenar las cartas
        rospy.loginfo("Ordenando las cartas en la mesa...")
        cartas.sort()
        rospy.loginfo(f"Cartas ordenadas: {cartas}")

    def run(self):
        # Ciclo principal
        while not rospy.is_shutdown():
            rospy.loginfo("Inicio del ciclo")

            # Paso 1: Detectar cartas (múltiples lecturas)
            self.readings = self.realizar_varias_lecturas(num_lecturas=5)
            cartas_detectadas = self.calcular_valores_frecuentes()
            rospy.loginfo(f"Cartas finales detectadas: {cartas_detectadas}")

            # Paso 2: Ordenar cartas
            self.ordenar_cartas(cartas_detectadas)

            # Paso 3: Esperar a que el humano desordene
            rospy.loginfo("Esperando a que el humano desordene las cartas...")
            rospy.sleep(5)  # Simulación de espera

            # Paso 4: Volver a detectar cartas (repetir ciclo)
            rospy.loginfo("Reiniciando detección de cartas...")
            self.readings = []

if __name__ == '__main__':
    try:
        robot = Robot()
        robot.run()
    except rospy.ROSInterruptException:
        pass
