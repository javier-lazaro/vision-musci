import cv2
import numpy as np

# Definir tamaño del tablero de ajedrez y criterios de refinamiento
CHECKERBOARD = (7, 7)
CHECKER_SIZE = 15  # Tamaño de cada cuadrado en mm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Definir los puntos del tablero en 3D
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * CHECKER_SIZE

# Arrays para almacenar los puntos 3D y 2D
objpoints = []  # Puntos 3D del mundo real
imgpoints = []  # Puntos 2D en la imagen

# Iniciar la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)


    if ret:
        # Refinar las esquinas y almacenarlas
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            print('Imagen detectada')
            objpoints.append(objp)
            imgpoints.append(corners2)

        # Dibujar las esquinas detectadas en la imagen
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)

    cv2.imshow('Calibración de Cámara', frame)

    # Salir del bucle con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Realizar la calibración de la cámara si se han capturado puntos
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Matriz Intrínseca:", mtx)
    print("Coeficientes de Distorsión:", dist)

    cap = cv2.VideoCapture(0)
    # Mostrar en tiempo real la corrección de la imagen
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)
        cv2.imshow('Imagen sin Distorsión', undistorted_frame)

        # Salir del bucle con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print("No se encontraron suficientes puntos para calibrar la cámara.")

np.savez('calibration_data_2.npz', camera_matrix = mtx, dist_coeffs = dist)