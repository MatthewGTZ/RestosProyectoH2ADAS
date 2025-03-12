import cv2
import pickle
import numpy as np

# Cargar parámetros de calibración de la cámara
cameraMatrix_file = "C:/Users/matthias/Desktop/Trabajo/Programacion/3DReconstruccion/CalibracionEstereo/ResultadosCamara1/cameraMatrix.pkl"
dist_file = "C:/Users/matthias/Desktop/Trabajo/Programacion/3DReconstruccion/CalibracionEstereo/ResultadosCamara1/dist.pkl"

# Cargar los parámetros desde los archivos
with open(cameraMatrix_file, "rb") as f:
    cameraMatrix = pickle.load(f)

with open(dist_file, "rb") as f:
    dist = pickle.load(f)

# Inicializar la cámara
cap = cv2.VideoCapture(1)  # Cambia el índice si usas otra cámara

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Configurar el detector ArUco
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Función para calcular rvecs y tvecs usando solvePnP
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([
        [-marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]
    ], dtype=np.float32)

    rvecs = []
    tvecs = []

    for c in corners:
        _, rvec, tvec = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_ITERATIVE)
        rvecs.append(rvec)
        tvecs.append(tvec)

    return rvecs, tvecs

# Función para proyectar un cubo o una esfera flotando sobre el marcador
def draw_3d_object(frame, corners, rvec, tvec, cameraMatrix, dist, object_type):
    if object_type == "cube":
        axis = np.float32([
            [0, 0, 1.1], [0, 1, 1.1], [1, 1, 1.1], [1, 0, 1.1],  # Base elevada del cubo
            [0, 0, 0.1], [0, 1, 0.1], [1, 1, 0.1], [1, 0, 0.1]  # Parte superior del cubo
        ])
        axis = axis * 0.05  # Escalar el cubo
    elif object_type == "sphere":
        # Generar puntos para una esfera
        phi, theta = np.mgrid[0:np.pi:10j, 0:2*np.pi:10j]
        x = 0.05 * np.sin(phi) * np.cos(theta)
        y = 0.05 * np.sin(phi) * np.sin(theta)
        z = 0.05 * np.cos(phi) + 0.05  # Flotando encima
        axis = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Proyectar puntos en la imagen
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, cameraMatrix, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    if object_type == "cube":
        # Dibujar el cubo
        frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)
        for i, j in zip(range(4), range(4, 8)):
            frame = cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 2)
        frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2)
    elif object_type == "sphere":
        # Dibujar la esfera
        for pt in imgpts:
            frame = cv2.circle(frame, tuple(pt), 1, (255, 255, 0), -1)

    return frame

print("Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break

    # Corregir distorsión
    h, w = frame.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)

    # Recortar la imagen a la región válida (opcional)
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]

    # Detectar marcadores ArUco
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    # Dibujar los marcadores detectados
    if ids is not None:
        rvecs, tvecs = my_estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, dist)
        for i in range(len(ids)):
            cv2.drawFrameAxes(undistorted, cameraMatrix, dist, rvecs[i], tvecs[i], 0.1)
            object_type = "cube" if ids[i] == 1 else "sphere"
            undistorted = draw_3d_object(undistorted, corners[i], rvecs[i], tvecs[i], cameraMatrix, dist, object_type)

    # Mostrar la imagen corregida con los marcadores detectados
    cv2.imshow("Imagen Corregida con Marcadores", undistorted)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
