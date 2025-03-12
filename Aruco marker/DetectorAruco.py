import cv2
import pickle

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
        cv2.aruco.drawDetectedMarkers(undistorted, corners, ids)
        print(f"Marcadores detectados: {ids.flatten()}")

    # Mostrar la imagen corregida con los marcadores detectados
    cv2.imshow("Imagen Corregida con Marcadores", undistorted)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
