import numpy as np
import cv2

# Definimos el tipo de marcador y los IDs que queremos generar
ARUCO_TYPE = "DICT_6X6_250"
OUTPUT_PREFIX = "aruco_marker_"  # Prefijo para los nombres de los archivos de salida
ARUCO_IDS = [0, 1, 2, 3]  # IDs de los marcadores a generar

# Diccionario de tipos de ArUco disponibles
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

# Verificamos si el tipo de ArUco especificado está soportado
if ARUCO_TYPE not in ARUCO_DICT:
    print(f"[ERROR] ArUCo tag of type '{ARUCO_TYPE}' is not supported.")
    exit(1)

# Cargamos el diccionario correspondiente
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[ARUCO_TYPE])

# Generamos y guardamos los marcadores
for aruco_id in ARUCO_IDS:
    print(f"[INFO] Generating ArUCo marker ID: {aruco_id}")
    tag = np.zeros((300, 300, 1), dtype="uint8")  # Crear imagen vacía para el marcador
    cv2.aruco.drawMarker(arucoDict, aruco_id, 300, tag, 1)  # Dibujar el marcador

    # Guardar el marcador en un archivo
    output_path = f"{OUTPUT_PREFIX}{aruco_id}.png"
    cv2.imwrite(output_path, tag)
    print(f"[INFO] ArUCo marker saved to: {output_path}")

    # Mostrar el marcador
    cv2.imshow(f"ArUCo Marker ID {aruco_id}", tag)

cv2.waitKey(0)
cv2.destroyAllWindows()
