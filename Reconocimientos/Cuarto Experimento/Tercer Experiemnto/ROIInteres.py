import cv2 as cv
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Índices de los contornos del ojo izquierdo y derecho
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# Índices del iris izquierdo y derecho
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

PUNTO_NARIZ = 1

# Tamaño constante para la visualización de la ROI del ojo
EYE_ROI_DISPLAY_SIZE = 150  # Tamaño para la visualización
ZOOM_FACTOR = 6  # Factor de zoom para el recorte

# Tamaño constante de la ventana de la cara
FACE_WINDOW_SIZE = 300

# Parámetros de la cámara (debe ajustarse según tu cámara)
FOCAL_LENGTH_MM = 3.6  # Longitud focal típica para una cámara de portátil
SENSOR_WIDTH_MM = 4.8  # Ancho del sensor en mm (ajustar según tu cámara)

# Factor de corrección
CORRECTION_FACTOR = 1.13  # Ajustar este valor basado en observaciones

# Margen adicional para el rectángulo alrededor de la cara
MARGIN = 20  # Ajustar este valor según tus necesidades

# Función para calcular la distancia usando el diámetro del iris
def calculate_distance(iris_diameter_px, focal_length_px, real_iris_diameter_mm=11.7):
    # Usar la relación de semejanza para estimar la distancia
    distance_mm = (real_iris_diameter_mm * focal_length_px) / iris_diameter_px
    distance_cm = distance_mm / 10  # Convertir a cm
    return distance_cm

# Filtro de media móvil para suavizar las distancias y las coordenadas del rectángulo
class MovingAverageFilter:
    def __init__(self, size=20):
        self.size = size
        self.values = []

    def add_value(self, value):
        if len(self.values) >= self.size:
            self.values.pop(0)
        self.values.append(value)

    def get_average(self):
        if not self.values:
            return 0
        return sum(self.values) / len(self.values)

left_eye_filter = MovingAverageFilter(size=20)
right_eye_filter = MovingAverageFilter(size=20)
rect_x_min_filter = MovingAverageFilter(size=20)
rect_y_min_filter = MovingAverageFilter(size=20)
rect_x_max_filter = MovingAverageFilter(size=20)
rect_y_max_filter = MovingAverageFilter(size=20)

cap = cv.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            # Calcular el rectángulo alrededor de la cara
            x_min = np.min(mesh_points[:, 0]) - MARGIN
            y_min = np.min(mesh_points[:, 1]) - MARGIN
            x_max = np.max(mesh_points[:, 0]) + MARGIN
            y_max = np.max(mesh_points[:, 1]) + MARGIN
            
            # Asegurarse de que las coordenadas están dentro de los límites de la imagen
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)

            # Aplicar el filtro de media móvil a las coordenadas del rectángulo
            rect_x_min_filter.add_value(x_min)
            rect_y_min_filter.add_value(y_min)
            rect_x_max_filter.add_value(x_max)
            rect_y_max_filter.add_value(y_max)
            
            smoothed_x_min = int(rect_x_min_filter.get_average())
            smoothed_y_min = int(rect_y_min_filter.get_average())
            smoothed_x_max = int(rect_x_max_filter.get_average())
            smoothed_y_max = int(rect_y_max_filter.get_average())
            
            # Extraer la región de interés de la cara
            face_roi = frame[smoothed_y_min:smoothed_y_max, smoothed_x_min:smoothed_x_max]

            # Redimensionar la región de interés al tamaño de la ventana constante
            face_window = cv.resize(face_roi, (FACE_WINDOW_SIZE, FACE_WINDOW_SIZE))

            # Mostrar la ventana de la cara
            cv.imshow('Face ROI', face_window)

        cv.imshow('Eye Landmarks and Iris', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
