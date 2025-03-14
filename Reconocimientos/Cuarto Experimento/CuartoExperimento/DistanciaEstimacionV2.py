import cv2 as cv
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Índices de los contornos del ojo izquierdo y derecho
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# Índices del iris izquierdo y derecho
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Tamaño constante para la visualización de la ROI del ojo
EYE_ROI_DISPLAY_SIZE = 150  # Tamaño para la visualización
ZOOM_FACTOR = 6  # Factor de zoom para el recorte

# Parámetros de la cámara (debe ajustarse según tu cámara)
FOCAL_LENGTH_MM = 3.6  # Longitud focal típica para una cámara de portátil
SENSOR_WIDTH_MM = 4.8  # Ancho del sensor en mm (ajustar según tu cámara)

# Factor de corrección
CORRECTION_FACTOR = 1.13  # Ajustar este valor basado en observaciones

# Función para calcular la distancia usando el diámetro del iris
def calculate_distance(iris_diameter_px, focal_length_px, real_iris_diameter_mm=11.7):
    # Usar la relación de semejanza para estimar la distancia
    distance_mm = (real_iris_diameter_mm * focal_length_px) / iris_diameter_px
    distance_cm = distance_mm / 10  # Convertir a cm
    return distance_cm

# Filtro de media móvil para suavizar las distancias
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

            # Dibujar contornos de los ojos
            cv.polylines(frame, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0, 255, 0), 1, cv.LINE_AA)

            # Calcular el centro del ojo izquierdo y derecho
            left_eye_center = np.mean(mesh_points[LEFT_EYE], axis=0).astype(int)
            right_eye_center = np.mean(mesh_points[RIGHT_EYE], axis=0).astype(int)
            
            # Calcular el círculo del iris izquierdo y derecho
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left_iris = np.array([l_cx, l_cy], dtype=np.int32)
            center_right_iris = np.array([r_cx, r_cy], dtype=np.int32)

            # Dibujar el centro del ojo izquierdo y derecho
            cv.circle(frame, tuple(left_eye_center), 1, (0, 0, 255), -1, cv.LINE_AA)
            cv.circle(frame, tuple(right_eye_center), 1, (0, 0, 255), -1, cv.LINE_AA)

            # Dibujar el círculo del iris
            cv.circle(frame, tuple(center_left_iris), int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, tuple(center_right_iris), int(r_radius), (255, 0, 255), 1, cv.LINE_AA)

            # Dibujar el centro de la pupila
            cv.circle(frame, tuple(center_left_iris), 1, (255, 0, 0), -1, cv.LINE_AA)
            cv.circle(frame, tuple(center_right_iris), 1, (255, 0, 0), -1, cv.LINE_AA)

            # Calcular el tamaño de recorte con zoom
            zoom_size = EYE_ROI_DISPLAY_SIZE // ZOOM_FACTOR

            # Extraer y redimensionar ROI de los ojos con zoom
            left_eye_roi = frame[max(0, left_eye_center[1] - zoom_size):min(left_eye_center[1] + zoom_size, img_h),
                                 max(0, left_eye_center[0] - zoom_size):min(left_eye_center[0] + zoom_size, img_w)]
            right_eye_roi = frame[max(0, right_eye_center[1] - zoom_size):min(right_eye_center[1] + zoom_size, img_h),
                                  max(0, right_eye_center[0] - zoom_size):min(right_eye_center[0] + zoom_size, img_w)]

            if left_eye_roi.size > 0:
                left_eye_roi_resized = cv.resize(left_eye_roi, (EYE_ROI_DISPLAY_SIZE, EYE_ROI_DISPLAY_SIZE))
                cv.imshow('Left Eye ROI', left_eye_roi_resized)

            if right_eye_roi.size > 0:
                right_eye_roi_resized = cv.resize(right_eye_roi, (EYE_ROI_DISPLAY_SIZE, EYE_ROI_DISPLAY_SIZE))
                cv.imshow('Right Eye ROI', right_eye_roi_resized)

            # Calcular la distancia focal en píxeles
            focal_length_px = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * img_w

            # Calcular la distancia a la cámara para ambos ojos y mostrarla en la imagen
            left_eye_distance = calculate_distance(l_radius * 2, focal_length_px)
            right_eye_distance = calculate_distance(r_radius * 2, focal_length_px)

            # Aplicar el filtro de media móvil
            left_eye_filter.add_value(left_eye_distance)
            right_eye_filter.add_value(right_eye_distance)
            smoothed_left_eye_distance = left_eye_filter.get_average()
            smoothed_right_eye_distance = right_eye_filter.get_average()
            
            # Promediar las distancias de ambos ojos para mayor estabilidad
            average_distance = (smoothed_left_eye_distance + smoothed_right_eye_distance) / 2
            
            # Aplicar el factor de corrección
            corrected_distance = average_distance * CORRECTION_FACTOR
            
            # Redondear la distancia al centímetro más cercano
            rounded_distance = round(corrected_distance)
            
            cv.putText(frame, f'Distance: {rounded_distance} cm', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv.imshow('Eye Landmarks and Iris', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
