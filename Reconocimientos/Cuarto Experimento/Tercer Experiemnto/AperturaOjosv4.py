import cv2 as cv
import numpy as np
import mediapipe as mp
from math import sqrt
import os
import logging

import time

# Variables para calcular los FPS
prev_frame_time = 0
new_frame_time = 0

# Deshabilitar los mensajes de advertencia de MediaPipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
# Inicializa MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
#mp_drawing = mp.solutions.drawing_utils

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
MARGIN = 10  # Ajustar este valor según tus necesidades

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

# Contadores para parpadeos
COUNTER = 0
TOTAL_BLINKS = 0

FONT = cv.FONT_HERSHEY_SIMPLEX

# Función para detectar los landmarks del mesh
def landmarksDetection(image, results, draw=False):
    image_height, image_width = image.shape[:2]
    mesh_coordinates = [(int(point.x * image_width), int(point.y * image_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        for point in mesh_coordinates:
            cv.circle(image, point, 1, (0, 255, 0), -1)
    return mesh_coordinates

# Función para calcular la distancia euclidiana entre dos puntos
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Función para calcular la razón de parpadeo
def blinkRatio(image, landmarks, right_indices, left_indices):
    right_eye_landmark1 = landmarks[right_indices[0]]
    right_eye_landmark2 = landmarks[right_indices[8]]
    right_eye_landmark3 = landmarks[right_indices[12]]
    right_eye_landmark4 = landmarks[right_indices[4]]

    left_eye_landmark1 = landmarks[left_indices[0]]
    left_eye_landmark2 = landmarks[left_indices[8]]
    left_eye_landmark3 = landmarks[left_indices[12]]
    left_eye_landmark4 = landmarks[left_indices[4]]

    right_eye_horizontal_distance = euclideanDistance(right_eye_landmark1, right_eye_landmark2)
    right_eye_vertical_distance = euclideanDistance(right_eye_landmark3, right_eye_landmark4)
    left_eye_vertical_distance = euclideanDistance(left_eye_landmark3, left_eye_landmark4)
    left_eye_horizontal_distance = euclideanDistance(left_eye_landmark1, left_eye_landmark2)

    right_eye_ratio = right_eye_horizontal_distance / right_eye_vertical_distance
    left_eye_ratio = left_eye_horizontal_distance / left_eye_vertical_distance

    eyes_ratio = (right_eye_ratio + left_eye_ratio) / 2

    return eyes_ratio

def Apertura(image, landmarks, right_indices, left_indices):
    
        # Puntos para el ojo derecho
    right_eye_upper = landmarks[right_indices[12]]  # Índice 159
    right_eye_lower = landmarks[right_indices[4]]  # Índice 145

    # Puntos para el ojo izquierdo
    left_eye_upper = landmarks[left_indices[12]]  # Índice 386
    left_eye_lower = landmarks[left_indices[4]]  # Índice 374

    # Calcular la distancia vertical para cada ojo
    AperturaOjoD = euclideanDistance(right_eye_upper, right_eye_lower)
    AperturaOjoI = euclideanDistance(left_eye_upper, left_eye_lower)

    return AperturaOjoD , AperturaOjoI 
def scale_to_percentage(distance, min_distance, max_distance):
    # Asegúrate de que la distancia no exceda los límites establecidos
    if distance < min_distance:
        return 0
    elif distance > max_distance:
        return 100
    else:
        # Escala lineal de la distancia a porcentaje
        return (distance - min_distance) / (max_distance - min_distance) * 100
# Crear filtros para los porcentajes de apertura de cada ojo
left_eye_percentage_filter = MovingAverageFilter(size=20)
right_eye_percentage_filter = MovingAverageFilter(size=20)    



# Distancia de referencia en cm (e.g., cuando la persona está a 50cm de la cámara)
reference_distance = 60.0
# Valores originales de min_distance y max_distance a la distancia de referencia
original_min_distance = 5.1
original_max_distance = 14.4


def adjust_aperture_limits(current_distance, reference_distance, original_min, original_max):
    if current_distance == 0:
        return original_min, original_max
    scale_factor = reference_distance / current_distance
    return original_min * scale_factor, original_max * scale_factor




def round_to_nearest_ten(num):
    return round(num / 10.0) * 10

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
            # Trabajar con una copia del frame sin anotaciones
            frame_clean = frame.copy()
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            # Dibujar contornos de los ojos en el frame original
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

            # Dibujar el centro del ojo izquierdo y derecho en el frame original
            cv.circle(frame, tuple(left_eye_center), 1, (0, 0, 255), -1, cv.LINE_AA)
            cv.circle(frame, tuple(right_eye_center), 1, (0, 0, 255), -1, cv.LINE_AA)

            # Dibujar el círculo del iris en el frame original
            cv.circle(frame, tuple(center_left_iris), int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, tuple(center_right_iris), int(r_radius), (255, 0, 255), 1, cv.LINE_AA)

            # Dibujar el centro de la pupila en el frame original
            cv.circle(frame, tuple(center_left_iris), 1, (255, 0, 0), -1, cv.LINE_AA)
            cv.circle(frame, tuple(center_right_iris), 1, (255, 0, 0), -1, cv.LINE_AA)

            # Calcular la distancia focal en píxeles
            focal_length_px = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * img_w

            # Calcular la distancia a la cámara para ambos ojos y mostrarla en la imagen original
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

            cv.circle(frame, mesh_points[PUNTO_NARIZ], 2, (130, 100, 0), -1, cv.LINE_AA)




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
            
            # Extraer la región de interés de la cara del frame limpio
            face_roi = frame_clean[smoothed_y_min:smoothed_y_max, smoothed_x_min:smoothed_x_max]

            # Redimensionar la región de interés al tamaño de la ventana constante
            face_window = cv.resize(face_roi, (FACE_WINDOW_SIZE, FACE_WINDOW_SIZE))

            # Convertir la región de interés a RGB
            face_window_rgb = cv.cvtColor(face_window, cv.COLOR_BGR2RGB)

            # Aplicar FaceMesh independiente para el conteo de parpadeos en la región de interés
            with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as face_mesh_roi:
                face_mesh_results = face_mesh_roi.process(face_window_rgb)

                if face_mesh_results.multi_face_landmarks:
                    mesh_coordinates = landmarksDetection(face_window, face_mesh_results, True)  # Mostrar puntos para depuración
                    eyes_ratio = blinkRatio(face_window, mesh_coordinates, RIGHT_EYE, LEFT_EYE)
                    # Después de calcular la apertura de los ojos
                    AperturaOjoD, AperturaOjoI = Apertura(face_window, mesh_coordinates, RIGHT_EYE, LEFT_EYE)

                                # En el bucle principal, después de calcular la distancia
                    corrected_distance = average_distance * CORRECTION_FACTOR  # Esta es la distancia actual calculada
                    min_distance, max_distance = adjust_aperture_limits(corrected_distance, reference_distance, original_min_distance, original_max_distance)

                    # Ahora, usa min_distance y max_distance ajustados para calcular el porcentaje de apertura
                    porcentaje_derecho_bruto = scale_to_percentage(AperturaOjoD, min_distance, max_distance)
                    porcentaje_izquierdo_bruto = scale_to_percentage(AperturaOjoI, min_distance, max_distance)

                    # Aplicar filtro de media móvil
                    left_eye_percentage_filter.add_value(porcentaje_izquierdo_bruto)
                    right_eye_percentage_filter.add_value(porcentaje_derecho_bruto)

                    smoothed_left_percentage = left_eye_percentage_filter.get_average()
                    smoothed_right_percentage = right_eye_percentage_filter.get_average()

                    # Redondear los porcentajes suavizados
                    porcentaje_derecho = round_to_nearest_ten(smoothed_right_percentage)
                    porcentaje_izquierdo = round_to_nearest_ten(smoothed_left_percentage)

                    # Mostrar los porcentajes en el frame
                    cv.putText(frame, f'Porcentaje Ojo Derecho: {porcentaje_derecho}%', (10, 130), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv.putText(frame, f'Porcentaje Ojo Izquierdo: {porcentaje_izquierdo}%', (10, 160), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    if eyes_ratio > 9.0:  # Umbral ajustado para parpadeo
                        COUNTER += 1
                    else:
                        if COUNTER > 2:  # Umbral ajustado para evitar falsos positivos
                            TOTAL_BLINKS += 1
                            COUNTER = 0

                    cv.rectangle(face_window, (20, 240), (290, 280), (0, 0, 0), -1)
                    cv.putText(face_window, f'Total Blinks: {TOTAL_BLINKS}', (30, 270), FONT, 1, (0, 255, 0), 2)

            # Mostrar la ventana de la cara sin anotaciones
            cv.imshow('Face ROI', face_window)


                # Calcula los FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Convierte FPS a string y redondea
        fps = str(int(fps))

        # Pone el texto de los FPS en el frame
        cv.putText(frame, fps, (600, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv.LINE_AA)


        cv.imshow('Eye Landmarks and Iris', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
