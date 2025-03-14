import cv2 as cv
import numpy as np
import mediapipe as mp


Mostrar_Print = False
Zoom_Ojos = False


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

# Variables para conteo de parpadeos
COUNTER = 0
TOTAL_BLINKS = 0

def euclideanDistance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def adjust_aperture_limits(distance, reference_distance, original_min_distance, original_max_distance):
    ratio = distance / reference_distance
    return original_min_distance * ratio, original_max_distance * ratio

def round_to_nearest_ten(number):
    return round(number / 10) * 10

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

def blinkRatio(landmarks, right_indices, left_indices):
    # Puntos para el ojo derecho
    right_eye_upper = landmarks[right_indices[12]]  # Índice 159
    right_eye_lower = landmarks[right_indices[4]]  # Índice 145

    # Puntos para el ojo izquierdo
    left_eye_upper = landmarks[left_indices[12]]  # Índice 386
    left_eye_lower = landmarks[left_indices[4]]  # Índice 374

    # Calcular la distancia vertical para cada ojo
    AperturaOjoD = euclideanDistance(right_eye_upper, right_eye_lower)
    AperturaOjoI = euclideanDistance(left_eye_upper, left_eye_lower)

    # Calcular la distancia horizontal para cada ojo
    right_eye_horizontal = euclideanDistance(landmarks[right_indices[0]], landmarks[right_indices[8]])
    left_eye_horizontal = euclideanDistance(landmarks[left_indices[0]], landmarks[left_indices[8]])

    # Calcular la relación de parpadeo
    ratio_d = AperturaOjoD / right_eye_horizontal
    ratio_i = AperturaOjoI / left_eye_horizontal

    return (ratio_d + ratio_i) / 2.0

def count_blinks(blink_ratio, counter, total_blinks, threshold=0.2):
    if blink_ratio < threshold:  # Si la relación es menor que el umbral, se considera un parpadeo
        counter += 1
    else:
        if counter > 2:  # Umbral para evitar falsos positivos
            total_blinks += 1
            counter = 0
    return counter, total_blinks

left_eye_percentage_filter = MovingAverageFilter(size=20)
right_eye_percentage_filter = MovingAverageFilter(size=20)

# Crear la ventana y los sliders
cv.namedWindow('Eye Landmarks and Iris')
cv.createTrackbar('Min Distance', 'Eye Landmarks and Iris', 5, 50, lambda x: None)
cv.createTrackbar('Max Distance', 'Eye Landmarks and Iris', 14, 50, lambda x: None)
cv.createTrackbar('Reference Distance', 'Eye Landmarks and Iris', 60, 100, lambda x: None)
cv.createTrackbar('Threshold', 'Eye Landmarks and Iris', 2, 10, lambda x: None)


cv.namedWindow('Calibration Instructions')

instructions = [
    "1. Ajustar la distancia referencia.",
    "2. Abrir los ojos y colocar la barra MAx Distance hasta llegar a 100%.",
    "3. Cerrar los ojos y colocar la barra Min Distance hasta llegar a 0%."
]

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

            if(Zoom_Ojos == True):


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
            
            # Calcular la apertura de los ojos
            AperturaOjoD, AperturaOjoI = Apertura(frame, mesh_points, RIGHT_EYE, LEFT_EYE)
            
            # Obtener los valores de los sliders
            min_distance = cv.getTrackbarPos('Min Distance', 'Eye Landmarks and Iris')
            max_distance = cv.getTrackbarPos('Max Distance', 'Eye Landmarks and Iris')
            reference_distance = cv.getTrackbarPos('Reference Distance', 'Eye Landmarks and Iris')
            threshold = cv.getTrackbarPos('Threshold', 'Eye Landmarks and Iris') / 10.0

            # Calcular el porcentaje de apertura del ojo
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
            
            # Calcular la relación de parpadeo de los ojos
            blink_ratio = blinkRatio(mesh_points, RIGHT_EYE, LEFT_EYE)
            
            # Llamar a la función de conteo de pestañeos
            COUNTER, TOTAL_BLINKS = count_blinks(blink_ratio, COUNTER, TOTAL_BLINKS, threshold)

            cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if Mostrar_Print:


                # Imprimir valores para depuración
                print(f'AperturaOjoD: {AperturaOjoD}, AperturaOjoI: {AperturaOjoI}')
                print(f'Blink Ratio: {blink_ratio}')
                print(f'Threshold: {threshold}')
                print(f'COUNTER: {COUNTER}, TOTAL_BLINKS: {TOTAL_BLINKS}')
        
       # Formatear valores para mostrar con tres cifras
        formatted_AperturaOjoI = f"{AperturaOjoI:.3f}"  # Tres cifras decimales
        formatted_AperturaOjoD = f"{AperturaOjoD:.3f}"
        formatted_blink_ratio = f"{blink_ratio:.3f}"

        # Posiciones iniciales del texto
        base_y = img_h - 15  # Base line para texto, ajusta este valor si es necesario
        offset = 3  # Ajusta este valor para modificar el espacio entre las líneas de texto

        # Mostrar los valores en el frame, cada uno en una línea diferente
        cv.putText(frame, f'AperturaOjoI= {formatted_AperturaOjoI}', (10, base_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (25, 100, 160), 2)
        cv.putText(frame, f'AperturaOjoD= {formatted_AperturaOjoD}', (250, base_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (25, 100, 160), 2)
        cv.putText(frame, f'blink_ratio = {formatted_blink_ratio}', (500, base_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (25, 100, 160), 2)

        # Llamar a la función de conteo de pestañeos
        COUNTER, TOTAL_BLINKS = count_blinks(blink_ratio, COUNTER, TOTAL_BLINKS, threshold)
        # Mostrar instrucciones en la ventana de calibración
        calibration_frame = np.zeros((150, 400, 3), dtype=np.uint8)
        y0, dy = 20, 30
        for i, line in enumerate(instructions):
            y = y0 + i * dy
            cv.putText(calibration_frame, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        cv.imshow('Eye Landmarks and Iris', frame)

        cv.imshow('Calibration Instructions', calibration_frame)

        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
