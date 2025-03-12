import pyrealsense2 as rs
import cv2
import os
from datetime import datetime
import numpy as np
import time
from collections import deque

# Configuración de la cámara RealSense
pipeline = rs.pipeline()
config = rs.config()

# Configura la resolución y formato de captura (color e IR)
frame_width = 1280
frame_height = 720
frame_rate = 30
config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, frame_rate)
config.enable_stream(rs.stream.infrared, frame_width, frame_height, rs.format.y8, frame_rate)

# Inicia la cámara
pipeline.start(config)

# Desactiva el emisor infrarrojo (proyector de puntos)
device = pipeline.get_active_profile().get_device()
depth_sensor = device.query_sensors()[0]
depth_sensor.set_option(rs.option.emitter_enabled, 0)

# Crear estructura de carpetas para guardar los videos
main_folder = "./grabacionesRGBIR_simultaneo"
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_folder = os.path.join(main_folder, f"grabacion_{current_time}")
os.makedirs(output_folder, exist_ok=True)

# Configuración del archivo de salida de video combinado
video_filename = os.path.join(output_folder, f'video_combinado_{current_time}.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(video_filename, fourcc, frame_rate, (frame_width, frame_height))

print(f"Grabando video combinado. Presiona 'q' para detener.")
print(f"Video se guardará en: {video_filename}")

# Parámetros de detección
luminosity_threshold_dark = 70  # Umbral para detectar oscuridad
luminosity_threshold_bright = 100  # Umbral para volver a RGB
luminosity_queue = deque(maxlen=10)  # Promedio móvil para suavizar cambios
current_mode = "RGB"  # Estado inicial

try:
    while True:
        start_time = time.time()  # Marca de tiempo inicial

        # Captura un frame de la cámara
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        ir_frame = frames.get_infrared_frame()

        if not color_frame or not ir_frame:
            continue

        # Convierte los frames en arrays de numpy
        color_image = np.asanyarray(color_frame.get_data())
        ir_image = np.asanyarray(ir_frame.get_data())

        # Calcula la luminosidad promedio del frame IR
        avg_luminosity = np.mean(ir_image)
        luminosity_queue.append(avg_luminosity)  # Añade al promedio móvil
        smoothed_luminosity = np.mean(luminosity_queue)  # Promedio móvil

        # Cambia entre modos basado en la luminosidad promedio
        if current_mode == "RGB" and smoothed_luminosity < luminosity_threshold_dark:
            current_mode = "IR"
            print(f"Cambiando a modo IR. Luminosidad promedio: {smoothed_luminosity:.2f}")
        elif current_mode == "IR" and smoothed_luminosity > luminosity_threshold_bright:
            current_mode = "RGB"
            print(f"Cambiando a modo RGB. Luminosidad promedio: {smoothed_luminosity:.2f}")

        # Selecciona el frame a guardar y mostrar
        if current_mode == "IR":
            frame_to_save = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
        else:
            frame_to_save = color_image

        frame_to_save = cv2.flip(frame_to_save,-1)

        # Muestra el frame actual y el modo seleccionado
        #cv2.putText(frame_to_save, f"Modo: {current_mode}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('RealSense - Video Combinado', frame_to_save)

        # Escribe el frame en el archivo de video
        output_video.write(frame_to_save)

        # Detiene la grabación si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Espera el tiempo restante para completar el frame rate
        elapsed_time = time.time() - start_time
        if elapsed_time < 1.0 / frame_rate:
            time.sleep(1.0 / frame_rate - elapsed_time)

finally:
    # Libera los recursos
    pipeline.stop()
    output_video.release()
    cv2.destroyAllWindows()

print("Grabación finalizada. Video guardado en:", video_filename)
