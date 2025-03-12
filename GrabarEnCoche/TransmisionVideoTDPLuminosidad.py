import pyrealsense2 as rs
import cv2
import numpy as np
import socket
import struct
import time
from collections import deque

def connect_socket(ip, port, max_retries=5, retry_delay=3):
    """Intenta conectar un socket TCP con reintentos."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connected = False
    for attempt in range(max_retries):
        try:
            sock.connect((ip, port))
            connected = True
            print(f"Conexión exitosa en el intento {attempt + 1}.")
            break
        except socket.error as e:
            print(f"Intento {attempt + 1} fallido: {e}. Reintentando en {retry_delay} segundos...")
            time.sleep(retry_delay)

    if not connected:
        print("No se pudo establecer la conexión después de múltiples intentos. Saliendo.")
        exit(1)

    return sock

# Configuración del socket TCP
TCP_IP = ""  # Dirección IP del receptor
TCP_PORT = 5005           # Puerto
sock = connect_socket(TCP_IP, TCP_PORT)

# Configuración de la cámara RealSense
pipeline = rs.pipeline()
config = rs.config()

frame_width = 1280
frame_height = 720
frame_rate = 30

config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, frame_rate)
config.enable_stream(rs.stream.infrared, frame_width, frame_height, rs.format.y8, frame_rate)

# Inicia la cámara
pipeline.start(config)

# Parámetros de detección
luminosity_threshold_dark = 50  # Umbral para detectar oscuridad
luminosity_threshold_bright = 100  # Umbral para volver a RGB
luminosity_queue = deque(maxlen=10)  # Promedio móvil para suavizar cambios
current_mode = "RGB"  # Estado inicial

print(f"Transmitiendo video combinado a {TCP_IP}:{TCP_PORT}. Presiona 'Ctrl+C' para salir.")

try:
    while True:
        try:
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

            # Selecciona el frame a transmitir y mostrar
            if current_mode == "IR":
                frame_to_transmit = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
            else:
                frame_to_transmit = color_image

            # Codifica la imagen como JPEG
            _, encoded = cv2.imencode('.jpg', frame_to_transmit)
            data = encoded.tobytes()

            # Envía el tamaño del frame seguido del contenido
            sock.sendall(struct.pack(">I", len(data)) + data)

            # Muestra el stream localmente
            cv2.imshow("RealSense - Video Combinado", frame_to_transmit)

            # Detiene la transmisión si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except (socket.error, BrokenPipeError):
            print("Conexión perdida. Intentando reconectar...")
            sock.close()
            sock = connect_socket(TCP_IP, TCP_PORT)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    sock.close()

print("Transmisión finalizada.")
