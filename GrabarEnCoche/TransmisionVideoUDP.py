import pyrealsense2 as rs
import cv2
import numpy as np
import socket
import struct

# Configuración del socket UDP
UDP_IP = ""  # Dirección IP del receptor
UDP_PORT = 5005           # Puerto
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Tamaño máximo de paquete UDP
MAX_PACKET_SIZE = 65000

# Selección del tipo de cámara
CAMERA_TYPE = input("Selecciona el tipo de imagen a transmitir ('color' o 'ir'): ").strip().lower()
if CAMERA_TYPE not in ["color", "ir"]:
    print("Error: Selección inválida. Usa 'color' o 'ir'.")
    exit(1)

# Configuración de la cámara RealSense
pipeline = rs.pipeline()
config = rs.config()

frame_width = 640
frame_height = 480
frame_rate = 30
if CAMERA_TYPE == "color":
    config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, frame_rate)
elif CAMERA_TYPE == "ir":
    config.enable_stream(rs.stream.infrared, frame_width, frame_height, rs.format.y8, frame_rate)

# Inicia la cámara
pipeline.start(config)

frame_id = 0  # Identificador único de frame

print(f"Transmitiendo video de {CAMERA_TYPE} a {UDP_IP}:{UDP_PORT}. Presiona 'Ctrl+C' para salir.")

try:
    while True:
        # Captura un frame de la cámara
        frames = pipeline.wait_for_frames()
        if CAMERA_TYPE == "color":
            frame = frames.get_color_frame()
        elif CAMERA_TYPE == "ir":
            frame = frames.get_infrared_frame()

        if not frame:
            continue

        # Convierte el frame en un array de numpy
        image = np.asanyarray(frame.get_data())

        # Codifica la imagen como JPEG
        _, encoded = cv2.imencode('.jpg', image)
        data = encoded.tobytes()

        # Agrega el ID del frame y su longitud como encabezado
        header = struct.pack(">II", frame_id, len(data))
        frame_id += 1

        # Fragmentar y enviar los datos
        for i in range(0, len(data), MAX_PACKET_SIZE):
            fragment = data[i:i + MAX_PACKET_SIZE]
            sock.sendto(header + fragment, (UDP_IP, UDP_PORT))

        # Muestra el stream localmente
        cv2.imshow(f"{CAMERA_TYPE.capitalize()} Stream", image)

        # Detiene la transmisión si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    sock.close()

print("Transmisión finalizada.")
