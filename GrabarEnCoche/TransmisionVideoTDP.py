import pyrealsense2 as rs
import cv2
import numpy as np
import socket
import struct

# Configuración del socket TCP
TCP_IP = ""  # Dirección IP del receptor
TCP_PORT = 5005           # Puerto
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((TCP_IP, TCP_PORT))  # Conecta al receptor

# Configuración de la cámara RealSense
pipeline = rs.pipeline()
config = rs.config()

frame_width = 640
frame_height = 480
frame_rate = 30

# Habilitar el stream del giroscopio
config.enable_stream(rs.stream.gyro)

# Selección del tipo de cámara
CAMERA_TYPE = input("Selecciona el tipo de imagen a transmitir ('color' o 'ir'): ").strip().lower()
if CAMERA_TYPE == "color":
    config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, frame_rate)
elif CAMERA_TYPE == "ir":
    config.enable_stream(rs.stream.infrared, frame_width, frame_height, rs.format.y8, frame_rate)
else:
    print("Error: Selección inválida. Usa 'color' o 'ir'.")
    exit(1)

# Inicia la cámara
pipeline.start(config)

def get_camera_orientation(gyro_data):
    """Determina la orientación de la cámara en función de los datos del giroscopio."""
    # Usamos la orientación sobre los ejes para determinar el flip
    if gyro_data.z > 1:  # Cámara invertida verticalmente
        return 'vertical_flip'
    elif gyro_data.y > 1:  # Cámara rotada horizontalmente
        return 'horizontal_flip'
    else:
        return 'normal'

print(f"Transmitiendo video de {CAMERA_TYPE} a {TCP_IP}:{TCP_PORT}. Presiona 'Ctrl+C' para salir.")

try:
    while True:
        # Captura un frame de la cámara
        frames = pipeline.wait_for_frames()
        if CAMERA_TYPE == "color":
            frame = frames.get_color_frame()
        elif CAMERA_TYPE == "ir":
            frame = frames.get_infrared_frame()

        gyro_frame = frames.first_or_default(rs.stream.gyro)

        if not frame or not gyro_frame:
            continue

        # Convierte el frame en un array de numpy
        image = np.asanyarray(frame.get_data())

        # Obtén los datos del giroscopio
        gyro_data = gyro_frame.as_motion_frame().get_motion_data()

        # Determina la orientación y aplica transformaciones
        orientation = get_camera_orientation(gyro_data)
        if orientation == 'vertical_flip':
            image = cv2.flip(image, 0)
        elif orientation == 'horizontal_flip':
            image = cv2.flip(image, 1)

        # Codifica la imagen como JPEG
        _, encoded = cv2.imencode('.jpg', image)
        data = encoded.tobytes()

        # Envía el tamaño del frame seguido del contenido
        sock.sendall(struct.pack(">I", len(data)) + data)

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
