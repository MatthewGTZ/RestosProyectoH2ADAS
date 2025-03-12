import pyrealsense2 as rs
import cv2
import os
import time
from datetime import datetime
import numpy as np

def main():
    # -------------------------------------------------
    # CONFIGURACIÓN
    # -------------------------------------------------
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    DESIRED_FPS = 30  # Pedimos 30 fps a la cámara y grabamos también a 30
    
    # 1) Crear pipeline y configurar solo el stream de COLOR
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, DESIRED_FPS)
    
    # Iniciar streaming
    profile = pipeline.start(config)

    # (Opcional) Desactivar emisor infrarrojo si tu cámara lo soporta
    device = profile.get_device()
    # A veces el "depth_sensor" es [0] o [1], depende del modelo
    sensors = device.query_sensors()
    # Buscar sensor de profundidad para desactivar emisor
    for s in sensors:
        if s.get_info(rs.camera_info.name).lower().startswith('depth'):
            s.set_option(rs.option.emitter_enabled, 0)

    # Crear carpeta de salida
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_folder = f"C:/Users/matthias/Desktop/Trabajo/Programacion/GrabarEnCoche/capturas_{current_time}"
    os.makedirs(output_folder, exist_ok=True)
    
    # VideoWriter (solo color)
    color_filename = os.path.join(output_folder, f"video_color_{current_time}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    color_out = cv2.VideoWriter(color_filename, fourcc, DESIRED_FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    print("** Arrancando RealSense (solo color).")
    print("** Calentando 10 segundos para estabilizar la cámara...")

    # -------------------------------------------------
    # 2) DESCARTAR FRAMES DURANTE 10 SEGUNDOS
    # -------------------------------------------------
    warmup_start = time.time()
    while True:
        if time.time() - warmup_start >= 10:
            break
        # Simplemente esperamos frames sin hacer nada con ellos
        pipeline.wait_for_frames()

    print("** Calentamiento finalizado. Comenzando grabación de video.")
    print("Presiona 'q' para terminar.")

    try:
        while True:
            # Capturar frames (solo color)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convertir a numpy
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.flip(color_image, -1)
            # Mostrar en vivo (opcional)
            cv2.imshow("Color", color_image)

            # Guardar en el video
            color_out.write(color_image)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Liberar recursos
        pipeline.stop()
        color_out.release()
        cv2.destroyAllWindows()

    print("Grabación finalizada.")
    print(f"Archivo de video: {color_filename}")

if __name__ == "__main__":
    main()
