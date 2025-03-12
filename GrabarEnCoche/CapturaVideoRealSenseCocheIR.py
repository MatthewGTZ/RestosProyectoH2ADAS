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
    
    # 1) Crear pipeline y configurar solo el stream de INFRARROJO (IR)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, DESIRED_FPS)
    
    # Iniciar streaming
    profile = pipeline.start(config)
     # APAGAR el emisor infrarrojo desde el inicio
    device = profile.get_device()
    sensors = device.query_sensors()
    for s in sensors:
        if s.supports(rs.option.emitter_enabled):
            s.set_option(rs.option.emitter_enabled, 0)  # 0=OFF
    # Crear carpeta de salida
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_folder = f"C:/Users/matthias/Desktop/Trabajo/Programacion/GrabarEnCoche/capturas_{current_time}"
    os.makedirs(output_folder, exist_ok=True)
    
    # VideoWriter (solo infrarrojo)
    ir_filename = os.path.join(output_folder, f"video_IR_{current_time}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ir_out = cv2.VideoWriter(ir_filename, fourcc, DESIRED_FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    print("** Arrancando RealSense (solo infrarrojo).")
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
            # Capturar frames (solo IR)
            frames = pipeline.wait_for_frames()
            ir_frame = frames.get_infrared_frame()
            if not ir_frame:
                continue

            # Convertir a numpy
            ir_image = np.asanyarray(ir_frame.get_data())

            # Mostrar en vivo (opcional)
            cv2.imshow("Infrarrojo", ir_image)

            # Convertir IR a 3 canales (BGR) antes de guardar
            ir_bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
            ir_out.write(ir_bgr)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Liberar recursos
        pipeline.stop()
        ir_out.release()
        cv2.destroyAllWindows()

    print("Grabación finalizada.")
    print(f"Archivo de video: {ir_filename}")

if __name__ == "__main__":
    main()
