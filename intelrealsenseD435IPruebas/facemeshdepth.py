import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import datetime as dt

font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 100)
fontScale = .5
color = (0, 50, 255)
thickness = 1

# ====== Realsense ======
realsense_ctx = rs.context()
connected_devices = []
for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    print(f"{detected_camera}")
    connected_devices.append(detected_camera)
device = connected_devices[0]
pipeline = rs.pipeline()
config = rs.config()
background_removed_color = 153  # Grey

# ====== Mediapipe ======
mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=color)

# ====== Enable Streams ======
config.enable_device(device)

stream_res_x = 640
stream_res_y = 480
stream_fps = 30

config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# ====== Get depth Scale ======
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"\tDepth Scale for Camera SN {device} is: {depth_scale}")

# ====== Set clipping distance ======
clipping_distance_in_meters = 2
clipping_distance = clipping_distance_in_meters / depth_scale
print(f"\tConfiguration Successful for SN {device}")

# ====== Get and process images ====== 
print(f"Starting to capture images on SN: {device}")

while True:
    start_time = dt.datetime.today().timestamp()

    # Get and align frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not aligned_depth_frame or not color_frame:
        continue

    # Convert frames to arrays for processing
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Process face mesh
    results = face_mesh.process(color_image_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(
                image=color_image,
                landmark_list=face_landmarks,
                connections=mpFaceMesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=mpDraw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
            )
            
            # Calculate the center of the face
            h, w, _ = color_image.shape
            cx = int(face_landmarks.landmark[1].x * w)
            cy = int(face_landmarks.landmark[1].y * h)
            
            # Get depth value at the center of the face
            depth = depth_image[cy, cx] * depth_scale
            cv2.putText(color_image, f"Depth: {depth:.2f}m", (cx, cy), font, fontScale, color, thickness, cv2.LINE_AA)

    # Display image
    cv2.imshow("Face Mesh", color_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

print("Closing application")
pipeline.stop()
cv2.destroyAllWindows()
