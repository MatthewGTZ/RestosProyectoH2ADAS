from PIL import Image
import numpy as np

# Cargar la imagen
image_path = 'C:\\Users\\matthias\\Desktop\\Trabajo\\Programacion\\Edicion Imagenes\\icono2app.png'  # Cambia esta ruta a la de tu imagen
image = Image.open(image_path).convert("RGBA")

# Convertir la imagen a un array de numpy
data = np.array(image)

# Definir el color del fondo gris (esto puede necesitar ajustes)
checker_gray = [192, 192, 192]  # Este es un color gris común, ajústalo si es necesario

# Encontrar píxeles que coincidan con el color de fondo y hacerlos transparentes
matches = np.all(data[:, :, :3] == checker_gray, axis=-1)
data[matches, 3] = 0  # Establecer el canal alpha a 0 donde se detecta el fondo

# Convertir de nuevo a imagen y guardar el resultado
icon_without_background = Image.fromarray(data, "RGBA")
output_path = 'icono_sin_fondo.png'  # Cambia esta ruta para guardar el resultado
icon_without_background.save(output_path)

print(f"Imagen procesada guardada en: {output_path}")