import cv2
import numpy as np

# Cargar la imagen en formato RGB
imagen = cv2.imread('C:\\Users\\matthias\\Desktop\\Trabajo\\Programacion\\Edicion Imagenes\\icono2app.png')

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Crear una máscara para detectar el blanco (255 en escala de grises)
_, mascara = cv2.threshold(gris, 240, 255, cv2.THRESH_BINARY)

# Invertir la máscara para que el blanco sea 0 (transparente) y lo demás sea 255
mascara_invertida = cv2.bitwise_not(mascara)

# Agregar el canal alpha a la imagen original
rgba = cv2.cvtColor(imagen, cv2.COLOR_BGR2BGRA)
rgba[:, :, 3] = mascara_invertida

# Guardar la imagen con el fondo blanco convertido en transparente
cv2.imwrite('icono con transparencia.png', rgba)