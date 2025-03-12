import socket

# Definir el host y puerto
HOST = '0.0.0.0'  # Escucha en todas las interfaces disponibles
PORT = 5672       # El puerto que quieras usar

# Crear el socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Servidor escuchando en {HOST}:{PORT}")

# Aceptar la conexión
client_socket, client_address = server_socket.accept()
print(f"Conexión aceptada desde {client_address}")

# Enviar un mensaje al cliente
client_socket.sendall(b"Hola desde Python!")

# Recibir mensaje del cliente
data = client_socket.recv(1024)
print(f"Mensaje recibido: {data.decode()}")

# Cerrar la conexión
client_socket.close()
server_socket.close()
