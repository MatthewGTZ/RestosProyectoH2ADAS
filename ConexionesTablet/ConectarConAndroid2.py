import socket

# Configura el puerto
PORT = 5672  # Cambia el puerto aquí si deseas usar otro

# Detecta la IP local de la máquina
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

# Crear el socket y enlazarlo a la IP y puerto
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((local_ip, PORT))
server_socket.listen(1)

print(f"Servidor escuchando en {local_ip}:{PORT}")

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