import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("0.0.0.0", port))
s.listen(16)
print(f"foreign app listening on 0.0.0.0:{port}", flush=True)
while True:
    c, _ = s.accept()
    try:
        c.recv(4096)
        c.sendall(b"HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 11\r\n\r\nFOREIGN-APP")
    finally:
        c.close()
