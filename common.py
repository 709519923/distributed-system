import pickle
import struct
import socket

def send_data(conn: socket.socket, obj):
    payload = pickle.dumps(obj)
    conn.sendall(struct.pack(">I", len(payload)))
    conn.sendall(payload)

def recv_data(conn: socket.socket):
    raw = _recv_exact(conn, 4)
    length = struct.unpack(">I", raw)[0]
    payload = _recv_exact(conn, length)
    return pickle.loads(payload)

def _recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError("connection closed")
        buf.extend(chunk)
    return bytes(buf)