import torch
import socket
from common import send_data, recv_data
from model_utils import load_worker

HOST, PORT = "0.0.0.0", 29500

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Worker] loading on {device} ...")
    parts = load_worker(device)
    print("[Worker] model loaded, waiting for connections ...")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)

    while True:
        conn, addr = srv.accept()
        print(f"[Worker] connected by {addr}")
        with conn:
            try:
                while True:
                    msg = recv_data(conn)
                    if msg is None:
                        break
                    hidden = msg.to(device)
                    with torch.no_grad():
                        for layer in parts["layers"]:
                            hidden = layer(hidden)[0]
                        hidden = parts["ln_f"](hidden)
                        logits = parts["lm_head"](hidden)
                    send_data(conn, logits.cpu())
            except ConnectionError:
                print("[Worker] client disconnected")

if __name__ == "__main__":
    main()