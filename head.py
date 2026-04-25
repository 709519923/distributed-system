import torch
import socket
from common import send_data, recv_data
from model_utils import load_head

WORKER_HOST = "10.50.0.57"
WORKER_PORT = 29500

def generate(prompt: str, max_new_tokens: int = 50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Head] loading on {device} ...")
    tokenizer, parts = load_head(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((WORKER_HOST, WORKER_PORT))
        print("[Head] connected to worker")

        for step in range(max_new_tokens):
            with torch.no_grad():
                # embedding
                pos_ids = torch.arange(generated.shape[1], device=device).unsqueeze(0)
                hidden = parts["wte"](generated) + parts["wpe"](pos_ids)
                hidden = parts["drop"](hidden)

                # first 2 layers
                for layer in parts["layers"]:
                    hidden = layer(hidden)[0]

            # send to A800
            send_data(sock, hidden.cpu())
            logits = recv_data(sock)  # [1, seq_len, vocab]

            next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            generated = torch.cat([generated, next_token.to(device)], dim=1)

            word = tokenizer.decode(next_token[0])
            print(word, end="", flush=True)

            if next_token.item() == tokenizer.eos_token_id:
                break

        # signal done
        send_data(sock, None)

    print()
    return tokenizer.decode(generated[0])

if __name__ == "__main__":
    result = generate("The meaning of life is", max_new_tokens=40)
    print(f"\n--- Full output ---\n{result}")