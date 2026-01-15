import socket, pickle, torch, cv2, numpy as np, argparse
from logic import ImageFeatureCNN, generate_chaos, get_diffusion_matrices, apply_diffusion, encrypt_keys_rsa
from cryptography.hazmat.primitives import serialization

def sender():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', required=True)
    parser.add_argument('--mode', choices=['ENCRYPTED', 'PLAIN'], default='ENCRYPTED')
    args = parser.parse_args()

    img = cv2.imread('photo.png') 
    if img is None: return
    h, w, c = img.shape
    
    sock = socket.socket()
    sock.connect((args.ip, 5555))
    pub_key_bytes = sock.recv(1024)
    public_key = serialization.load_pem_public_key(pub_key_bytes)

    if args.mode == 'ENCRYPTED':
        print("--- Mode: ENCRYPTED ---")
        img_t = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0)/255.0
        k = ImageFeatureCNN()(img_t).astype(np.float32)
        
        alpha = (k[0] % 1) * 100 + np.floor(k[0])
        beta = (round(k[1] * 10**10) % 35) + 2
        x0, y0 = (k[2] % 1), (k[3] % 1)
        
        X, Y = generate_chaos(alpha, beta, x0, y0, h*w*c)
        idx = np.argsort(X) 
        scrambled = img.flatten()[idx]
        
        # תיקון: Unpacking של 4 המטריצות לפי המאמר
        diff_keys = get_diffusion_matrices(X, Y) 
        encrypted_image = apply_diffusion(scrambled, diff_keys)
        
        envelope = encrypt_keys_rsa(public_key, k)
        msg_obj = {'shape': (h, w, c), 'envelope': envelope, 'image': encrypted_image}
        marker = b"START_ENC"
    else:
        print("--- Mode: PLAIN ---")
        msg_obj = {'image': img}
        marker = b"START_PLN"

    sock.sendall(marker + pickle.dumps(msg_obj))
    sock.close()
    print(f"Sent successfully with {marker.decode()}")

if __name__ == "__main__":
    sender()