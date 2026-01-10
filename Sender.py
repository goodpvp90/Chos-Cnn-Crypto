import socket
import pickle
import argparse
import torch
import cv2
import numpy as np
from logic import ImageFeatureCNN, generate_chaos, apply_dna_diffusion

def sender():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', required=True, help="Receiver IP address")
    parser.add_argument('--mode', choices=['PLAIN', 'ENCRYPTED'], default='ENCRYPTED')
    args = parser.parse_args()

    # טעינת התמונה
    img = cv2.imread('photo.png')
    if img is None:
        print("Error: photo.jpg not found!")
        return
    
    h, w, c = img.shape
    
    if args.mode == 'ENCRYPTED':
        print(f"--- Mode: ENCRYPTED (Chaos + CNN + DNA) ---")
        # 1. הפקת מפתחות CNN
        img_t = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0)/255.0
        k = ImageFeatureCNN()(img_t)
        
        # 2. פרמטרים כאוטיים
        x0, y0, a, b = (k[0]%1), (k[1]%1), 4.5+(k[2]%0.5), 4.5+(k[3]%0.5)
        
        # 3. Scrambling (ערבול)
        x_c, y_c = generate_chaos(x0, y0, a, b, h*w*c)
        idx = np.argsort(x_c)
        scrambled = img.flatten()[idx]
        
        # 4. DNA Diffusion (דיפוזיה)
        key_stream = (y_c * 255).astype(np.uint8)
        encrypted_data = apply_dna_diffusion(scrambled, key_stream)
        
        msg_obj = {'mode': 'ENCRYPTED', 'data': encrypted_data, 'params': (x0, y0, a, b, h, w, c)}
        marker = b"START_ENC"
    else:
        print("--- Mode: PLAIN (Raw Image) ---")
        msg_obj = {'mode': 'PLAIN', 'data': img}
        marker = "START_PLN".encode() # המרקר ל-Wireshark

    # אריזת המידע
    payload = pickle.dumps(msg_obj)
    full_package = marker + payload  # הצמדת המרקר לתחילת הבייטים
    
    # פתיחת Socket ושליחה
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((args.ip, 5555))
        print(f"Sending {len(full_package)} bytes to {args.ip}...")
        sock.sendall(full_package)
        print("Transmission complete.")
    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    sender()