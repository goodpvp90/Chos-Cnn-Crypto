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
        # 1. הפקת מפתחות מהתמונה (K1-K4) באמצעות ה-CNN
        img_t = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0)/255.0
        k_vals = ImageFeatureCNN()(img_t)
        
        # 2. חישוב פרמטרים התחלתיים לכאוס (מיפוי K לערכים מתמטיים)
        x0, y0, a = (k_vals[0]%1), (k_vals[1]%1), 3.9 + (k_vals[2]%0.1)
        print(f"Chaos Params: x0={x0:.4f}, y0={y0:.4f}, a={a:.4f}")
        # 3. יצירת Key-streams וביצוע ערבול (Scrambling)
        x_c, y_c = generate_chaos(x0, y0, a, h*w*c)
        idx = np.argsort(x_c)
        scrambled = img.flatten()[idx]
        
        # 4. פעפוע (Diffusion) מבוסס DNA
        key_stream = (y_c * 255).astype(np.uint8)
        encrypted_data = apply_dna_diffusion(scrambled, key_stream)
        
        # יצירת אובייקט ההודעה
        msg_obj = {
            'mode': 'ENCRYPTED', 
            'data': encrypted_data, 
            'params': (x0, y0, a, h, w, c)
        }
        marker = "START_ENC".encode() # המרקר ל-Wireshark
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