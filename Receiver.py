import socket
import pickle
import cv2
import numpy as np
from logic import apply_dna_diffusion, generate_chaos

def receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('0.0.0.0', 5555))
    sock.listen(1)
    print("Receiver Ready. Waiting for data...")

    while True:
        conn, addr = sock.accept()
        data = b""
        try:
            while True:
                packet = conn.recv(4096 * 16)
                if not packet: break
                data += packet
            
            if len(data) < 9:
                print("Error: Received data is too short.")
                continue

            # --- שלב המיקרו: הפרדת המרקר מה-Payload ---
            marker = data[:9].decode(errors='ignore') # 9 התווים הראשונים
            actual_payload = data[9:]                # כל השאר

            print(f"Detected Marker: {marker}")

            # טעינת המידע באמצעות pickle על השארית בלבד
            msg = pickle.loads(actual_payload)
            
            if msg['mode'] == 'PLAIN':
                print("Received PLAIN image.")
                cv2.imshow('Receiver Output (PLAIN)', msg['data'])
            
            elif msg['mode'] == 'ENCRYPTED':
                print("Received ENCRYPTED image. Starting Decryption...")
                enc_data = msg['data']
                x0, y0, a, b, h, w, c = msg['params']
                
                x_c, y_c = generate_chaos(x0, y0, a, b, h*w*c)
                key_s = (y_c * 255).astype(np.uint8)
                
                # פענוח: חיסור DNA ואז היפוך ערבול
                undiffused = apply_dna_diffusion(enc_data, key_s, decrypt=True)
                idx = np.argsort(x_c)
                inv_idx = np.zeros_like(idx); inv_idx[idx] = np.arange(len(idx))
                decrypted = undiffused[inv_idx].reshape(h, w, c)
                
                cv2.imshow('Encrypted Data (As seen in Wireshark)', enc_data.reshape(h, w, c))
                cv2.imshow('Decrypted Result', decrypted)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error processing data: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    receiver()