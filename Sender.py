import argparse
import pickle
import socket

import cv2

from cryptography.hazmat.primitives import serialization

from logic import encrypt_image_payload_rsa

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

    pub_key_bytes = sock.recv(4096)
    public_key = serialization.load_pem_public_key(pub_key_bytes)

    if args.mode == 'ENCRYPTED':
        print("--- Mode: ENCRYPTED ---")
        msg_obj = encrypt_image_payload_rsa(public_key, img)
        # Send an unencrypted preview for demo/side-by-side visualization.
        # (Not required for decryption; purely for UI.)
        msg_obj['plain_preview'] = img
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