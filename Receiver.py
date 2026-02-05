import pickle
import socket

import cv2

from cryptography.hazmat.primitives import serialization

from logic import decrypt_image_payload_rsa, generate_rsa_pair

APP_VERSION = "2026-02-04"

def receiver():
    priv, pub = generate_rsa_pair()
    pub_pem = pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)

    sock = socket.socket()
    sock.bind(('0.0.0.0', 5555)); sock.listen(1)
    print(f"Receiver Ready... (v{APP_VERSION})")
    
    while True:
        conn, _ = sock.accept()

        # Demo key exchange (not part of the paper)
        conn.sendall(pub_pem)

        data = b""
        while True:
            pkt = conn.recv(65536)
            if not pkt: break
            data += pkt
        
        marker = data[:9].decode(errors='ignore')
        msg = pickle.loads(data[9:])

        if marker == "START_ENC":
            plain_preview = msg.get('plain_preview')
            if plain_preview is not None:
                cv2.imshow("Plain (sent preview)", plain_preview)

            # Show what the encrypted image looks like (it should look like noise).
            cipher = msg.get("cipher")
            if cipher is not None:
                cv2.imshow("Cipher (pre-decrypt)", cipher)

            decrypted = decrypt_image_payload_rsa(priv, msg)
            cv2.imshow("Decrypted", decrypted)
        else:
            cv2.imshow("Plain", msg['image'])

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        conn.close()

if __name__ == "__main__":
    receiver()