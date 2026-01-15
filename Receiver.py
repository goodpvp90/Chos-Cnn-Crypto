import socket, pickle, cv2, numpy as np
from logic import generate_rsa_pair, decrypt_keys_rsa, generate_chaos, get_diffusion_matrices, apply_diffusion
from cryptography.hazmat.primitives import serialization

def receiver():
    priv, pub = generate_rsa_pair()
    pub_pem = pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    
    sock = socket.socket()
    sock.bind(('0.0.0.0', 5555)); sock.listen(1)
    print("Receiver Ready...")
    
    while True:
        conn, _ = sock.accept()
        conn.sendall(pub_pem)
        
        data = b""
        while True:
            pkt = conn.recv(65536)
            if not pkt: break
            data += pkt
        
        marker = data[:9].decode(errors='ignore')
        msg = pickle.loads(data[9:])

        if marker == "START_ENC":
            k = decrypt_keys_rsa(priv, msg['envelope'])
            h, w, c = msg['shape']
            
            alpha = (k[0] % 1) * 100 + np.floor(k[0])
            beta = (round(k[1] * 10**10) % 35) + 2
            x0, y0 = (k[2] % 1), (k[3] % 1)
            #x0 += 0.000000001 
            X, Y = generate_chaos(alpha, beta, x0, y0, h*w*c)
            diff_keys = get_diffusion_matrices(X, Y)
            
            undiffused = apply_diffusion(msg['image'], diff_keys)
            
            idx = np.argsort(X)
            inv = np.zeros_like(idx); inv[idx] = np.arange(len(idx))
            decrypted = undiffused[inv].reshape(h, w, c)
            cv2.imshow("Decrypted", decrypted)
        else:
            cv2.imshow("Plain", msg['image'])

        cv2.waitKey(0)
        conn.close()

if __name__ == "__main__":
    receiver()