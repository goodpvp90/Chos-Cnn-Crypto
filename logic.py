import numpy as np
import torch
import torch.nn as nn
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

# --- CNN אקדמי (ללא שינוי) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return nn.functional.relu(x + self.conv(x))

class ImageFeatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=3), 
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, padding=2), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            ResidualBlock(64), 
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 4, 1), 
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
    def forward(self, x):
        return self.features(x).view(-1).detach().numpy()

# --- Chaos (2D-SICM) ---
def generate_chaos(alpha, beta, x0, y0, size):
    x, y = np.zeros(size), np.zeros(size)
    x[0], y[0] = x0, y0
    eps = 1e-10 
    for n in range(size - 1):
        x[n+1] = np.sin(alpha * np.pi * x[n] + (beta / (y[n] + eps))) % 1
        y[n+1] = np.sin(beta * np.pi * y[n] + (alpha / (x[n+1] + eps))) % 1
    return x, y

# --- מימוש Eq. (7) מהמאמר: יצירת 4 מטריצות לדיפוזיה ---
def get_diffusion_matrices(X, Y):
    # המאמר דורש X1, X2, X3, X4 מבוססי Modulo 256
    X1 = (X * 10**10).astype(np.int64) % 256
    X2 = (Y * 10**10).astype(np.int64) % 256
    X3 = ((X + Y) * 10**10).astype(np.int64) % 256
    X4 = ((X - Y) * 10**10).astype(np.int64) % 256
    return X1, X2, X3, X4 # החזרת ה-Tuple המלא

def apply_diffusion(img_flat, key_matrices):
    # הפונקציה מקבלת את רשימת המטריצות ומבצעת XOR רב-שכבתי
    res = img_flat.astype(np.uint8)
    for key in key_matrices:
        res = np.bitwise_xor(res, key.astype(np.uint8))
    return res

# --- RSA (ללא שינוי) ---
def generate_rsa_pair():
    private_key = rsa.generate_private_key(65537, 2048)
    return private_key, private_key.public_key()

def encrypt_keys_rsa(public_key, k):
    return public_key.encrypt(k.tobytes(), padding.OAEP(padding.MGF1(hashes.SHA256()), hashes.SHA256(), None))

def decrypt_keys_rsa(priv_key, blob):
    decrypted = priv_key.decrypt(blob, padding.OAEP(padding.MGF1(hashes.SHA256()), hashes.SHA256(), None))
    return np.frombuffer(decrypted, dtype=np.float32)