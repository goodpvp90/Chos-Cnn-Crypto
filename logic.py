import numpy as np
import torch
import torch.nn as nn
import cv2

# --- CNN משופר להפקת מפתחות ---
class ImageFeatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42) # מבטיח משקולות קבועות לשחזור
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)) # מפיק 4 ערכים (K1-K4)
        )
    def forward(self, x):
        return self.net(x).view(-1).detach().numpy()

# --- מערכת Hyper-Chaos 2D-SICM משופרת ---
def generate_chaos(x0, y0, a, b, size):
    x, y = np.zeros(size), np.zeros(size)
    x[0], y[0] = x0, y0
    for n in range(size - 1):
        # מימוש נוסחת 2D-SICM מהמאמר (Sine-Iterative Chaotic Map)
        x[n+1] = np.sin(a * np.pi / (y[n] + 0.1)) % 1
        y[n+1] = np.sin(b * np.pi / (x[n+1] + 0.1)) % 1
    return x, y

# --- שכבת DNA: קידוד, פענוח וחיבור אלגברי ---
def dna_encode(arr):
    # הופך בייטים לרצף DNA (00:A, 01:C, 10:G, 11:T)
    bin_arr = np.unpackbits(arr.astype(np.uint8)).reshape(-1, 4, 2)
    dna = np.zeros((bin_arr.shape[0], 4), dtype=int)
    for i in range(2): dna += bin_arr[:,:,1-i] * (2**i)
    return dna # מחזיר מערך של ערכים 0-3

def dna_decode(dna_arr):
    # מחזיר DNA לבייטים
    bits = np.zeros((dna_arr.shape[0], 4, 2), dtype=np.uint8)
    bits[:,:,0] = (dna_arr >> 1) & 1
    bits[:,:,1] = dna_arr & 1
    return np.packbits(bits.flatten())

def dna_add(a, b): # חיבור DNA לפי חוק 1 (Mod 4)
    return (a + b) % 4

def dna_sub(a, b): # חיסור DNA לפענוח
    return (a - b + 4) % 4

def apply_dna_diffusion(img_flat, key_flat, decrypt=False):
    # דיפוזיה אלגברית במקום XOR פשוט
    img_dna = dna_encode(img_flat)
    key_dna = dna_encode(key_flat)
    if not decrypt:
        res_dna = dna_add(img_dna, key_dna)
    else:
        res_dna = dna_sub(img_dna, key_dna)
    return dna_decode(res_dna)