import numpy as np
import torch
import torch.nn as nn
import cv2

# --- ה-CNN שמפיק מפתחות מהתמונה ---
class ImageFeatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # קיבוע הזרע מבטיח משקולות זהות בכל פעם שהאובייקט נוצר
        torch.manual_seed(42)
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)) # מפיק 4 ערכים: K1, K2, K3, K4
        )
    def forward(self, x):
        return self.net(x).view(-1).detach().numpy()

# --- מערכת ה-Hyper-Chaos (2D-SICM) ---
def generate_chaos(x0, y0, a, size):
    x, y = np.zeros(size), np.zeros(size)
    x[0], y[0] = x0, y0
    for n in range(size - 1):
        x[n+1] = np.sin(np.pi * a * (y[n] + 3) * x[n] * (1 - x[n]))
        y[n+1] = np.sin(np.pi * a * (x[n+1] + 3) * y[n] * (1 - y[n]))
    return x, y

# --- DNA Encoding (מיקרו: המרת בייטים לאותיות) ---
DNA_MAP = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
def apply_dna_diffusion(img_flat, key_flat):
    # מדמה את הדיפוזיה של המאמר (XOR הוא הבסיס לחיבור DNA בינארי)
    return np.bitwise_xor(img_flat, key_flat)