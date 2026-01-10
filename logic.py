import numpy as np
import torch
import torch.nn as nn
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

# --- Residual Block matching Grey labels (2x2 conv) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Grey blocks inside the diagram represent 2x2 convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 2, padding=0),
            nn.ZeroPad2d((0, 1, 0, 1)), # Padding to maintain spatial dimensions for the skip connection
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 2, padding=0),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        # Implements the skip connection (x + conv(x)) shown in the diagram
        return nn.functional.relu(x + self.conv(x))

class ImageFeatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        
        self.features = nn.Sequential(
            # 1. INITIAL LAYERS
            nn.Conv2d(3, 16, 7, padding=3),  # Orange block: 7x7 conv
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, padding=2), # Green block: 5x5 conv
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), # Light Blue block: 3x3 conv
            nn.ReLU(),
            
            # 2. GENERATOR DASHED BOX
            ResidualBlock(64), # 3 Residual Blocks as shown in the diagram
            ResidualBlock(64),
            ResidualBlock(64),
            
            # 3. ADDITIONAL LAYERS
            # First Blue layer: 3x3 transposeconv
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            
            # Second Blue layer: 3x3 transposeconv
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Last Orange layer: 7x7 conv
            # Reduces to 4 channels to provide parameters for alpha, beta, x0, y0
            nn.Conv2d(16, 4, 7, padding=3), 
            
            # 4. FIX: GLOBAL POOLING
            # This condenses the high-resolution feature maps into exactly 4 values.
            # This fix prevents the "Data too long for key size" RSA error.
            nn.AdaptiveAvgPool2d((1, 1)) 
        )

    def forward(self, x):
        # Returns a flat array of 4 floating point values
        return self.features(x).view(-1).detach().numpy()

# --- Chaos (2D-SICM) ---
def generate_chaos(alpha, beta, x0, y0, size):
    x, y = np.zeros(size), np.zeros(size)
    x[0], y[0] = x0, y0
    eps = 1e-10 
    for n in range(size - 1):
        # Implementation of the 2D Sine-Iterative Chaotic Map formulas
        x[n+1] = np.sin(alpha * np.pi * x[n] + (beta / (y[n] + eps))) % 1
        y[n+1] = np.sin(beta * np.pi * y[n] + (alpha / (x[n+1] + eps))) % 1
    return x, y

# --- Diffusion Matrices: Implements Eq. (7) from the paper ---
def get_diffusion_matrices(X, Y):
    # Generates 4 diffusion matrices based on chaotic sequences X and Y
    X1 = (X * 10**10).astype(np.int64) % 256
    X2 = (Y * 10**10).astype(np.int64) % 256
    X3 = ((X + Y) * 10**10).astype(np.int64) % 256
    X4 = ((X - Y) * 10**10).astype(np.int64) % 256
    return X1, X2, X3, X4

def apply_diffusion(img_flat, key_matrices):
    # Performs multi-layer XOR diffusion using the key matrices
    res = img_flat.astype(np.uint8)
    for key in key_matrices:
        res = np.bitwise_xor(res, key.astype(np.uint8))
    return res

# --- RSA Secure Key Exchange ---
def generate_rsa_pair():
    # Generates a 2048-bit RSA key pair
    private_key = rsa.generate_private_key(65537, 2048)
    return private_key, private_key.public_key()

def encrypt_keys_rsa(public_key, k):
    # Encrypts the CNN-extracted keys using RSA-OAEP
    # The fix in ImageFeatureCNN ensures k.tobytes() is small enough for RSA.
    return public_key.encrypt(
        k.tobytes(), 
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()), 
            algorithm=hashes.SHA256(), 
            label=None
        )
    )

def decrypt_keys_rsa(priv_key, blob):
    # Decrypts the keys on the receiver side
    decrypted = priv_key.decrypt(
        blob, 
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()), 
            algorithm=hashes.SHA256(), 
            label=None
        )
    )
    return np.frombuffer(decrypted, dtype=np.float32)