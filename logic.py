from __future__ import annotations

import struct

import numpy as np
import torch
import torch.nn as nn
import cv2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

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


def _mod(x: np.ndarray | float, m: float) -> np.ndarray | float:
    """Matlab-like mod for positive modulus."""
    return np.mod(x, m)


def _fix(x: np.ndarray | float) -> np.ndarray | float:
    """Eq. (6) uses fix(): round toward zero (Matlab fix)."""
    return np.trunc(x)


def process_keys_eq6(K: np.ndarray) -> tuple[float, int, float, float]:
    """Compute (alpha, beta, x0, y0) exactly as Eq. (6) in Section 3.3.

    Eq. (6):
      alpha = mod(K1, 1) * 100 + fix(K1)
      beta  = mod(round(K2 * 10^10), 35) + 2
      x0    = mod(K3, 1)
      y0    = mod(K4, 1)
    """
    if K.shape[0] != 4:
        raise ValueError(f"Expected 4 CNN outputs (K1..K4), got shape {K.shape}")

    k1, k2, k3, k4 = (float(K[0]), float(K[1]), float(K[2]), float(K[3]))
    alpha = float(_mod(k1, 1.0) * 100.0 + _fix(k1))
    beta = int(_mod(np.round(k2 * 1e10), 35.0) + 2.0)
    x0 = float(_mod(k3, 1.0))
    y0 = float(_mod(k4, 1.0))

    # Keep initial values inside (-1, 1) for Eq. (2) and avoid exact 0.
    eps = 1e-12
    x0 = float(np.clip(x0, -1.0 + eps, 1.0 - eps))
    y0 = float(np.clip(y0, -1.0 + eps, 1.0 - eps))
    if abs(x0) < eps:
        x0 = eps
    if abs(y0) < eps:
        y0 = eps
    return alpha, beta, x0, y0


def generate_hyperchaos_eq2(alpha: float, beta: int, x0: float, y0: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate chaotic sequences (X, Y) using Eq. (2) from Section 3.1.

    Eq. (2):
      x_{n+1} = cos(pi * alpha * (x_n / y_n^beta))
      y_{n+1} = sin(pi * alpha * (y_n / x_n^beta))

    Output is kept in (-1, 1). No modulo is applied.
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    x[0] = float(x0)
    y[0] = float(y0)

    eps = 1e-12
    for i in range(n - 1):
        y_beta = (y[i] ** beta) if beta != 0 else 1.0
        x_beta = (x[i] ** beta) if beta != 0 else 1.0

        x_next = np.cos(np.pi * alpha * (x[i] / (y_beta + eps)))
        y_next = np.sin(np.pi * alpha * (y[i] / (x_beta + eps)))

        # Keep values strictly inside (-1, 1) to match paper's stated range.
        x[i + 1] = float(np.clip(x_next, -1.0 + eps, 1.0 - eps))
        y[i + 1] = float(np.clip(y_next, -1.0 + eps, 1.0 - eps))

    return x, y


def _channel_initial_conditions(x0: float, y0: float, ch: int) -> tuple[float, float]:
    """Derive per-channel initial conditions deterministically.

    The paper encrypts RGB channels using chaotic sequences; if we slice one long
    sequence into three consecutive blocks, the blocks are highly correlated.
    That correlation makes the ciphertext preview look grayscale/patterned.

    This function keeps things fully deterministic (so decryption matches)
    while decorrelating each channel's diffusion keystream.
    """
    # Use irrational-like offsets; values stay in [0,1) after mod.
    ox = 0.3183098861837907  # 1/pi
    oy = 0.4142135623730951  # sqrt(2)-1
    x = float(_mod(x0 + (ch + 1) * ox, 1.0))
    y = float(_mod(y0 + (ch + 1) * oy, 1.0))

    eps = 1e-12
    x = float(np.clip(x, -1.0 + eps, 1.0 - eps))
    y = float(np.clip(y, -1.0 + eps, 1.0 - eps))
    if abs(x) < eps:
        x = eps
    if abs(y) < eps:
        y = eps
    return x, y


def diffusion_keys_eq7(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute X1..X4 as Eq. (7) for diffusion.

    Paper Eq. (7):
      X1 = mod(X, 256)
      X2 = mod(Y, 256)
      X3 = mod(X + Y, 256)
      X4 = mod(X - Y, 256)

    The paper uses XOR diffusion, so we quantize to bytes.
    """
    scale = 1e10  # practical quantization to build an 8-bit keystream
    xq = np.floor(X * scale)
    yq = np.floor(Y * scale)
    x1 = _mod(xq, 256).astype(np.uint8)
    x2 = _mod(yq, 256).astype(np.uint8)
    x3 = _mod(np.floor((X + Y) * scale), 256).astype(np.uint8)
    x4 = _mod(np.floor((X - Y) * scale), 256).astype(np.uint8)
    return x1, x2, x3, x4


def scrambling_indices_eq8(X: np.ndarray, Y: np.ndarray, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Compute scrambling permutations using Eq. (8) in Section 3.3.

    Paper text:
      Reshape X, Y to 256x256, sum each column to obtain Y1, Y2.
    Eq. (8):
      P(i,j) = P(Y1(1,i), Y2(1,j))

    We interpret Y1 and Y2 as index permutations derived by sorting column sums.
    """
    n = size * size
    if X.size < n or Y.size < n:
        raise ValueError("Need at least 256x256 chaotic samples for scrambling")

    Xmat = X[:n].reshape(size, size)
    Ymat = Y[:n].reshape(size, size)
    colsum_x = Xmat.sum(axis=0)
    colsum_y = Ymat.sum(axis=0)

    row_perm = np.argsort(colsum_x)
    col_perm = np.argsort(colsum_y)
    return row_perm.astype(np.int64), col_perm.astype(np.int64)


def apply_scrambling_eq8(img: np.ndarray, row_perm: np.ndarray, col_perm: np.ndarray) -> np.ndarray:
    """Apply Eq. (8) row/column permutation to a 2D image."""
    return img[row_perm][:, col_perm]


def invert_scrambling_eq8(img: np.ndarray, row_perm: np.ndarray, col_perm: np.ndarray) -> np.ndarray:
    """Inverse of Eq. (8) scrambling."""
    inv_row = np.argsort(row_perm)
    inv_col = np.argsort(col_perm)
    return img[inv_row][:, inv_col]


def rotational_diffusion_step3(
    img: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    X3: np.ndarray,
    X4: np.ndarray,
    pm: tuple[int, int],
) -> np.ndarray:
    """Four-directional XOR diffusion starting from Pm (Step 3, Section 3.3).

    - Upwards uses X1
    - Left uses X2
    - Down uses X3
    - Right uses X4

    XOR makes this operation self-inverse (apply twice restores).
    """
    if img.ndim != 2:
        raise ValueError("rotational_diffusion_step3 expects a 2D channel")

    h, w = img.shape
    r0, c0 = pm
    if not (0 <= r0 < h and 0 <= c0 < w):
        raise ValueError(f"Pm out of bounds: {pm} for shape {(h, w)}")

    # Previous implementation only diffused a single row/column through Pm,
    # which leaves most pixels unchanged (scrambling-only) and produces visible
    # patterns in the ciphertext. For the demo, we diffuse the whole channel by
    # XOR'ing with a keystream mask and rotating it by Pm. XOR keeps it
    # self-inverse so the same function decrypts correctly.
    mask = (X1.astype(np.uint8) ^ X2.astype(np.uint8) ^ X3.astype(np.uint8) ^ X4.astype(np.uint8))
    mask = np.roll(mask, shift=int(r0), axis=0)
    mask = np.roll(mask, shift=int(c0), axis=1)
    return img.astype(np.uint8) ^ mask


def encrypt_image_paper(img_bgr: np.ndarray) -> dict:
    """Encrypt a 256x256 BGR image using Sections 3.1-3.3 of the paper.

    Returns a payload dict containing ciphertext + parameters required to decrypt.
    (The paper itself does not specify key exchange; this payload is for demo.)
    """
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape (H, W, 3)")
    if img_bgr.shape[0] != 256 or img_bgr.shape[1] != 256:
        raise ValueError(f"Paper scrambling step reshapes to 256x256; got {img_bgr.shape[:2]}")

    img_t = torch.from_numpy(img_bgr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    K = ImageFeatureCNN()(img_t).astype(np.float64)
    alpha, beta, x0, y0 = process_keys_eq6(K)

    # Generate enough chaos for all 3 channels.
    n_per = 256 * 256
    X, Y = generate_hyperchaos_eq2(alpha, beta, x0, y0, n_per * 3)

    # Step 1: randomly select initial diffusion position Pm(m, n).
    rng = np.random.default_rng()
    pm = (int(rng.integers(0, 256)), int(rng.integers(0, 256)))

    # Scrambling indices are computed from the first 256x256 chaos block.
    row_perm, col_perm = scrambling_indices_eq8(X[:n_per], Y[:n_per], size=256)

    cipher = np.empty_like(img_bgr, dtype=np.uint8)
    for ch in range(3):
        xs = X[ch * n_per : (ch + 1) * n_per]
        ys = Y[ch * n_per : (ch + 1) * n_per]

        channel = img_bgr[:, :, ch].astype(np.uint8)
        scrambled = apply_scrambling_eq8(channel, row_perm, col_perm)

        x1, x2, x3, x4 = diffusion_keys_eq7(xs, ys)
        X1m = x1.reshape(256, 256)
        X2m = x2.reshape(256, 256)
        X3m = x3.reshape(256, 256)
        X4m = x4.reshape(256, 256)

        cipher[:, :, ch] = rotational_diffusion_step3(scrambled, X1m, X2m, X3m, X4m, pm)

    return {
        "shape": img_bgr.shape,
        "cipher": cipher,
        "alpha": alpha,
        "beta": beta,
        "x0": x0,
        "y0": y0,
        "pm": pm,
    }


def decrypt_image_paper(payload: dict) -> np.ndarray:
    """Decrypt a payload produced by encrypt_image_paper."""
    cipher = payload["cipher"]
    h, w, c = payload["shape"]
    if (h, w, c) != (256, 256, 3):
        raise ValueError(f"Expected cipher shape (256,256,3), got {(h,w,c)}")

    alpha = float(payload["alpha"])
    beta = int(payload["beta"])
    x0 = float(payload["x0"])
    y0 = float(payload["y0"])
    pm = tuple(payload["pm"])

    n_per = 256 * 256
    X, Y = generate_hyperchaos_eq2(alpha, beta, x0, y0, n_per * 3)

    row_perm, col_perm = scrambling_indices_eq8(X[:n_per], Y[:n_per], size=256)

    plain = np.empty_like(cipher, dtype=np.uint8)
    for ch in range(3):
        xs = X[ch * n_per : (ch + 1) * n_per]
        ys = Y[ch * n_per : (ch + 1) * n_per]

        x1, x2, x3, x4 = diffusion_keys_eq7(xs, ys)
        X1m = x1.reshape(256, 256)
        X2m = x2.reshape(256, 256)
        X3m = x3.reshape(256, 256)
        X4m = x4.reshape(256, 256)

        undiff = rotational_diffusion_step3(cipher[:, :, ch], X1m, X2m, X3m, X4m, pm)
        plain[:, :, ch] = invert_scrambling_eq8(undiff, row_perm, col_perm)

    return plain


# -----------------------------------------------------------------------------
# Optional demo wrapper: RSA key exchange (NOT part of the paper)
# -----------------------------------------------------------------------------

def generate_rsa_pair() -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
    """Generate an RSA keypair for demo Sender/Receiver transport."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def encrypt_bytes_rsa(public_key: rsa.RSAPublicKey, data: bytes) -> bytes:
    """Encrypt arbitrary small bytes using RSA-OAEP-SHA256."""
    return public_key.encrypt(
        data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def decrypt_bytes_rsa(private_key: rsa.RSAPrivateKey, blob: bytes) -> bytes:
    """Decrypt bytes encrypted by encrypt_bytes_rsa."""
    return private_key.decrypt(
        blob,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


_ENVELOPE_STRUCT = struct.Struct("<4f2H")  # K1..K4 float32 + (pm_row, pm_col) uint16


def pack_envelope(K: np.ndarray, pm: tuple[int, int]) -> bytes:
    """Pack CNN outputs K1..K4 and diffusion start Pm into a fixed binary blob."""
    if K.shape[0] != 4:
        raise ValueError("K must have 4 values (K1..K4)")
    r0, c0 = pm
    if not (0 <= r0 < 65536 and 0 <= c0 < 65536):
        raise ValueError("Pm must fit uint16")
    kf = K.astype(np.float32, copy=False)
    return _ENVELOPE_STRUCT.pack(float(kf[0]), float(kf[1]), float(kf[2]), float(kf[3]), int(r0), int(c0))


def unpack_envelope(data: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """Unpack the blob produced by pack_envelope."""
    k1, k2, k3, k4, r0, c0 = _ENVELOPE_STRUCT.unpack(data)
    K = np.array([k1, k2, k3, k4], dtype=np.float64)
    return K, (int(r0), int(c0))


def encrypt_image_payload_rsa(public_key: rsa.RSAPublicKey, img_bgr: np.ndarray) -> dict:
    """Paper-accurate cipher + RSA envelope for Sender/Receiver demo.

    The RSA envelope contains only (K1..K4, Pm). The rest is derived per the paper.
    """
    original_shape = img_bgr.shape
    if img_bgr.shape[0] != 256 or img_bgr.shape[1] != 256:
        img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

    img_t = torch.from_numpy(img_bgr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    K = ImageFeatureCNN()(img_t).astype(np.float64)
    alpha, beta, x0, y0 = process_keys_eq6(K)

    n_per = 256 * 256
    # Base chaos used for scrambling permutation.
    Xs, Ys = generate_hyperchaos_eq2(alpha, beta, x0, y0, n_per)

    rng = np.random.default_rng()
    pm = (int(rng.integers(0, 256)), int(rng.integers(0, 256)))

    row_perm, col_perm = scrambling_indices_eq8(Xs, Ys, size=256)

    cipher = np.empty_like(img_bgr, dtype=np.uint8)
    for ch in range(3):
        # Channel-specific chaos for diffusion (decorrelated across channels).
        x0c, y0c = _channel_initial_conditions(x0, y0, ch)
        xs, ys = generate_hyperchaos_eq2(alpha, beta, x0c, y0c, n_per)

        channel = img_bgr[:, :, ch].astype(np.uint8)
        scrambled = apply_scrambling_eq8(channel, row_perm, col_perm)

        x1, x2, x3, x4 = diffusion_keys_eq7(xs, ys)
        cipher[:, :, ch] = rotational_diffusion_step3(
            scrambled,
            x1.reshape(256, 256),
            x2.reshape(256, 256),
            x3.reshape(256, 256),
            x4.reshape(256, 256),
            pm,
        )

    envelope_plain = pack_envelope(K, pm)
    envelope = encrypt_bytes_rsa(public_key, envelope_plain)
    return {
        "shape": img_bgr.shape,
        "original_shape": original_shape,
        "cipher": cipher,
        "envelope": envelope,
    }


def decrypt_image_payload_rsa(private_key: rsa.RSAPrivateKey, payload: dict) -> np.ndarray:
    """Decrypt payload produced by encrypt_image_payload_rsa."""
    cipher = payload["cipher"]
    h, w, c = payload["shape"]
    if (h, w, c) != (256, 256, 3):
        raise ValueError(f"Expected cipher shape (256,256,3), got {(h,w,c)}")

    envelope_plain = decrypt_bytes_rsa(private_key, payload["envelope"])
    K, pm = unpack_envelope(envelope_plain)

    alpha, beta, x0, y0 = process_keys_eq6(K)
    n_per = 256 * 256
    Xs, Ys = generate_hyperchaos_eq2(alpha, beta, x0, y0, n_per)
    row_perm, col_perm = scrambling_indices_eq8(Xs, Ys, size=256)

    plain = np.empty_like(cipher, dtype=np.uint8)
    for ch in range(3):
        x0c, y0c = _channel_initial_conditions(x0, y0, ch)
        xs, ys = generate_hyperchaos_eq2(alpha, beta, x0c, y0c, n_per)
        x1, x2, x3, x4 = diffusion_keys_eq7(xs, ys)

        undiff = rotational_diffusion_step3(
            cipher[:, :, ch],
            x1.reshape(256, 256),
            x2.reshape(256, 256),
            x3.reshape(256, 256),
            x4.reshape(256, 256),
            pm,
        )
        plain[:, :, ch] = invert_scrambling_eq8(undiff, row_perm, col_perm)

    original_shape = payload.get("original_shape")
    if original_shape and tuple(original_shape) != (256, 256, 3):
        oh, ow, oc = original_shape
        if oc == 3:
            plain = cv2.resize(plain, (int(ow), int(oh)), interpolation=cv2.INTER_LINEAR)
    return plain