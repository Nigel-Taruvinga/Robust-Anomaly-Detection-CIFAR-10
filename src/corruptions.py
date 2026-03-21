import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def gaussian_noise(img, severity=1):
    levels = [0.02, 0.05, 0.08, 0.12, 0.18]
    std = levels[severity - 1]

    x = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, std, x.shape).astype(np.float32)
    x_noisy = np.clip(x + noise, 0, 1)
    x_noisy = (x_noisy * 255).astype(np.uint8)
    return Image.fromarray(x_noisy)


def salt_pepper_noise(img, severity=1):
    levels = [0.01, 0.03, 0.05, 0.08, 0.12]
    amount = levels[severity - 1]

    x = np.array(img).copy()
    h, w, c = x.shape

    num_pixels = int(amount * h * w)

    # salt
    coords = (np.random.randint(0, h, num_pixels), np.random.randint(0, w, num_pixels))
    x[coords[0], coords[1], :] = 255

    # pepper
    coords = (np.random.randint(0, h, num_pixels), np.random.randint(0, w, num_pixels))
    x[coords[0], coords[1], :] = 0

    return Image.fromarray(x)


def gaussian_blur(img, severity=1):
    levels = [0.5, 1.0, 1.5, 2.0, 2.5]
    radius = levels[severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def brightness(img, severity=1):
    levels = [0.8, 0.6, 0.5, 0.4, 0.3]
    factor = levels[severity - 1]
    return ImageEnhance.Brightness(img).enhance(factor)


def cutout(img, severity=1):
    levels = [6, 10, 14, 18, 22]
    size = levels[severity - 1]

    x = np.array(img).copy()
    h, w, c = x.shape

    y = np.random.randint(0, h)
    x0 = np.random.randint(0, w)

    y1 = np.clip(y - size // 2, 0, h)
    y2 = np.clip(y + size // 2, 0, h)
    x1 = np.clip(x0 - size // 2, 0, w)
    x2 = np.clip(x0 + size // 2, 0, w)

    x[y1:y2, x1:x2, :] = 0
    return Image.fromarray(x)


# IMPORTANT: this must be AFTER the functions above
CORRUPTION_FUNCS = {
    "gaussian_noise": gaussian_noise,
    "salt_pepper": salt_pepper_noise,
    "gaussian_blur": gaussian_blur,
    "brightness": brightness,
    "cutout": cutout,
}
