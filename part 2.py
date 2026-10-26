import os
import glob
import random
import shutil
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
IMG_SIZE = 640
AUG_PER_IMAGE = 5
SPLIT = (0.70, 0.20, 0.10)
CLASS_NAMES = ["object"]
N_VERIFY = 10

random.seed(SEED)
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "dataset")
CHECK_DIR = os.path.join(SCRIPT_DIR, "part2_outputs")

IMG_DIRS = [os.path.join(SCRIPT_DIR, "raw_images"), os.path.join(SCRIPT_DIR, "images"), SCRIPT_DIR]
LBL_DIRS = [os.path.join(SCRIPT_DIR, "raw_labels"), os.path.join(SCRIPT_DIR, "labels"), SCRIPT_DIR]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")

def read_labels(path):
    """YOLO txt -> list of [cls, cx, cy, w, h] (normalized)."""
    boxes = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 5:
                boxes.append([int(p[0])] + [float(v) for v in p[1:]])
    return boxes


def collect_pairs():
    img_files = []
    for d in IMG_DIRS:
        if os.path.isdir(d):
            found = [os.path.join(d, f) for f in sorted(os.listdir(d))
                     if f.lower().endswith(IMG_EXT) and not f.startswith("comparison_grid_")]
            if found:
                img_files = found
                break
    pairs = []
    for ip in img_files:
        base = os.path.splitext(os.path.basename(ip))[0]
        lp = next((os.path.join(d, base + ".txt") for d in LBL_DIRS
                   if os.path.exists(os.path.join(d, base + ".txt"))), None)
        if lp is None:
            print(f"[WARNING] No label for {base}, skipping")
            continue
        pairs.append((ip, lp))
    return pairs

def letterbox(img, boxes, size=IMG_SIZE):
    """Aspect-preserving resize + gray padding to size x size; boxes updated."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = round(w * s), round(h * s)
    resized = cv.resize(img, (nw, nh), interpolation=cv.INTER_AREA)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = resized
    new = [[c, (cx * nw + px) / size, (cy * nh + py) / size, bw * nw / size, bh * nh / size]
           for c, cx, cy, bw, bh in boxes]
    return canvas, new

def hflip(img, boxes):
    return cv.flip(img, 1), [[c, 1 - cx, cy, w, h] for c, cx, cy, w, h in boxes]

def vflip(img, boxes):
    return cv.flip(img, 0), [[c, cx, 1 - cy, w, h] for c, cx, cy, w, h in boxes]

def rot90_cw(img, boxes):
    return cv.rotate(img, cv.ROTATE_90_CLOCKWISE), [[c, 1 - cy, cx, h, w] for c, cx, cy, w, h in boxes]

def rot90_ccw(img, boxes):
    return cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE), [[c, cy, 1 - cx, h, w] for c, cx, cy, w, h in boxes]

def scale_translate(img, boxes, s_range=(0.8, 1.2), t_max=0.10, min_keep=0.7):
    """Random scale about the center + translation. Returns None if the object gets cut too much."""
    s = random.uniform(*s_range)
    tx, ty = random.uniform(-t_max, t_max), random.uniform(-t_max, t_max)
    S = img.shape[0]
    M = np.float32([[s, 0, (1 - s) * S / 2 + tx * S],
                    [0, s, (1 - s) * S / 2 + ty * S]])
    out = cv.warpAffine(img, M, (S, S), flags=cv.INTER_LINEAR,
                        borderMode=cv.BORDER_CONSTANT, borderValue=(114, 114, 114))
    new = []
    for c, cx, cy, w, h in boxes:
        x1, y1 = s * (cx - w / 2 - 0.5) + 0.5 + tx, s * (cy - h / 2 - 0.5) + 0.5 + ty
        x2, y2 = s * (cx + w / 2 - 0.5) + 0.5 + tx, s * (cy + h / 2 - 0.5) + 0.5 + ty
        full = (x2 - x1) * (y2 - y1)
        x1c, y1c, x2c, y2c = max(x1, 0), max(y1, 0), min(x2, 1), min(y2, 1)
        if x2c <= x1c or y2c <= y1c or (x2c - x1c) * (y2c - y1c) / full < min_keep:
            return None
        new.append([c, (x1c + x2c) / 2, (y1c + y2c) / 2, x2c - x1c, y2c - y1c])
    return out, new

def brightness_contrast(img):
    return cv.convertScaleAbs(img, alpha=random.uniform(0.7, 1.3), beta=random.uniform(-30, 30))

def hsv_shift(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.randint(-10, 10)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.8, 1.2), 0, 255)
    return cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)

def gaussian_noise(img):
    noise = np.random.normal(0, random.uniform(5, 20), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def salt_pepper(img):
    amount = random.uniform(0.002, 0.01)
    out, r = img.copy(), np.random.rand(*img.shape[:2])
    out[r < amount / 2] = 0
    out[r > 1 - amount / 2] = 255
    return out

def blur(img):
    if random.random() < 0.5:
        k = random.choice([3, 5])
        return cv.GaussianBlur(img, (k, k), 0)
    k = random.choice([5, 7, 9])
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1
    M = cv.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), random.uniform(0, 180), 1)
    kernel = cv.warpAffine(kernel, M, (k, k))
    return cv.filter2D(img, -1, kernel / kernel.sum())


def augment(img, boxes):
    if random.random() < 0.5:
        img, boxes = hflip(img, boxes)
    if random.random() < 0.5:
        img, boxes = vflip(img, boxes)
    rot = random.choice([None, rot90_cw, rot90_ccw])
    if rot:
        img, boxes = rot(img, boxes)
    if random.random() < 0.7:
        res = scale_translate(img, boxes)
        if res:
            img, boxes = res
    if random.random() < 0.8:
        img = brightness_contrast(img)
    if random.random() < 0.6:
        img = hsv_shift(img)
    if random.random() < 0.5:
        img = random.choice([gaussian_noise, salt_pepper])(img)
    if random.random() < 0.4:
        img = blur(img)
    return img, boxes

def save_sample(img, boxes, split, name):
    ip = os.path.join(OUT_DIR, split, "images", name + ".jpg")
    lp = os.path.join(OUT_DIR, split, "labels", name + ".txt")
    cv.imwrite(ip, img, [cv.IMWRITE_JPEG_QUALITY, 95])
    with open(lp, "w") as f:
        for c, cx, cy, w, h in boxes:
            f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return ip, lp


def draw_yolo_boxes(img, label_path, color=(0, 255, 0)):
    """Same idea as Part 1: read the saved YOLO txt, convert to pixels, draw."""
    h, w = img.shape[:2]
    out = img.copy()
    for _, cx, cy, bw, bh in read_labels(label_path):
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out

def main():
    pairs = collect_pairs()
    if not pairs:
        print("[ERROR] No image/label pairs found.")
        return
    print(f"[INFO] {len(pairs)} image/label pairs found")

    random.shuffle(pairs)
    n = len(pairs)
    n_tr, n_va = int(SPLIT[0] * n), int(SPLIT[1] * n)
    splits = {"train": pairs[:n_tr], "val": pairs[n_tr:n_tr + n_va], "test": pairs[n_tr + n_va:]}

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    for s in splits:
        os.makedirs(os.path.join(OUT_DIR, s, "images"))
        os.makedirs(os.path.join(OUT_DIR, s, "labels"))
    os.makedirs(CHECK_DIR, exist_ok=True)

    aug_saved = []
    counts = {}
    for split, items in splits.items():
        counts[split] = 0
        for ip, lp in items:
            base = os.path.splitext(os.path.basename(ip))[0]
            img = cv.imread(ip)
            if img is None:
                print(f"[WARNING] Could not read {ip}")
                continue
            img, boxes = letterbox(img, read_labels(lp))

            save_sample(img, boxes, split, f"{base}_orig")
            counts[split] += 1
            for k in range(AUG_PER_IMAGE):
                a_img, a_boxes = augment(img.copy(), [b[:] for b in boxes])
                aug_saved.append(save_sample(a_img, a_boxes, split, f"{base}_aug{k}"))
                counts[split] += 1
        print(f"[INFO] {split}: {len(items)} base images -> {counts[split]} images")

    total = sum(counts.values())
    print(f"[INFO] Total images: {total}  ({'OK' if total >= 250 else 'LESS THAN 250!'})")

    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write("train: train/images\nval: val/images\ntest: test/images\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")

    picks = random.sample(aug_saved, min(N_VERIFY, len(aug_saved)))
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for ax, (ip, lp) in zip(axes.ravel(), picks):
        vis = draw_yolo_boxes(cv.imread(ip), lp)
        ax.imshow(cv.cvtColor(vis, cv.COLOR_BGR2RGB))
        ax.set_title(os.path.basename(ip), fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    save_path = os.path.join(CHECK_DIR, "augmentation_check.png")
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"[SUCCESS] Verification grid saved: {save_path}")


if __name__ == "__main__":
    main()