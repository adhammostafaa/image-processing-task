import os
import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"[INFO] Looking for files inside: {SCRIPT_DIR}")

POSSIBLE_IMG_DIRS = [
    os.path.join(SCRIPT_DIR, "raw_images"),
    os.path.join(SCRIPT_DIR, "images"),
    SCRIPT_DIR
]
POSSIBLE_LBL_DIRS = [
    os.path.join(SCRIPT_DIR, "raw_labels"),
    os.path.join(SCRIPT_DIR, "labels"),
    SCRIPT_DIR
]

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

image_files = []
RAW_IMAGES_DIR = None
for d in POSSIBLE_IMG_DIRS:
    if os.path.exists(d):
        found = [
            os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.lower().endswith(IMG_EXTENSIONS) and not f.startswith("comparison_grid_")
        ]
        if len(found) > 0:
            image_files = found
            RAW_IMAGES_DIR = d
            break

RAW_LABELS_DIR = SCRIPT_DIR
for d in POSSIBLE_LBL_DIRS:
    if os.path.exists(d):
        txt_found = [f for f in os.listdir(d) if f.lower().endswith(".txt")]
        if len(txt_found) > 0:
            RAW_LABELS_DIR = d
            break

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "part1_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INFO] Images folder detected: {RAW_IMAGES_DIR} ({len(image_files)} images found)")
print(f"[INFO] Labels folder detected: {RAW_LABELS_DIR}")
print(f"[INFO] Output folder created at: {OUTPUT_DIR}")

if len(image_files) == 0:
    print("\n[ERROR] No .png or .jpg images were found!")
    print(f"Please put your 50 Blender images either directly inside '{SCRIPT_DIR}' or inside a '{os.path.join(SCRIPT_DIR, 'raw_images')}' folder.")
    exit()

def parse_yolo_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        print(f"  [WARNING] Label file not found: {label_path}")
        return boxes
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            
            x_min = int((cx - bw / 2) * img_w)
            y_min = int((cy - bh / 2) * img_h)
            x_max = int((cx + bw / 2) * img_w)
            y_max = int((cy + bh / 2) * img_h)
            boxes.append((class_id, x_min, y_min, x_max, y_max))
    return boxes

def process_and_visualize_sample(img_path, label_path, save_path):
    img_bgr = cv.imread(img_path)
    if img_bgr is None:
        print(f"  [ERROR] Could not read image: {img_path}")
        return
    img_h, img_w = img_bgr.shape[:2]
    
    panel1_gt = img_bgr.copy()
    gt_boxes = parse_yolo_boxes(label_path, img_w, img_h)
    for (_, x1, y1, x2, y2) in gt_boxes:
        cv.rectangle(panel1_gt, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv.putText(panel1_gt, "GT YOLO", (x1, max(y1 - 10, 20)), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    smoothed_bgr = cv.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv.cvtColor(smoothed_bgr, cv.COLOR_BGR2GRAY)
    blurred_gray = cv.GaussianBlur(gray, (9, 9), 0)

    blurred_bgr = cv.GaussianBlur(smoothed_bgr, (9, 9), 0)
    hsv = cv.cvtColor(blurred_bgr, cv.COLOR_BGR2HSV)

    med_h, med_s, med_v = np.median(hsv.reshape(-1, 3), axis=0)
    lower_ground = np.array([max(0, med_h - 15), max(30, med_s - 45), max(40, med_v - 50)], dtype=np.uint8)
    upper_ground = np.array([min(179, med_h + 15), 255, min(200, med_v + 50)], dtype=np.uint8)
    
    ground_mask = cv.inRange(hsv, lower_ground, upper_ground)
    raw_mask = cv.bitwise_not(ground_mask)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    mask_opened = cv.morphologyEx(raw_mask, cv.MORPH_OPEN, kernel, iterations=1)
    clean_mask = cv.morphologyEx(mask_opened, cv.MORPH_CLOSE, kernel_close, iterations=2)

    sigma = 0.5
    median = np.median(blurred_gray)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1 + sigma) * median))
    edges = cv.Canny(clean_mask, lower, upper)
    contours, _ = cv.findContours(clean_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    panel5_comparison = img_bgr.copy()
    panel5_comparison[edges > 0] = [255, 255, 0]
    
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        cv.drawContours(panel5_comparison, [largest_contour], -1, (0, 255, 255), 2)
        
        cx_min, cy_min, cw, ch = cv.boundingRect(largest_contour)
        cv.rectangle(panel5_comparison, (cx_min, cy_min), (cx_min + cw, cy_min + ch), (0, 0, 255), 3)
        cv.putText(panel5_comparison, "Classical Box", (cx_min, min(cy_min + ch + 25, img_h - 10)), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for (_, x1, y1, x2, y2) in gt_boxes:
        cv.rectangle(panel5_comparison, (x1, y1), (x2, y2), (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    
    axes[0].imshow(cv.cvtColor(panel1_gt, cv.COLOR_BGR2RGB))
    axes[0].set_title("1. Original + GT YOLO Box")
    
    axes[1].imshow(blurred_gray, cmap="gray")
    axes[1].set_title("2. Smoothed (Bilateral+Gaussian)")
    
    axes[2].imshow(raw_mask, cmap="gray")
    axes[2].set_title("3. Raw HSV Mask (cv.inRange)")
    
    axes[3].imshow(clean_mask, cmap="gray")
    axes[3].set_title("4. Morphologically Cleaned")
    
    axes[4].imshow(cv.cvtColor(panel5_comparison, cv.COLOR_BGR2RGB))
    axes[4].set_title("5. Contours & Box Comparison\n(Green=GT, Red=Classical)")
    
    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved comparison grid: {save_path}")

for idx, img_path in enumerate(image_files[:5], 1):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(RAW_LABELS_DIR, f"{base_name}.txt")
    save_path = os.path.join(OUTPUT_DIR, f"comparison_grid_{idx}_{base_name}.png")
    process_and_visualize_sample(img_path, label_path, save_path)