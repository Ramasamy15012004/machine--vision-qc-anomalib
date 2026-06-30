# tools/convert_bayer_to_mono.py
import cv2
import os
import glob

# ==================== CONFIGURATION ====================
# 1. Folder where your BayerRG8 images are currently stored
INPUT_DIR = "data/bayer"

# 2. Folder where you want to save the new Mono8 images
OUTPUT_DIR = "data/mono"

# 3. File extension of your source images (e.g., *.jpg, *.bmp, or *.png)
FILE_PATTERN = "*.jpg"
# =======================================================

def convert_bayer_to_mono():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] Created output directory: {OUTPUT_DIR}")

    # Find all matching files
    files = glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN))
    print(f"[INFO] Found {len(files)} images to convert.")

    for file_path in files:
        # Load the image
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[ERROR] Could not read: {file_path}")
            continue

        try:
            # Check if image is already 3-channel (BGR) or 1-channel (Raw Bayer)
            if len(img.shape) == 3:
                # If it has 3 channels, it is likely already a BGR image
                # We convert BGR to GRAY
                mono_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                suffix = "(BGR to Mono)"
            else:
                # If it has 1 channel, it is raw Bayer data
                # We use BayerRG2GRAY
                mono_img = cv2.cvtColor(img, cv2.COLOR_BayerRG2GRAY)
                suffix = "(Bayer to Mono)"

            # Define output path
            file_name = os.path.basename(file_path)
            output_path = os.path.join(OUTPUT_DIR, file_name)

            # Save the Mono8 image (1-channel)
            cv2.imwrite(output_path, mono_img)
            print(f"[OK] Converted: {file_name} {suffix}")

        except Exception as e:
            print(f"[ERROR] Failed to convert {file_path}: {e}")

    print("\n" + "="*40)
    print(" CONVERSION COMPLETE")
    print(f" Converted images are in: {OUTPUT_DIR}")
    print("="*40)

if __name__ == "__main__":
    convert_bayer_to_mono()
