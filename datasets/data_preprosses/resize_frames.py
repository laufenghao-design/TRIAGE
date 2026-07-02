import os

cpu_num = 10
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

import cv2
from tqdm import tqdm

cv2.setNumThreads(cpu_num)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

ROOT_DIR = "./datasets/AutoLaparo_Task1"
SRC_DIR = os.path.join(ROOT_DIR, "frames")
DST_DIR = os.path.join(ROOT_DIR, "frames_256")
TARGET_SIZE = (256, 256)

video_names = sorted(os.listdir(SRC_DIR))
total = 0

for video_name in video_names:
    src_video = os.path.join(SRC_DIR, video_name)
    dst_video = os.path.join(DST_DIR, video_name)
    if not os.path.isdir(src_video):
        continue
    os.makedirs(dst_video, exist_ok=True)

    frames = sorted(os.listdir(src_video))
    for fname in tqdm(frames, desc=f"Video {video_name}"):
        src_path = os.path.join(src_video, fname)
        dst_path = os.path.join(dst_video, fname)
        if os.path.exists(dst_path):
            continue
        img = cv2.imread(src_path)
        if img is None:
            print(f"Warning: failed to read {src_path}")
            continue
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(dst_path, img)
        total += 1

print(f"Done! Resized {total} frames to {TARGET_SIZE}")
print(f"Output: {DST_DIR}")
