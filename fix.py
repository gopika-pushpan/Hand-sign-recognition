import os
import shutil

src_root = 'hand_sign'
dst_root = 'dataset'

if not os.path.exists(dst_root):
    os.makedirs(dst_root)

for label in os.listdir(src_root):
    label_path = os.path.join(src_root, label)
    if os.path.isdir(label_path):
        for i, file in enumerate(os.listdir(label_path)):
            if file.endswith('.npy'):
                src_file = os.path.join(label_path, file)
                dst_file = os.path.join(dst_root, f"{label}_{i}.npy")
                shutil.move(src_file, dst_file)

print("✅ All .npy files successfully moved to dataset/")
