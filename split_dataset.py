import os
import shutil
import random

def split_data(data_path, train_pct=0.8):
    # Define directories
    images_path = os.path.join(data_path, 'images')
    labels_path = os.path.join(data_path, 'labels')

    # Create output directories
    os.makedirs(os.path.join(data_path, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'val', 'labels'), exist_ok=True)

    # Get all image files
    all_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_files)

    # Split files
    num_train = int(len(all_files) * train_pct)
    train_files = all_files[:num_train]
    val_files = all_files[num_train:]

    # Copy files to new directories
    for f in train_files:
        shutil.copy(os.path.join(images_path, f), os.path.join(data_path, 'train', 'images', f))
        shutil.copy(os.path.join(labels_path, f.replace(os.path.splitext(f)[1], '.txt')), os.path.join(data_path, 'train', 'labels', f.replace(os.path.splitext(f)[1], '.txt')))

    for f in val_files:
        shutil.copy(os.path.join(images_path, f), os.path.join(data_path, 'val', 'images', f))
        shutil.copy(os.path.join(labels_path, f.replace(os.path.splitext(f)[1], '.txt')), os.path.join(data_path, 'val', 'labels', f.replace(os.path.splitext(f)[1], '.txt')))

    print(f"Successfully split dataset. Train: {len(train_files)} files, Validation: {len(val_files)} files.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--train_pct', type=float, default=0.9)
    args = parser.parse_args()
    split_data(args.datapath, args.train_pct)
