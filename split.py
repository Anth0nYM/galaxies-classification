# Script para divisão do dataset em treino, validação e teste.

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

original_file_path = "database/Binary_2_5_dataset.h5"
new_file_path = "Binary_2_5_dataset.h5"

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

with h5py.File(original_file_path, 'r') as original_file:
    labels = original_file['labels'][:]
    images = original_file['images'][:]
    num_samples = len(labels)

    indices = np.arange(num_samples)

    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, labels,
        test_size=(1 - train_ratio),
        stratify=labels,
        random_state=42
    )

    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices, temp_labels,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        stratify=temp_labels,
        random_state=42
    )

    split = np.zeros(num_samples, dtype=np.uint8)
    split[train_indices] = 0  # Train
    split[val_indices] = 1    # Val
    split[test_indices] = 2   # Test

    with h5py.File(new_file_path, 'w') as new_file:
        new_file.create_dataset('images', data=images)
        new_file.create_dataset('labels', data=labels)
        new_file.create_dataset('split', data=split)

    print(f"Novo arquivo salvo como {new_file_path}")
    print(f"Divisão: {len(train_indices)} treino, "
          f"{len(val_indices)} validação, "
          f"{len(test_indices)} teste.")
