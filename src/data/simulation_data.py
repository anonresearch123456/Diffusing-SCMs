import os
import json
import h5py
import numpy as np
from pathlib import Path
from pathlib import Path
import json
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset


def gkern(kernlen=21, nsig=3):
    import numpy
    import scipy.stats as st
    
    """Returns a 2D Gaussian kernel array."""
    
    interval = (2*nsig+1.)/(kernlen)
    x = numpy.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = numpy.diff(st.norm.cdf(x))
    kernel_raw = numpy.sqrt(numpy.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def simulate_and_save_data(
    n: int = 512,
    image_size: int = 32,
    quadrant_size: int = 16,
    nsig: float = 5.0,
    noise_std: float = 0.01,
    val_frac: float = 0.2,
    out_dir: str = "sim_data"
):
    """
    Simulate data following the recipe:
    
    - Create a total of 2*n subjects for the training/validation set.
    - For group 0 (first n subjects): label=0, cf ~ Uniform(1,4), mf ~ Uniform(1,4).
    - For group 1 (last n subjects): label=1, cf ~ Uniform(3,6), mf ~ Uniform(3,6).
    - The simulated image is created in 4 quadrants using a Gaussian kernel multiplied by
      either mf (top-left & bottom-right) or cf (top-right & bottom-left). Noise is added.
    - Shuffle the dataset and split into train and val.
    
    Additionally, a separate test set is created (of size equal to the validation set) with the
    confounding variable effects reversed:
      - For subjects with label 0: cf ~ Uniform(3,6) (instead of 1,4).
      - For subjects with label 1: cf ~ Uniform(1,4) (instead of 3,6).
    The mf values are generated as in the training set.
    
    Two HDF5 files are written:
      - `sim_trainval.h5`: Contains all training/validation samples.
      - `sim_test.h5`: Contains the test samples.
      
    A JSON file `split.json` is also written that contains the split of the keys into
    "train", "val", and "test". The keys (record IDs) are strings.
    """
    np.random.seed(1)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Simulate train/val data ---
    total = 2 * n  # total subjects for train+val
    # Labels: first n subjects are 0, next n subjects are 1
    labels = np.zeros(total)
    labels[n:] = 1

    # Confounding (cf) and major (mf) effects for each group:
    cf = np.empty(total)
    mf = np.empty(total)
    cf[:n] = np.random.uniform(1, 4, size=n)
    cf[n:] = np.random.uniform(3, 6, size=n)
    mf[:n] = np.random.uniform(1, 4, size=n)
    mf[n:] = np.random.uniform(3, 6, size=n)

    # Pre-calculate the Gaussian kernel for the quadrant.
    kernel = gkern(quadrant_size, nsig)

    # Allocate images array (store as 2D; no channel dimension)
    images = np.empty((total, image_size, image_size), dtype=np.float32)
    for i in range(total):
        # Create an empty image.
        img = np.zeros((image_size, image_size), dtype=np.float32)
        # Top-left quadrant: rows 0:16, cols 0:16 = mf * kernel
        img[0:quadrant_size, 0:quadrant_size] = kernel * mf[i]
        # Top-right quadrant: rows 0:16, cols 16:32 = cf * kernel
        img[0:quadrant_size, quadrant_size:image_size] = kernel * cf[i]
        # Bottom-left quadrant: rows 16:32, cols 0:16 = cf * kernel
        img[quadrant_size:image_size, 0:quadrant_size] = kernel * cf[i]
        # Bottom-right quadrant: rows 16:32, cols 16:32 = mf * kernel
        img[quadrant_size:image_size, quadrant_size:image_size] = kernel * mf[i]
        # Add noise
        img += np.random.normal(0, noise_std, size=(image_size, image_size))
        images[i] = img

    # Shuffle the train/val data
    indices = np.arange(total)
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]
    cf = cf[indices]
    mf = mf[indices]

    cf_std = (cf - cf.mean()) / cf.std()

    # --- Create HDF5 file for train/val ---
    trainval_h5_path = out_path / "sim_trainval.h5"
    with h5py.File(trainval_h5_path, "w") as f:
        # We also record additional variables if needed.
        keys_trainval = []
        for idx in range(total):
            key = str(idx)  # unique id as string
            grp = f.create_group(key)
            # Save the image (stored as 2D; the dataset will add a channel dimension on load)
            grp.create_dataset("img", data=images[idx])
            grp.create_dataset("label", data=labels[idx])
            grp.create_dataset("cf", data=cf[idx])
            grp.create_dataset("mf", data=mf[idx])
            grp.create_dataset("cf_std", data=cf_std[idx])
            keys_trainval.append(key)

    # --- Split into train and validation ---
    n_val = int(val_frac * total)
    n_train = total - n_val
    # Here we simply use the first n_train keys for training and the remaining for validation.
    train_keys = keys_trainval[:n_train]
    val_keys = keys_trainval[n_train:]

    # --- Simulate test data with reversed confounding effects ---
    # Test set size is set equal to the validation set size.
    test_size = n_val
    # For balanced groups, we split test_size approximately in half.
    n_test_group0 = test_size // 2
    n_test_group1 = test_size - n_test_group0

    test_labels = np.concatenate([np.zeros(n_test_group0), np.ones(n_test_group1)])
    test_cf = np.empty(test_size)
    test_mf = np.empty(test_size)
    # Reverse cf: group 0 now gets uniform(3,6), group 1 gets uniform(1,4)
    test_cf[:n_test_group0] = np.random.uniform(3, 6, size=n_test_group0)
    test_cf[n_test_group0:] = np.random.uniform(1, 4, size=n_test_group1)
    # mf is generated as before:
    test_mf[:n_test_group0] = np.random.uniform(1, 4, size=n_test_group0)
    test_mf[n_test_group0:] = np.random.uniform(3, 6, size=n_test_group1)

    test_images = np.empty((test_size, image_size, image_size), dtype=np.float32)
    for i in range(test_size):
        img = np.zeros((image_size, image_size), dtype=np.float32)
        # Use the same quadrant scheme
        img[0:quadrant_size, 0:quadrant_size] = kernel * test_mf[i]
        img[0:quadrant_size, quadrant_size:image_size] = kernel * test_cf[i]
        img[quadrant_size:image_size, 0:quadrant_size] = kernel * test_cf[i]
        img[quadrant_size:image_size, quadrant_size:image_size] = kernel * test_mf[i]
        img += np.random.normal(0, noise_std, size=(image_size, image_size))
        test_images[i] = img

    # Shuffle test set
    test_indices = np.arange(test_size)
    np.random.shuffle(test_indices)
    test_images = test_images[test_indices]
    test_labels = test_labels[test_indices]
    test_cf = test_cf[test_indices]
    test_mf = test_mf[test_indices]

    test_cf_std = (test_cf - test_cf.mean()) / test_cf.std()

    # Write test HDF5 file
    test_h5_path = out_path / "sim_test.h5"
    with h5py.File(test_h5_path, "w") as f:
        test_keys = []
        for idx in range(test_size):
            key = str(idx)
            grp = f.create_group(key)
            grp.create_dataset("img", data=test_images[idx])
            grp.create_dataset("label", data=test_labels[idx])
            grp.create_dataset("cf", data=test_cf[idx])
            grp.create_dataset("mf", data=test_mf[idx])
            grp.create_dataset("cf_std", data=test_cf_std[idx])
            test_keys.append(key)

    # --- Write split file ---
    split = {
        "train": train_keys,
        "val": val_keys,
        "test": test_keys
    }
    split_file = out_path / "split.json"
    with open(split_file, "w") as f:
        json.dump(split, f, indent=4)

    print(f"Simulated train/val data saved to: {trainval_h5_path}")
    print(f"Simulated test data saved to: {test_h5_path}")
    print(f"Split file saved to: {split_file}")


class SimDataset(Dataset):
    """
    A simple Dataset for simulated data saved in an HDF5 file.
    
    Each record in the HDF5 file is expected to be a group with the following datasets:
      - "img": a 2D NumPy array (H x W). This dataset will be unsqueezed to have a channel dimension.
      - "label": a scalar (0 or 1).
      - "cf": confounding effect value.
      - "mf": major effect value.
      
    The unique key (record id) is used as the identifier.
    """
    def __init__(self, h5_path: str, ids: list):
        """
        Args:
            h5_path: Path to the HDF5 file.
            ids: List of keys (as strings) to load from the file.
        """
        self.h5_path = h5_path
        self.ids = ids
        # Open the file in read-only mode.
        self.file = h5py.File(self.h5_path, "r")
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        key = self.ids[index]
        group = self.file[key]
        # Load image as a NumPy array and add a channel dimension.
        img = group["img"][:]  # shape: (H, W)
        img = np.expand_dims(img, axis=0)  # new shape: (1, H, W)
        img_tensor = torch.as_tensor(img, dtype=torch.float32)
        
        # Load other values.
        label = torch.as_tensor(group["label"][()], dtype=torch.float32)
        cf = torch.as_tensor(group["cf"][()], dtype=torch.float32)
        mf = torch.as_tensor(group["mf"][()], dtype=torch.float32)
        id = torch.as_tensor(int(key), dtype=torch.long)
        cf_std = torch.as_tensor(group["cf_std"][()], dtype=torch.float32)

        sample = {
            "id": id,
            "img": img_tensor,
            "label": label,
            "cf": cf,
            "mf": mf,
            "cf_std": cf_std,
        }
        return sample
    
    def __del__(self):
        if hasattr(self, "file") and self.file:
            self.file.close()


class SimDataModule(pl.LightningDataModule):
    """
    A simple LightningDataModule for the simulated data.
    
    Assumes that the simulation HDF5 files and split.json file have been created.
    """
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        """
        Args:
            data_dir: Directory containing the simulated data (HDF5 files and split.json).
            batch_size: Batch size.
            num_workers: Number of workers for the DataLoader.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Paths to the h5 files.
        self.trainval_h5 = str(self.data_dir / "sim_trainval.h5")
        self.test_h5 = str(self.data_dir / "sim_test.h5")
        # Load split file.
        split_file = self.data_dir / "split.json"
        with open(split_file, "r") as f:
            self.splits = json.load(f)

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = SimDataset(self.trainval_h5, self.splits["train"])
            self.val_dataset = SimDataset(self.trainval_h5, self.splits["val"])
        if stage == "test" or stage is None:
            self.test_dataset = SimDataset(self.test_h5, self.splits["test"])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    # Run the simulation with default parameters.
    simulate_and_save_data(n=2100)
