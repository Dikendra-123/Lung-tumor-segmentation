import os
from typing import Optional, Callable, Tuple

import numpy as np
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import torch
from torch.utils.data import Dataset, DataLoader


class LungTumorDataset(Dataset):
    def __init__(self, data_dir: str, masks_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            data_dir (str): Directory with all the input images.
            masks_dir (str): Directory with all the corresponding masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.masks_dir = masks_dir
        self.all_file_names = os.listdir(self.data_dir)
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.all_file_names)

    def _augment(self, data: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies data augmentation to the input data and mask.

        Args:
            data (np.ndarray): The input image data.
            mask (np.ndarray): The corresponding mask.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Augmented image and mask.
        """
        # Fix to lack of randomness problem when using something other than PyTorch
        random_seed = torch.randint(0, 1000000, (1,)).item()
        ia.seed(random_seed)

        mask = SegmentationMapsOnImage(mask, shape=mask.shape)
        aug_data, aug_mask = self.transform(image=data, segmentation_maps=mask)
        return aug_data, aug_mask.get_arr()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the image and mask at the specified index, optionally applying transformations.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image and mask as PyTorch tensors.
        """
        data_path = os.path.join(self.data_dir, self.all_file_names[idx])
        mask_path = os.path.join(self.masks_dir, self.all_file_names[idx])

        data = np.load(data_path)
        mask = np.load(mask_path)

        if self.transform:
            data, mask = self._augment(data, mask)

        # Convert to PyTorch tensors and adjust dimensions
        data = torch.from_numpy(np.moveaxis(data, -1, 0)).float()
        mask = torch.from_numpy(np.expand_dims(mask, axis=0)).float()

        return data, mask


def get_dataset(preprocessed_input_dir: str, aug_pipeline: Optional[Callable] = None, data_type: str = 'train') -> LungTumorDataset:
    """
    Helper function to create a LungTumorDataset instance.

    Args:
        preprocessed_input_dir (str): Root directory containing the preprocessed data.
        aug_pipeline (callable, optional): Optional augmentation pipeline to apply.
        data_type (str): Type of dataset to load ('train', 'val', 'test', etc.).

    Returns:
        LungTumorDataset: Dataset instance.
    """
    data_dir = os.path.join(preprocessed_input_dir, data_type, 'data')
    label_dir = os.path.join(preprocessed_input_dir, data_type, 'mask')

    if not os.path.exists(data_dir) or not os.path.exists(label_dir):
        raise FileNotFoundError(f"Data or mask directory not found for data type: {data_type}")

    return LungTumorDataset(data_dir, label_dir, transform=aug_pipeline)
