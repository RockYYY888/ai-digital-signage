# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from torch.utils.data import DataLoader
from util import get_resource_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""torch.device: The device (CUDA GPU or CPU) used for tensor operations."""
print(f'Using device: {device}')

# Gender mapping
gender_mapping = {'Male': 0, 'Female': 1}
"""dict: Mapping of gender strings to integer labels."""

# Race mapping
race_mapping = {
    'White': 0,
    'Black': 1,
    'Asian': 2,
    'Indian': 3,
    'Others_hispanic_latino_middle_eastern': 4
}
"""dict: Mapping of race/ethnicity strings to integer labels."""

emotion_mapping = {
    0: "angry",
    1: "happy",
    2: "sad",
    3: "neutral",
}
"""dict: Mapping of emotion integer predictions to string labels."""

# Initial size: 200 * 200
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
"""transforms.Compose: Preprocessing pipeline for resizing, tensor conversion, and normalization of images."""


def age_group(age):
    """Categorize an age into a discrete group.

    Args:
        age (int): The age value to categorize.

    Returns:
        int: An integer representing the age group (0: <17, 1: 17-30, 2: 31-55, 3: 56+).
    """
    if age < 17:
        return 0
    elif age < 31:
        return 1
    elif age < 56:
        return 2
    else:
        return 3


def age_group_transform(age_group):
    """Convert an age group integer to a human-readable string.

    Args:
        age_group (int): The age group integer (0-3).

    Returns:
        str: A string representing the age range (e.g., '17-35', '35-50', '50+').
    """
    if age_group == 0 or 1:
        return "17-35"
    elif age_group == 2:
        return "35-50"
    else:
        return "50+"


class UTKFaceDataset(Dataset):
    """A custom Dataset class for the UTKFace dataset.

    This class loads images and annotations from the UTKFace dataset, applies transformations,
    and provides demographic labels (age, gender, race).

    Args:
        csv_file (str): Path to the CSV file containing metadata.
        img_dir (str): Directory containing the image files.
        transform (callable, optional): Optional transform to be applied to the images.
        gender_mapping (dict, optional): Mapping of gender strings to integers.
        race_mapping (dict, optional): Mapping of race strings to integers.

    Attributes:
        annotations (pandas.DataFrame): DataFrame containing the dataset metadata.
        img_dir (str): Directory where images are stored.
        transform (callable): Image transformation pipeline.
        gender_mapping (dict): Gender label mapping.
        race_mapping (dict): Race label mapping.
    """

    def __init__(self, csv_file, img_dir, transform=None, gender_mapping=None, race_mapping=None):
        """Initialize the UTKFaceDataset with metadata and mappings."""
        self.annotations = pd.read_csv(csv_file, sep=',')
        self.img_dir = img_dir
        self.transform = transform
        self.gender_mapping = gender_mapping
        self.race_mapping = race_mapping

        # Handle missing values
        self.annotations.dropna(inplace=True)

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """Retrieve an image and its labels by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): The transformed image tensor.
                - labels (dict): Dictionary with 'age', 'gender', and 'race' as tensor labels.

        Raises:
            ValueError: If gender_mapping or race_mapping is not provided.
        """
        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx]['image_name'] + ".chip.jpg")
        image = Image.open(img_name).convert('RGB')

        age = self.annotations.iloc[idx]['age']
        gender = self.annotations.iloc[idx]['gender']
        race = self.annotations.iloc[idx]['race']

        if self.gender_mapping is not None:
            gender = self.gender_mapping[gender]
        else:
            raise ValueError("Pass in gender mapping")

        if self.race_mapping is not None:
            race = self.race_mapping[race]
        else:
            raise ValueError("Pass in race mapping")

        age = torch.tensor(age_group(age), dtype=torch.long)
        gender = torch.tensor(gender, dtype=torch.long)
        race = torch.tensor(race, dtype=torch.long)

        labels = {
            'age': age,
            'gender': gender,
            'race': race
        }

        if self.transform:
            image = self.transform(image)

        return image, labels


if __name__ == '__main__':
    """Main entry point for testing the UTKFaceDataset and DataLoader."""
    csv_file = get_resource_path('CV/utk_dataset_metadata.csv')
    img_dir = get_resource_path('UTKFace')

    # Create dataset instance
    dataset = UTKFaceDataset(
        csv_file=csv_file,
        img_dir=img_dir,
        transform=transform,
        gender_mapping=gender_mapping,
        race_mapping=race_mapping
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for images, labels in dataloader:
        print(f'Images batch shape: {images.size()}')
        print(f'Age labels batch shape: {labels["age"].size()}')
        print(f'Gender labels batch shape: {labels["gender"].size()}')
        print(f'Race labels batch shape: {labels["race"].size()}')
        break