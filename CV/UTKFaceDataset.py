import torch
from torch import Tensor
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# gender mapping
gender_mapping = {'Male': 0, 'Female': 1}

# race mapping
race_mapping = {
    'White': 0,
    'Black': 1,
    'Asian': 2,
    'Indian': 3,
    'Others_hispanic_latino_middle_eastern': 4
}
emotion_mapping = {
    0:"angry",
    1:"happy",
    2:"sad",
    3:"neutral",
}
# initial size: 200 * 200
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def age_group(age):
    if age < 17:
        return 0
    elif age < 31:
        return 1
    elif age < 56:
        return 2
    else:
        return 3


def age_group_transform(age_group):
    if age_group == 0 or 1:
        return "17-35"
    elif age_group == 2:
        return "35-50"
    else:
        return "50+"


class UTKFaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, gender_mapping=None, race_mapping=None):
        self.annotations = pd.read_csv(csv_file, sep=',')
        self.img_dir = img_dir
        self.transform = transform
        self.gender_mapping = gender_mapping
        self.race_mapping = race_mapping

        # missing value
        self.annotations.dropna(inplace=True)
        # self.annotations.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx]['image_name']+".chip.jpg")
        image = Image.open(img_name).convert('RGB')

        age = self.annotations.iloc[idx]['age']
        gender = self.annotations.iloc[idx]['gender']
        race = self.annotations.iloc[idx]['race']

        if self.gender_mapping is not None:
            gender = self.gender_mapping[gender]
        else:
            raise "Pass in gender mapping"

        if self.race_mapping is not None:
            race = self.race_mapping[race]
        else:
            raise "Pass in race mapping"

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
    csv_file = 'utk_dataset_metadata.csv'
    img_dir = 'UTKFace'

    # create dataset entity
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
