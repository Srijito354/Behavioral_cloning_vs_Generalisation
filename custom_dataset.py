import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class Data(Dataset):
    def __init__(self, data_dir, transform):
        self.df = pd.read_csv(data_dir)
        self.df = self.df.dropna(subset=['Center'])
        straight = self.df[self.df['Steering'].abs() <= 0.03]
        turns = self.df[self.df['Steering'].abs() > 0.03]
        self.df = self.df[['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake']]
        straight = straight.sample(frac = 0.4, random_state = 42)
        self.df = pd.concat([straight, turns])
        self.df = self.df.reset_index(drop=True)
        '''
        plt.hist(self.df["Steering"], bins=50)
        plt.show()
        '''
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #img_path = f"/media/dev-mk2/Expansion/First_trial_make_it_happen/{df["Center"][idx]}"
        center_img_path = str(self.df["Center"][idx]).removeprefix("/media/dev-mk2/Expansion/driving_data/")
        right_img_path = str(self.df["Right"][idx]).removeprefix("/media/dev-mk2/Expansion/driving_data/")
        left_img_path = str(self.df["Left"][idx]).removeprefix("/media/dev-mk2/Expansion/driving_data/")
        
        centre_img = self.transform(Image.open(center_img_path).convert("RGB"))
        right_img = self.transform(Image.open(right_img_path).convert("RGB"))
        left_img = self.transform(Image.open(left_img_path).convert("RGB"))

        steering = torch.tensor((self.df["Steering"].values)[idx], dtype = torch.float32)
        throttle = torch.tensor((self.df["Throttle"].values)[idx], dtype = torch.float32)
        brake = torch.tensor((self.df["Brake"].values)[idx], dtype = torch.float32)

        return {"Center_cam":centre_img, "Right_cam":right_img, "Left_cam":left_img, "Steering":steering, "Throttle":throttle, "Brake":brake}

'''
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = Data("driving_log.csv", transform)

dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

for batch in dataloader:
    print(batch["Center_cam"].shape)
    print(batch["Steering"].shape)
    print(batch["Throttle"].shape)
    print(batch["Brake"].shape)
    break
'''

'''
transform = transforms.Compose((
    transforms.Lambda(lambda img: img.crop((0, 60, img.width, img.height - 25))),
    transforms.Resize((224, 224)),
    #transforms.CenterCrop((60, 60)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
))

dataset = Data("driving_log.csv", transform)

steering = dataset.df["Steering"].values

hist, bins = np.histogram(steering, bins=25)

bin_ids = np.digitize(steering, bins[:-1])

hist = np.where(hist == 0, 1, hist)

weights = 1.0 / hist[bin_ids - 1]

weights = torch.DoubleTensor(weights)

sampler = WeightedRandomSampler(
    weights,
    num_samples=len(weights),
    replacement=True
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler
)

for batch in dataloader:
    print(batch["Center_cam"].shape)
    print(batch["Steering"].shape)
    print(batch["Throttle"].shape)
    print(batch["Brake"].shape)
    break
'''