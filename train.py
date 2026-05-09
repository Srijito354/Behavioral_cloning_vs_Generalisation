import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from Model import Modela
from torchvision import transforms
import torchvision.models as models
from custom_dataset import Data, DataLoader, WeightedRandomSampler

model = Modela()
model.load_state_dict(torch.load("trained_model_pre(non-panorama-only-center).pt")) # Best model so far (6th March, 20:51 IST)

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

#dataloader = DataLoader(data, batch_size = 32, shuffle = True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-7)
#optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

num_epochs = 200

for epoch in range(num_epochs):
    cumulative_per_epoch_loss = []
    for batch in dataloader:
        #print(batch["Center_cam"].shape)

        out = model(batch["Center_cam"])

        steering_loss = criterion(out[:, 0], batch["Steering"])
        throttle_loss = criterion(out[:, 1], batch["Throttle"])
        brake_loss = criterion(out[:, 2], batch["Brake"])

        optimizer.zero_grad()

        loss = 2.0*steering_loss + 1.0*throttle_loss + 0.0*brake_loss # assigning weights to loss, to make the optimizer focus more towards correcting the steering and not just the throttle.
        loss.backward()

        optimizer.step()

        #print(out[1].shape)

        #print(steering_loss.item(), " ", throttle_loss.item(), " ", brake_loss.item())
        cumulative_per_epoch_loss.append(loss)
        
    if (epoch+1)%10 == 0:
        torch.save(model.state_dict(), f"panorama_input_model_checkpoints/fine_tuned_resnet_{epoch+1}.pt")

    print(f"Epoch:{epoch+1}, Average loss:{sum(cumulative_per_epoch_loss)/len(cumulative_per_epoch_loss)}")

torch.save(model.state_dict(), "panorama_input_model_checkpoints/fine_tuned_resnet_final.pt")