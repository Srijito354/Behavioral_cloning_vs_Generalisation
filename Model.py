import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from custom_dataset import Data, DataLoader

class Modela(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        #self.model = models.resnet18()

        self.classifying_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            #nn.Tanh() # To keep output between -1 and 1.
        )

        for name, param in self.model.named_parameters(): # Claude helped me with this.
            if "fc" in name or "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.model.fc = self.classifying_regressor
    
    def forward(self, x):
        out = self.model(x)
        return out
'''
model = Modela()

transform = transforms.Compose((
    transforms.Resize((224, 224)),
    transforms.ToTensor()
))

data = Data("driving_log.csv", transform)
dataloader = DataLoader(data, batch_size = 32, shuffle = True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

num_epochs = 10

for batch in dataloader:
    print(batch["Center_cam"].shape)

    out = model(batch["Center_cam"])

    steering_loss = criterion(out[:, 0], batch["Steering"])
    throttle_loss = criterion(out[:, 1], batch["Throttle"])
    brake_loss = criterion(out[:, 2], batch["Brake"])

    loss = steering_loss + throttle_loss + brake_loss
    loss.backward()

    optimizer.zero_grad()
    optimizer.step()

    print(out[1].shape)

    print(steering_loss.item(), " ", throttle_loss.item(), " ", brake_loss.item())

    break
'''