import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader

from autoencoder import AutoEncoder

batch_size = 128
epochs = 1
device = 'mps'
model_path = 'model.pth'
transform = transforms.Compose([
    transforms.ToTensor(),
])

def main():
    model = AutoEncoder(28 * 28).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.train()

    trainset = datasets.MNIST('data', download=True, train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    criterion = LogCoshLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)

    for epoch in range(epochs):
        for images, _ in trainloader:
            optimizer.zero_grad()

            images = images.view(images.shape[0], -1).to(device)
            images += torch.randn_like(images) * 0.2

            output = model(images)
            loss = criterion(output, images)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} Training loss: {loss.item()}")
    
    model.eval()
    torch.save(model.state_dict(), model_path)
    
    for i in range(10):
        test_x = torch.randn_like(images[0]).to(device) * 0.01
        generated_model = model(test_x).view(28, 28).cpu().detach().numpy()
        plt.imshow(generated_model)
        plt.imsave('generated.png', generated_model)

        plt.show()


class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(loss)


if __name__ == '__main__':
    main()
