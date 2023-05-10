import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from autoencoder import AutoEncoder

batch_size = 64
epochs = 10
device = 'mps'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def main():
    model = AutoEncoder(28*28).to(device)
    model.load_state_dict(torch.load('model.pth'))

    trainset = datasets.MNIST('data', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for images, _ in trainloader:
            optimizer.zero_grad()

            # flatten by batch
            images = images.view(images.shape[0], -1).to(device, non_blocking=True)
            # apply noise
            images += torch.randn_like(images) * 0.2

            output = model(images)
            loss = criterion(output, images)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} Training loss: {loss.item()}")
    
    torch.save(model.state_dict(), 'model.pth')
    
    test_x = torch.randn(1, 28*28).to(device) * 0.2
    plt.imshow(model(test_x).view(28, 28).cpu().detach().numpy())
    plt.show()


if __name__ == '__main__':
    main()
