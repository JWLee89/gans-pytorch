import argparse
import os

import torch.nn as nn
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.cifar import CIFAR10
from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_d', type=float, default=0.001, help='Learning rate of discriminator')
    parser.add_argument('--lr_g', type=float, default=0.001, help='Learning rate of generator')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train model')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='The batch size while training')
    parser.add_argument('--save_path', type=str, default='train_results', help='Place to store training results')
    return parser.parse_args()


def get_cifar_10(batch_size, save_path: str, device):
    dataset_training = CIFAR10(save_path, download=True, train=True, transform=transforms.Compose(
        [
            transforms.ToTensor()
        ]
    ))
    dataset_testing = CIFAR10(save_path, download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    training_dataloader = torch.utils.data.DataLoader(dataset=dataset_training, batch_size=batch_size,
                                                      collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_dataloader = torch.utils.data.DataLoader(dataset=dataset_testing, batch_size=batch_size,
                                                  collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    return training_dataloader, test_dataloader


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 64, (3, 3), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.Conv2d(64, 128, (3, 3), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, X):
        output = self.layers(X)
        return self.fc(output)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(

        )

    def forward(self, X):
        # Input label into forward pass

        pass


def train_model(disc_lr: float, gen_lr: float, batch_size: int, epochs: int, save_path: str):

    # Initialize model and devices
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    disc = Discriminator().to(device)
    generator = Generator().to(device)

    # Initialize folder to save results
    os.makedirs(save_path, exist_ok=True)

    # Get data loader
    train_dataloader, test_dataloader = get_cifar_10(batch_size, save_path)

    # Initialize optimizer

    # Train the model
    for epoch in range(epochs):
        for i, (x_train, y_train) in enumerate(train_dataloader):
            batch_size: int = x_train.shape[0]

            # Generate random gaussian noise
            random_noise = torch.randn((batch_size, 3, 32, 32), device=device)
            # Concatenate labels to discriminator


            fake_images = generator(random_noise)

            random_noise = torch.randn((1, 3, 32, 32))
            output = disc(random_noise)

            print(f'X_train: {x_train.shape}, y_train: {y_train.shape}')
            break



if __name__ == '__main__':
    args = get_args()
    train_model(args.lr_d, args.lr_g, args.batch_size, args.epochs, args.save_path)
