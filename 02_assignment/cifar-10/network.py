import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()

        self.Model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(in_features=8*8*64, out_features=512),
            # 8*8*64 explain:
            # the size of the original image is 32*32*3 in CIFAR10
            # the conv layers here all have kernel_size=3, stride=1, padding=1, so the size of the image remains the same
            # the maxpool layers here all have kernel_size=2, stride=2, so the size of the image is reduced by half
            # there are two maxpool layers, so the size of the image is reduced by 4, which is 32/4=8
            # and the last conv layer has 64 filters, so the size of the image is 8*8*64
            nn.BatchNorm1d(512),

            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_class),

            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.Model(x)
        return x


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from dataset import CIFAR10
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph.
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break
