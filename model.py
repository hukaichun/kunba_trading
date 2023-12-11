import torch


class SimpleCNN(torch.nn.Module):
    def __init__(self, class_num:int):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 3, 2)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(3, 6, 2)

        self.fc1 = torch.nn.LazyLinear(2*class_num)
        self.fc2 = torch.nn.LazyLinear(2*class_num)
        self.fc3 = torch.nn.LazyLinear(2*class_num)
        self.fc4 = torch.nn.Linear(2*class_num, class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.pool(x)
        x = self.conv2(x)
        x = x.relu()
        x = torch.flatten(x, 1)
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x).relu()
        x = self.fc4(x)
        return x