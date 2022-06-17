"""
TODO
- team member 1: Raphael Anstett
- team member 2: Timo Lübbing

tasks:
    - add your team members' names at the top of the file
    - Take part in the challenge :)
"""

import argparse
import copy
import json
import os
from matplotlib.image import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset


class Boats(Dataset):

    def __init__(self, root_dir, transform=None, gt_json_path=''):
        self.root_dir = root_dir
        self.transform = transform
        self.gt_json_path = gt_json_path
        self.labels = json.load(open(gt_json_path, 'r'))
        self.image_list = sorted(os.listdir(root_dir))
        self.image_ids = dict(enumerate(self.image_list, start=0))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        img_name = self.image_ids[idx]
        label = self.labels[img_name]
        if self.transform:
            img = self.transform(img)
        return img, label

    def load_image(self, image_index):
        image_name = self.image_ids[image_index]
        path = os.path.join(self.root_dir, image_name)
        return imread(path)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 192 * 108, 1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return torch.sigmoid(x)

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(3 * 192 * 108, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        return torch.sigmoid(x)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        # Output size after conv filter:
        # ((w-f+2p) / s) + 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape = (batch, 12, 108, 192)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape = (batch, 12, 108, 192)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # Shape = (batch, 12, 54, 96)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape = (batch, 20, 54, 96)
        self.bn2 = nn.BatchNorm2d(num_features=20)
        # Shape = (batch, 20, 54, 96)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Shape = (batch, 20, 27, 48)

        self.fc1 = nn.Linear(20 * 27 * 48, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # print(f"Input: {x.shape}")
        x = F.relu(self.bn1(self.conv1(x)))
        # print(f"Conv1: {x.shape}")
        x = self.pool1(x)
        # print(f"Pool1: {x.shape}")
        x = F.relu(self.bn2(self.conv2(x)))
        # print(f"Conv2: {x.shape}")
        x = self.pool2(x)
        # print(f"Pool2: {x.shape}")
        x = torch.flatten(x, start_dim=1)
        # print(f"Flatten: {x.shape}")
        x = F.relu(self.fc1(x))
        # print(f"fc1: {x.shape}")
        x = F.relu(self.fc2(x))
        return torch.sigmoid(x)

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        # Output size after conv filter:
        # ((w-f+2p) / s) + 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape = (batch, 12, 108, 192)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=20)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Shape = (batch, 20, 54, 96)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Shape = (batch, 32, 27, 48)

        self.fc1 = nn.Linear(32 * 27 * 48, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # print(f"Input: {x.shape}")
        x = F.relu(self.bn1(self.conv1(x)))
        # print(f"Conv1: {x.shape}")
        x = F.relu(self.bn2(self.conv2(x)))
        # print(f"Conv2: {x.shape}")
        x = self.pool1(x)
        # print(f"Pool1: {x.shape}")
        x = F.relu(self.bn3(self.conv3(x)))
        # print(f"Conv3: {x.shape}")
        x = self.pool2(x)
        # print(f"Pool2: {x.shape}")
        x = torch.flatten(x, start_dim=1)
        # print(f"Flatten: {x.shape}")
        x = F.relu(self.fc1(x))
        # print(f"fc1: {x.shape}")
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(x)

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        # Output size after conv filter:
        # ((w-f+2p) / s) + 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        # Shape = (batch, 16, 108, 192)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Shape = (batch, 32, 54, 96)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Shape = (batch, 64, 27, 48)

        self.fc1 = nn.Linear(64 * 27 * 48, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        # print(f"Input: {x.shape}")
        x = F.relu(self.bn1(self.conv1(x)))
        # print(f"Conv1: {x.shape}")
        x = F.relu(self.bn2(self.conv2(x)))
        # print(f"Conv2: {x.shape}")
        x = self.pool1(x)
        # print(f"Pool1: {x.shape}")
        x = F.relu(self.bn3(self.conv3(x)))
        # print(f"Conv3: {x.shape}")
        x = self.pool2(x)
        # print(f"Pool2: {x.shape}")
        x = torch.flatten(x, start_dim=1)
        # print(f"Flatten: {x.shape}")
        x = F.relu(self.fc1(x))
        # print(f"fc1: {x.shape}")
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(x)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(args, model, device, train_loader, optimizer, scheduler, criterion, epoch):
    """
    Train a network
    You can find example code here: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    """
    model.train()
    processed = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, torch.unsqueeze(target, 1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        # batch accuracy during training
        pred = torch.round(output)
        processed += len(data)
        correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / processed

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {:.6f}\tAccuracy: {:.6f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                get_lr(optimizer), accuracy))
            if args.dry_run:
                break


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            output = model(data)
            loss = criterion(output, torch.unsqueeze(target, 1)).item()
            test_loss += float(loss)  # sum up batch loss
            pred = torch.round(output)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Ship Detection')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=35, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0008, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.95, metavar='M',
                        help='momentum (default: 0.95)'),
    parser.add_argument('--weightdecay', type=float, default=0.01, metavar='LR',
                        help='weight decay (default: 0.1)'),
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {'batch_size': args.batch_size}
    val_kwargs = {'batch_size': args.test_batch_size}
    print(f"Cuda? {use_cuda}")
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)

    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(6),
        # This normalization is used on the test server
        transforms.Normalize([0.2404, 0.2967, 0.3563], [0.0547, 0.0527, 0.0477])
        ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.2404, 0.2967, 0.3563], [0.0547, 0.0527, 0.0477])
    ])

    # Create train and test set
    path_to_dataset = "Boat_MNIST"   # TODO Set correct path
    train_set = Boats(root_dir=f'{path_to_dataset}/train', transform=transform,
                      gt_json_path=f'{path_to_dataset}/boat_mnist_labels_trainval.json')
    val_set = Boats(root_dir=f'{path_to_dataset}/val', transform=transform_val,
                    gt_json_path=f'{path_to_dataset}/boat_mnist_labels_trainval.json')

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, **val_kwargs)

    # Create network, optimizer and loss
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                              momentum=args.momentum, 
                                              weight_decay=args.weightdecay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                  max_lr=0.1, 
                                                  base_lr=args.lr,
                                                  mode="exp_range",
                                                  step_size_up=1500,
                                                  gamma=0.99)
    criterion = nn.MSELoss()

    print('Training of the model: \n', repr(model))

    # Train and validate
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, scheduler, criterion, epoch)
        acc = test(model, device, val_loader, criterion)
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Best accuracy (val): {best_acc}")

    if args.save_model:
        torch.save(model.state_dict(), "boat_classifier.pth")

    # --- Do not touch -----
    # Save model as onnx file
    dummy_input = torch.randn(1, 3, 108, 192, device=device)
    input_names = ["img_1"]
    output_names = ["output1"]
    torch.onnx.export(model, dummy_input, "ship_example.onnx", input_names=input_names, output_names=output_names)
    # ----------------------


if __name__ == '__main__':
    main()
