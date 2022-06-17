"""
TODO
- team member 1: <name>
- team member 2: <name>

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
        sample = (img, label)
        return sample

    def load_image(self, image_index):
        image_name = self.image_ids[image_index]
        path = os.path.join(self.root_dir, image_name)
        img = imread(path)
        return img


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 192 * 108, 1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        output = torch.sigmoid(x)
        return output


def train(args, model, device, train_loader, optimizer, criterion, epoch):
    """
    Train a network
    You can find example code here: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, torch.unsqueeze(target, 1))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
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
            test_loss += criterion(output, torch.unsqueeze(target, 1)).item()  # sum up batch loss
            pred = torch.round(output)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
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
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {'batch_size': args.batch_size}
    val_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)

    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        # This normalization is used on the test server
        transforms.Normalize([0.2404, 0.2967, 0.3563], [0.0547, 0.0527, 0.0477])
        ])

    # Create train and test set
    path_to_dataset = "/PATH/TO/Boat-MNIST"   # TODO Set correct path
    train_set = Boats(root_dir=f'{path_to_dataset}/train', transform=transform,
                      gt_json_path=f'{path_to_dataset}/boat_mnist_labels_trainval.json')
    val_set = Boats(root_dir=f'{path_to_dataset}/val', transform=transform,
                    gt_json_path=f'{path_to_dataset}/boat_mnist_labels_trainval.json')

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(val_set, **val_kwargs)

    # Create network, optimizer and loss
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Train and validate
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        acc = test(model, device, test_loader, criterion)
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Best accuracy (val): {best_acc}")

    if args.save_model:
        torch.save(model.state_dict(), "model.pth")

    # --- Do not touch -----
    # Save model as onnx file
    dummy_input = torch.randn(1, 3, 108, 192, device=device)
    input_names = ["img_1"]
    output_names = ["output1"]
    torch.onnx.export(model, dummy_input, "ship_example.onnx", input_names=input_names, output_names=output_names)
    # ----------------------


if __name__ == '__main__':
    main()
