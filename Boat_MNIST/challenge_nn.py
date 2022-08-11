"""
- team member 1: Finn Fassbender
- team member 2: Mathis Welker
"""

"""
model size: 2459 Parameters     0.01MB
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Net                                      [1, 1]                    33
├─Conv2d: 1-1                            [1, 16, 54, 96]           1,216
├─Conv2d: 1-2                            [1, 4, 55, 97]            1,028
├─Dropout: 1-3                           [1, 4, 27, 48]            --
├─Conv2d: 1-4                            [1, 1, 27, 48]            37
├─Dropout: 1-5                           [1, 1, 9, 16]             --
├─Linear: 1-6                            [1, 1]                    145
==========================================================================================
Total params: 2,459
Trainable params: 2,459
Non-trainable params: 0
Total mult-adds (M): 11.84
==========================================================================================
Input size (MB): 0.25
Forward/backward pass size (MB): 0.84
Params size (MB): 0.01
Estimated Total Size (MB): 1.10
==========================================================================================
"""

import argparse
import copy
import json
import os
import pathlib
from matplotlib.image import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from torchinfo import summary

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
        # First 2D convolutional layer, taking in 3 input channels (image),
        # outputting 16 convolutional features, with a square kernel size of 5
        self.conv1 = nn.Conv2d( 3, 16, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16,  4, 4, stride=1, padding=2)
        self.conv3 = nn.Conv2d( 4,  1, 3, stride=1, padding=1)

        # Designed to ensure that adjacent pixels are either all 0s or all 
        # active with an input probability
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layer that outputs the label
        self.fc1 = nn.Linear(144, 1)

    def forward(self, x):
        # Pass data through convolutional layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3)
        x = self.dropout1(x)
        # Prepare data for fully connected layers
        x = torch.flatten(x, 1)
        # Pass data through fully connected layers
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
            print('Train Epoch: {:2.0f} [{:4.0f}/{} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, criterion, name):
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

    print(name + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def size(model):
    # Print size of model
    param_size, param_num = 0, 0
    for param in model.parameters():
        param_num += param.nelement()
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.0f} Parameters\t{:.2f}MB'.format(param_num, size_all_mb))


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
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--test-model',  type=pathlib.Path, default='', 
                        metavar='PATH', help='For Testing the last saved Model')
    parser.add_argument('--export-model', type=pathlib.Path, default='', 
                        metavar='PATH', help='For exporting the specified Model to .onnx')
    parser.add_argument('--load-model', type=pathlib.Path, default='', 
                        metavar='PATH', help='For loading a previous Model state dict')
    parser.add_argument('--load-bestacc', type=float, default=0, metavar='N',
                        help='the previous best accuracy to use')
    parser.add_argument('--vis-model', action='store_true', default=False,
                        help='For Visualizing the current Model')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.set_num_threads(8)
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {'batch_size': args.batch_size}
    val_kwargs   = {'batch_size': args.test_batch_size}
    more_kwargs  = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(more_kwargs)
    val_kwargs.update(more_kwargs)

    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        # This normalization is used on the test server
        transforms.Normalize([0.2404, 0.2967, 0.3563], [0.0547, 0.0527, 0.0477])
        ])

    # Create train and test set
    path_to_dataset = "../boat-mnist-dataset"
    train_set = Boats(root_dir=f'{path_to_dataset}/train', transform=transform,
                      gt_json_path=f'{path_to_dataset}/boat_mnist_labels_trainval.json')
    val_set   = Boats(root_dir=f'{path_to_dataset}/val', transform=transform,
                      gt_json_path=f'{path_to_dataset}/boat_mnist_labels_trainval.json')

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    test_loader  = torch.utils.data.DataLoader(val_set, **val_kwargs)

    # Create network, optimizer and loss
    model = Net().to(device)
    size(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    # Load a previous model state (continues after)
    if args.load_model != pathlib.PurePath():
        model.load_state_dict(torch.load(args.load_model))

    # Test last saved model (stops after)
    if args.test_model != pathlib.PurePath():
        model.load_state_dict(torch.load(args.test_model))
        test(model, device, test_loader,  criterion, 'Test ')
        test(model, device, train_loader, criterion, 'Train')
        return

    # Export specified model (stops after)
    if args.export_model != pathlib.PurePath():
        model.load_state_dict(torch.load(args.export_model))
        test(model, device, test_loader,  criterion, 'Test ')
        test(model, device, train_loader, criterion, 'Train')
        # Save model as onnx file
        dummy_input = torch.randn(1, 3, 108, 192, device=device)
        input_names = ["img_1"]
        output_names = ["output1"]
        torch.onnx.export(model, dummy_input, "ship_example-" + str(args.export_model) + ".onnx", input_names=input_names, output_names=output_names)
        print("ship_example-" + str(args.export_model) + ".onnx")
        return

    # Visualize model (stops after)
    if args.vis_model:
        summary(model, input_size=(1, 3, 108, 192))
        return

    # Train and validate
    best_acc = args.load_bestacc if args.load_bestacc != 0 else 0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        acc = test(model, device, test_loader, criterion, 'Test ')
        #test(model, device, train_loader, criterion, 'Train')
        print('\n')
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f"Current best accuracy (val): {best_acc:.2f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Best accuracy (val): {best_acc}")

    # Save the model weights
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
