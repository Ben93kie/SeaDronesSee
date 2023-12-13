"""
TODO
-  Group 107
- [Names removed, see moodle version]

tasks:
    - add your team members' names at the top of the file
    - Take part in the challenge :)
"""

import argparse
import copy
import json
import os
from matplotlib.image import imread
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset

class Analyzer():
    def __init__(self):
        self.training_loss = []
        self.validation_loss = []
        self.percentage = []

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
        # remove alpha channel if needed
        if img.shape[2] == 4:
            img = img[:,:,:3]
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
        #self.fc1 = nn.Linear(3 * 192 * 108, 1)
        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 14, 14, stride = 2),
        nn.ReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(14, 14, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(616, 616),
        nn.ReLU(),
        nn.Linear(616, 616),
        nn.ReLU(),
        nn.Linear(616, 616),
        
        nn.ReLU(),
        nn.Linear(616, 32),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
        )
       


        """
        99.40%

        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 12, 14, stride = 2),
        nn.ReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(12, 12, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(528, 528),
        nn.ReLU(),
        nn.Linear(528, 528),
        nn.ReLU(),
        nn.Linear(528, 528),
        nn.ReLU(),
        nn.Linear(528, 32),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
        )


        epochs = 150

        -------------------------------------

        99.26958831341301%

        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 8, 14, stride = 2),
        nn.ReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(8, 8, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(352, 352),
        nn.ReLU(),
        nn.Linear(352, 352),
        nn.ReLU(),
        nn.Linear(352, 352),
        nn.ReLU(),
        nn.Linear(352, 32),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
        )

        epochs = 200



        ---------------------------------------------

        99.003% (1491/1506)

        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 4, 14, stride = 2),
        nn.ReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(4, 4, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(176, 176),
        nn.ReLU(),
        nn.Linear(176, 176),
        nn.ReLU(),
        nn.Linear(176, 176),
        nn.ReLU(),
        nn.Linear(176, 32),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
        )

        epochs = 100












        -------------------------------------------------

         VV This is the network documented in the pdf VV

        -------------------------------------------------
        98.937%


        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 4, 14, stride = 2),
        nn.ReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(4, 4, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(176, 176),
        nn.Dropout(p=0.05),
        nn.ReLU(),
        nn.Linear(176, 32),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
        )



        optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=0.005, momentum=0.0)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = args.epochs // 6, eta_min = 0.0001, verbose=True)
        criterion = nn.BCELoss()

        ----------------------------------------------------
        98,605%

        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 3, 14, stride = 2),
        nn.ReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(3, 3, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(132, 132),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(132, 64),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
        )

        optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=0.005, momentum=0.0)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.4),int(args.epochs*0.83)], gamma=0.1, verbose=True)
        criterion = nn.BCELoss()

        original set only(?)
        epochs = 20(?)


        -----------------------------------------



        98,804%

        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 3, 14, stride = 2),
        nn.ReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(3, 3, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(132, 132),
        nn.Dropout(p=0.05),
        nn.ReLU(),
        nn.Linear(132, 132),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(132, 1),
        nn.Sigmoid()
        )

         optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=0.005, momentum=0.0)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,48,70], gamma=0.3, verbose=True)
        criterion = nn.BCELoss()
        epochs = 80

        -----------------------------------------------------------------
        98,738%

        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 3, 14, stride = 2),
        nn.ReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(3, 3, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(132, 132),
        nn.Dropout(p=0.05),
        nn.ReLU(),
        nn.Linear(132, 132),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(132, 1),
        nn.Sigmoid()
        )

         optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=0.005, momentum=0.0)    
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12,25], gamma=0.1, verbose=True)
        criterion = nn.BCELoss()
        epochs=30


        98.671%

        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 3, 14, stride = 2),
        nn.ReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(3, 3, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(132, 132),
        nn.Dropout(p=0.05),
        nn.ReLU(),
        nn.Linear(132, 132),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(132, 1),
        nn.Sigmoid()
        )

        optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=0.005, momentum=0.0)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1, verbose=True)
        criterion = nn.BCELoss()

        epochs=30




        -----------------------------------------------------
        98.605%

        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 3, 14, stride = 2),
        nn.ReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(3, 3, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(132, 132),
        #nn.Dropout(p=0.05),
        nn.ReLU(),
        nn.Linear(132, 132),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(132, 1),
        nn.Sigmoid()
        )
        epochs=50, all datasets

        optimizer = optim.Adagrad(model.parameters(), lr=0.06, weight_decay=0.005)
        criterion = nn.BCELoss()


        ----------------------------------------------------
                        double dataset added
        -----------------------------------------------------
        98,53%

        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 3, 14, stride = 2),
        nn.ReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(3, 3, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(132, 132),
        nn.Dropout(p=0.05),
        nn.ReLU(),
        nn.Linear(132, 132),
        nn.Dropout(p=0.05),
        nn.ReLU(),
        nn.Linear(132, 1),
        nn.Sigmoid()
        )

        epochs:15
        optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=0.005, momentum=0.0)
        criterion = nn.BCELoss()
        orig, hor, vert


        ---------------------------------------------
                Custom Datasets Added: Hor, Ver
        --------------------------------------------
        98.14%
         self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 3, 14),
        nn.ReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(3, 3, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(825, 825),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(825, 825),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(825, 1),
        nn.Sigmoid()
        )

        optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=0.005, momentum=0.0)
        criterion = nn.BCELoss()


        --------------------------------------------
        97.54%
        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 3, 14),
        nn.ReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(3, 3, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(825, 825),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(825, 825),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(825, 1),
        nn.Sigmoid()
        )

        optimizer = optim.SGD(model.parameters(), lr=0.02, weight_decay=0.01, momentum=0.00)
        criterion = nn.BCELoss()

        ---------------------------------------------
        97.34%

         self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 3, 14),
        nn.ReLU(),
        nn.MaxPool2d(3,3),
        nn.Conv2d(3, 3, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(825, 825),
        nn.Dropout(p=0.2)
        nn.ReLU(),
        nn.Linear(825, 825),
        nn.Dropout(p=0.2)
        nn.ReLU(),
        nn.Linear(825, 1),
        nn.Sigmoid()
        )

        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01, momentum=0.0)
        criterion = nn.BCELoss()
        ------------------------------------------------
        97.07%

        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 3, 14),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(3, 3, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(2280, 2280),
        nn.ReLU(),
        nn.Linear(2280, 1028),
        nn.ReLU(),
        nn.Linear(1028, 1),
        nn.Sigmoid()
        )

        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01, momentum=0.0)
        criterion = nn.BCELoss()
        --------------------------------------------------------------------------------------

        97.34%
        super(Net, self).__init__()
        #self.fc1 = nn.Linear(3 * 192 * 108, 1)
        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 3, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(3, 3, 4),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(3036, 1028),
        nn.ReLU(),
        nn.Linear(1028, 1028),
        nn.ReLU(),
        nn.Linear(1028, 1),
        nn.Sigmoid()
        )

        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01, momentum=0.0)
        criterion = nn.BCELoss()
----------------------------------------------------
        96.74%

         self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 3, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(3, 3, 4),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(3036, 1028),
        nn.ReLU(),
        nn.Linear(1028, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
        )

        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01, momentum=0.0)
        criterion = nn.BCELoss()
    -------------------------------------------------------------

        95,949%
        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 3, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(3, 3, 4),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(3036, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
        )


         optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01, momentum=0.0)
         criterion = nn.BCELoss()
        ---------------------------------------------------
        95.21%

        self.network_stack = nn.Sequential(
        # convlutional processing
        nn.Conv2d(3, 3, 9),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),

        # FFNN
        nn.Linear(3 * 50 * 92, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
        )

        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01, momentum=0.0)
        criterion = nn.BCELoss()


        -----------------------------------------------------

    	85,32%
        self.network_stack = nn.Sequential(
        nn.Linear(3 * 192 * 108, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.ELU(),
        nn.Linear(64, 32),
        nn.ELU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
        )

        With: batch_size 14, lr 0.1,
        optimizer = optim.SGD(model.parameters(), lr=args.lr,  weight_decay=0.02, momentum=0.01)
        criterion = nn.MSELoss()

        --------------

        84.52%
        self.network_stack = nn.Sequential(
        nn.Linear(3 * 192 * 108, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
        nn.Sigmoid()
        )


        with: batch_size 16, lr 0.08
        optimizer = optim.SGD(model.parameters(), lr=args.lr,  weight_decay=0.02, momentum=0.01)
        criterion = nn.MSELoss()

        --------------------------

    	84.98%
         self.network_stack = nn.Sequential(
        nn.Linear(3 * 192 * 108, 1028),
        nn.Tanh(),
        nn.Linear(1028, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 1),
        nn.Sigmoid()
        )


        with:

        optimizer = optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        """

    def forward(self, image):
        #image = torch.flatten(image, start_dim=1)
        #output = self.fc1(image)
        #image = transforms.functional.adjust_saturation(image, 0.8)
        output = self.network_stack(image)
        return output


def train(args, model, device, train_loader, optimizer, criterion, epoch, scheduler, analyzer):
    """
    Train a network
    You can find example code here: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    """
    model.train()
    average_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, torch.unsqueeze(target, 1))
        loss.backward()
        average_loss += float(loss)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break       
    analyzer.training_loss.append(average_loss/len(train_loader))
    print(f"Average training loss: {average_loss/len(train_loader)}")


def test(model, device, test_loader, criterion, analyzer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            output = model(data)
            loss = criterion(output, torch.unsqueeze(target, 1)).item()  # sum up batch loss
            test_loss += loss
            pred = torch.round(output)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    analyzer.validation_loss.append(float(test_loss))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    analyzer.percentage.append(correct / len(test_loader.dataset))
    return 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Ship Detection')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
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
    print(f"using {device}")
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
    path_to_dataset = "./Boat-MNIST"   # TODO Set correct path
    train_set = Boats(root_dir=f'{path_to_dataset}/train', transform=transform,
                      gt_json_path=f'{path_to_dataset}/boat_mnist_labels_trainval.json')
    train_set_hor_mirrored = Boats(root_dir=f'{path_to_dataset}/custom_dataset/hor_mirrored', transform=transform,
                      gt_json_path=f'{path_to_dataset}/boat_mnist_labels_trainval.json')
    train_set_vert_mirrored = Boats(root_dir=f'{path_to_dataset}/custom_dataset/vert_mirrored', transform=transform,
                      gt_json_path=f'{path_to_dataset}/boat_mnist_labels_trainval.json') 
    train_set_double_mirrored = Boats(root_dir=f'{path_to_dataset}/custom_dataset/double_mirrored', transform=transform,
    gt_json_path=f'{path_to_dataset}/boat_mnist_labels_trainval.json')           
    val_set = Boats(root_dir=f'{path_to_dataset}/val', transform=transform,
                    gt_json_path=f'{path_to_dataset}/boat_mnist_labels_trainval.json')

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    train_loader_hor_mirrored = torch.utils.data.DataLoader(train_set_hor_mirrored, **train_kwargs)
    train_loader_vert_mirrored = torch.utils.data.DataLoader(train_set_vert_mirrored, **train_kwargs)
    train_loader_double_mirrored = torch.utils.data.DataLoader(train_set_double_mirrored, **train_kwargs)
    train_loader_val_hor_mirrored = torch.utils.data.DataLoader(train_set_hor_mirrored, **train_kwargs)
    train_loader_val_vert_mirrored = torch.utils.data.DataLoader(train_set_vert_mirrored, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(val_set, **val_kwargs)

    # Create network, optimizer and loss
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=0.005, momentum=0.0)
    #optimizer = optim.Adagrad(model.parameters(), lr=0.05, weight_decay=0.005)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.4),int(args.epochs*0.83)], gamma=0.1, verbose=True)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = 10,  eta_min = 0.0001, verbose=True)
    
    #criterion = nn.MSELoss()
    #Change loss function
    criterion = nn.BCELoss()
    #criterion = nn.CrossEntropyLoss()

    # Train and validate
    best_acc = 0
    best_epoch = 0
    analyzer = Analyzer()
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        if (epoch %6 == 0):
            print("Original set")
            train(args, model, device, train_loader, optimizer, criterion, epoch, scheduler, analyzer)
        elif (epoch %6== 1):
            print("Horizontally Mirrored set")
            train(args, model, device, train_loader_hor_mirrored, optimizer, criterion, epoch, scheduler, analyzer)   
        elif (epoch %6== 2):        
            print("Vertically Mirrored set")
            train(args, model, device, train_loader_vert_mirrored, optimizer, criterion, epoch, scheduler, analyzer)
        elif (epoch %6 == 3):
            print("Double Mirrored set")
            train(args, model, device, train_loader_double_mirrored, optimizer, criterion, epoch, scheduler, analyzer)
        elif (epoch %6 == 4):
            print("Horizontally Mirrored Verification set")
            train(args, model, device, train_loader_val_hor_mirrored, optimizer, criterion, epoch, scheduler, analyzer)
        else:
            print("Vertically Mirrored Verification set")
            train(args, model, device, train_loader_val_vert_mirrored, optimizer, criterion, epoch, scheduler, analyzer)


        acc = test(model, device, test_loader, criterion, analyzer)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step(epoch)

    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Best accuracy (val): {best_acc}   in epoch {best_epoch}")

    if args.save_model:
        torch.save(model.state_dict(), "model.pth")

    # --- Do not touch -----
    # Save model as onnx file
    dummy_input = torch.randn(1, 3, 108, 192, device=device)
    input_names = ["img_1"]
    output_names = ["output1"]
    torch.onnx.export(model, dummy_input, "ship_example.onnx", input_names=input_names, output_names=output_names)
    # ----------------------

    plt.plot(list(range(len(analyzer.training_loss))), analyzer.training_loss, label = "Training loss")
    plt.plot(list(range(len(analyzer.validation_loss))), analyzer.validation_loss, label = "Validation loss")
    plt.plot(list(range(len(analyzer.percentage))), analyzer.percentage, label = "percentage")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except(KeyboardInterrupt):
        print("-"*50)
        print("KeyboardInterrupt")
        print("-"*50)