"""

- team member 1: Aaron Riedling
- team member 2: Christoph Hoffmann
- Additional Learning Tutorial on Pytorch: "The Morpheus Tutorials" on Youtube

tasks:
    - add your team members' names at the top of the file
    - Take part in the challenge :)
"""

import argparse
import copy
import json
import os
import random
from matplotlib.image import imread
import shutil
from PIL import Image
import json
#from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset

# ========================================= classes ======================================================
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
    def __init__(self, dropout):
        super(Net, self).__init__()
        self.conv1a = nn.Conv2d(3, 12, kernel_size=5) # 3 input images: RGB
        self.conv2a = nn.Conv2d(12, 24, kernel_size=5) # 2 conv layers for feature extraction
        self.conv_DOa = nn.Dropout2d(dropout)
        self.conv1b = nn.Conv2d(3, 12, kernel_size=7) # 3 input images: RGB
        self.conv2b = nn.Conv2d(12, 24, kernel_size=7) # 2 conv layers for feature extraction
        self.conv_DOb = nn.Dropout2d(dropout)
        self.fc1 = nn.Linear(24*24*45 + 24*14*28, 512)
        self.fc2 = nn.Linear(512, 128)
        #self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        # a-arm of CNN --> smaller features
        xa = F.max_pool2d(self.conv1a(x),2)
        xa = F.leaky_relu(xa,0.05)
        xa = self.conv2a(xa)
        xa = self.conv_DOa(xa)
        xa = F.max_pool2d(xa , 2)
        xa = F.leaky_relu(xa , 0.05) # x.size -> BS*24*24*45 (BS: BatchSize)
        #print(xa.size())
        #exit()
        xa = xa.view(-1,24*24*45) # changes data format to 1 neuron per subimage

        # b-arm of CNN  --> larger features
        xb = F.max_pool2d(self.conv1b(x) , 3)
        xb = F.leaky_relu(xb,0.05)
        xb = self.conv2b(xb)
        xb = self.conv_DOb(xb)
        xb = F.max_pool2d(xb , 2)
        xb = F.leaky_relu(xb , 0.05) # x.size -> BS*24*10*19 (BS: BatchSize)
        #print(xb.size())
        #exit()
        xb = xb.view(-1,24*14*28) # changes data format to 1 neuron per subimage

        # combine
        x = torch.cat((xa, xb),1)
        x = F.leaky_relu(self.fc1(x), 0.05)
        x = F.leaky_relu(self.fc2(x), 0.05)
        #x = F.leaky_relu(self.fc3(x), 0.05)
        output = torch.sigmoid(self.fc4(x))
        return output

# =============================================== file methods ===========================================

def make_splitlist(input_path, kfolds):
    temp = os.listdir(input_path+"original_split/train")
    files_in_directory = ["train/" + file for file in temp]
    temp = os.listdir(input_path + "original_split/val")
    files_in_directory = files_in_directory + ["val/" + file for file in temp]

    # shuffle order of files since it might be ordered
    # random.shuffle(files_in_directory) shuffle activated in train loader

    file_split_dict= {}

    for file in files_in_directory:
        if file.endswith(".png"):
            file_split_dict[file] = random.randrange(0, kfolds , 1)

    return file_split_dict


def remove_split_files(path, extension):
    files_in_directory = os.listdir(path)
    filtered_files = [file for file in files_in_directory if file.endswith(extension)]
    for file in filtered_files:
        path_to_file = os.path.join(path, file)
        os.remove(path_to_file)


def split_files(path, split_list, k):
    """
    does a train/test split at the location
    creates two folders: val and train
    """

    train_path = path + "train/"
    val_path = path + "val/"
    label_path = path + "label/"

    print("clearing out old splits")
    if os.path.exists(train_path):
        remove_split_files(train_path, ".png")
        os.rmdir(train_path)

    if os.path.exists(val_path):
        remove_split_files(val_path, ".png")
        os.rmdir(val_path)

    if os.path.exists(label_path):
        remove_split_files(label_path, ".json")
        os.rmdir(label_path)

    print("creating new split")
    os.makedirs(train_path)
    os.makedirs(val_path)
    os.makedirs(label_path)

    for key in split_list.keys():
        if split_list[key] == k:
            shutil.copy(path +"original_split/"+ key, path + "val/" + key.split("/")[-1])
        else:
            shutil.copy(path +"original_split/"+ key, path + "train/" + key.split("/")[-1])

    shutil.copy(path+'original_split/boat_mnist_labels_trainval.json', label_path + 'boat_mnist_labels_trainval.json')

def inflate_train_data(train_path, label_file):
    '''
    Inflates train data by different transformations, might be worth testing multiple sets
    Currently Mirroring should be safe
    '''
    files_in_folder = os.listdir(train_path)

    # handle JSON
    label_list = {}
    with open(label_file) as fp:
        label_list = json.load(fp)

    # iterate over train images
    print("Creating inflated dataset...")
    count = 0
    for file in files_in_folder:
        count +=1
        if count % 500 == 0:
            print("Image " + str(count) + "...")

        Yval = label_list[file]
        orig_image = Image.open(train_path+file)

        # make and save changed items
        #hf_image = transforms.RandomHorizontalFlip(1.0)(orig_image)
        #hf_image.save(train_path + "HF_" + file)
        #label_list["HF_" + file] = Yval

        #fl_image = transforms.RandomHorizontalFlip(1.0)(orig_image)
        #fl_image = transforms.RandomVerticalFlip(1.0)(fl_image)
        #fl_image.save(train_path + "FL_" + file)
        #label_list["FL_" + file] = Yval

        gs_image = transforms.Grayscale(3)(orig_image) # three channels are needed to match the input dimensions
        gs_image.save(train_path + "GS_" + file)
        label_list["GS_" + file] = Yval

        #eq_image = transforms.RandomEqualize(1.0)(orig_image)
        #eq_image.save(train_path + "EQ_" + file)
        #label_list["EQ_" + file] = Yval

        #rp_image = transforms.RandomPerspective(distortion_scale=0.8, p=1.0, fill=(0,) * len(orig_image.getbands()))(orig_image)
        #rp_image.save(train_path+"RP_" + file)
        #T.functional.rotate(img=rgb, angle=degree_to_rotate, fill=(0,) * len(rgb.getbands()))
        #label_list["RP_" + file] = Yval

        #rr_image = transforms.RandomRotation(degrees=(0, 180))(orig_image)
        #rr_image.save(train_path+"RR_" + file)
        #label_list["RR_" + file] = Yval

        #ri_image = transforms.RandomInvert(1)(orig_image)
        #ri_image.save(train_path+"RI_" + file)
        #label_list["RI_" + file] = Yval


    # save modified JSON
    with open(label_file, 'w') as json_file:
        json.dump(label_list, json_file,
                  indent=None,
                  separators=(',', ': '))

def setup_standard_split(path):
    train_path = path + "train/"
    val_path = path + "val/"
    label_path = path + "label/"

    print("clearing out old splits")
    if os.path.exists(train_path):
        remove_split_files(train_path, ".png")
        os.rmdir(train_path)

    if os.path.exists(val_path):
        remove_split_files(val_path, ".png")
        os.rmdir(val_path)

    if os.path.exists(label_path):
        remove_split_files(label_path, ".json")
        os.rmdir(label_path)

    print("importing standard split")
    os.makedirs(label_path)
    shutil.copy(path + 'original_split/boat_mnist_labels_trainval.json', label_path + 'boat_mnist_labels_trainval.json')
    shutil.copytree(path + "original_split/train" , train_path)
    shutil.copytree(path + "original_split/val" , val_path)

# =============================================== training methods ===========================================

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    """
    Train a network
    You can find example code here: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    """
    train_loss = []
    model.train() # model is the Net loaded on the device, now set to train
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float()
        optimizer.zero_grad() # reset optimizer
        output = model(data)
        loss = criterion(output, torch.unsqueeze(target, 1))
        loss.backward() # does the backward propagation
        optimizer.step() # this is the learning step. Optimiser has been loaded with the net parameters

        if batch_idx % args.log_interval == 0:
            train_loss.append(loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    return sum(train_loss)/len(train_loss)

def test(model, device, test_loader, criterion):
    model.eval() # model set to eval, now it does not learn, parameters are frozen
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

    print('\nTest set: Average loss: {:.7f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

# =============================================== main function ===========================================

def main():
    kfolds = 4
    input_path = "/home/christoph/Documents/Semester_4/Intro_to_ANN/A05/Code_Boat_MNIST/input/"
    run_splits = kfolds

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Ship Detection')
    
    parser.add_argument('-i', type=str, metavar='i',
                        help='input path to all files')
    parser.add_argument('-k', type=int, default=4, metavar='k',
                        help='k-fold CV (default:4)')
    parser.add_argument('--nsplits', type=int, metavar='nsplits',
                        help='run number of splits, (default = k)')
    parser.add_argument('--reuse-split', action='store_true', default=False,
                        help='reuse the last existing train/test split (!Will double-inflate inflated data!)')
    parser.add_argument('--standard-split', action='store_true', default=False,
                        help='use the standard train/test split')
    #parser.add_argument('--final-train', action='store_true', default=False,
    #                    help='does the final training run with all data and saves the model')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0, metavar='M',
                        help='learning momentum (default: 0)')
    parser.add_argument('--lr-redux', type=float, default=0, metavar='LRR',
                        help='learning rate reduction per epoch, decrease reduces when getting closer to 0 lr')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='DO',
                        help='dropout layer (default: 0.2)')
    parser.add_argument('--full-lr-epochs', type=int, default=0, metavar='EPs',
                        help='epochs at full lr before reduction starts')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--inflate-train', action='store_true', default=False,
                        help='creates more training images by using transforms')
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

    # decide if to use cuda or not, then pick device accordingly
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # update arguments for training, including batch size and cuda
    train_kwargs = {'batch_size': args.batch_size}
    val_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)

    # Create transform, randomly do some flipping
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.RandomGrayscale(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        # This normalization is used on the test server
        transforms.Normalize([0.2404, 0.2967, 0.3563], [0.0547, 0.0527, 0.0477])
        ])

    # Create train and test set
    if not args.i == None:
        input_path = args.i
        if not input_path.endswith(os.sep):
            input_path = input_path + os.sep

    if not args.k == None:
        kfolds = args.k
    if not args.nsplits == None:
        run_splits = args.nsplits
        if run_splits > kfolds:
            run_splits = kfolds
            print("Maximal run splits is the number of folds")

    if args.standard_split:
        run_splits = 1

    print("\n======================================== new training run ========================================\nArgs for this run:")
    print(args)

    # create a train-test split
    split_list = make_splitlist(input_path, kfolds)


    for current_k in range(run_splits):
        print("\n======================================= running split " + str(current_k+1) + " =======================================\n")
        train_path = input_path + "train/"
        val_path = input_path + "val/"
        label_path = input_path + "label/"


        if not args.standard_split:
            if not args.reuse_split:
                split_files(input_path,split_list,current_k)
        else:
            setup_standard_split(input_path)

        # create additional training data
        if args.inflate_train:
            inflate_train_data(train_path,label_path + 'boat_mnist_labels_trainval.json')

        # create sets of images and Y variables
        train_set = Boats(root_dir=train_path, transform=transform,
                          gt_json_path= label_path + 'boat_mnist_labels_trainval.json')
        val_set = Boats(root_dir=val_path, transform=transform,
                        gt_json_path= label_path + 'boat_mnist_labels_trainval.json')

        # Create data loaders, contain both images and Y variables
        if use_cuda:
            print("training with cuda")
            train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
        else:# ensure data shuffling
            print("training with CPU")
            train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs, shuffle=True)

        test_loader = torch.utils.data.DataLoader(val_set, **val_kwargs)

        # Create network, optimizer and loss
        model = Net(args.dropout).to(device)

        # train with lr = learning rate
        #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) # stochastic gradient descent
        #criterion = nn.MSELoss() # maybe a class-balanced loss function is better?
        criterion = F.binary_cross_entropy # write our own balanced loss function?

        prev_lr = args.lr
        # Train and validate
        best_acc = 0
        best_trainloss = 10
        best_model_wts = copy.deepcopy(model.state_dict())
        for epoch in range(1, args.epochs + 1):
            # lr reduction
            new_lr = prev_lr

            if epoch > args.full_lr_epochs+1:
                if new_lr <= args.lr / 250 and args.lr_redux > 0:
                    print("LR at final floor, ending decrease")
                elif new_lr <= args.lr / 50 and args.lr_redux > 0:
                    print("LR at final floor, decreasing further")
                    new_lr = prev_lr-args.lr_redux/25
                elif new_lr <= args.lr / 10 and args.lr_redux > 0:
                    print("LR at first floor, slowing decrease")
                    new_lr = prev_lr-args.lr_redux/5
                else:
                    new_lr = prev_lr-args.lr_redux

            prev_lr = new_lr

            print("\nStarting new epoch, current LR: " + str(round(new_lr,ndigits=8)))

            #optimizer = optim.SGD(model.parameters(), lr=new_lr, momentum=args.momentum)
            optimizer = optim.Adam(model.parameters(), lr=new_lr)

            train_loss = train(args, model, device, train_loader, optimizer, criterion, epoch)
            print("Average train loss: " + str(train_loss))
            acc = test(model, device, test_loader, criterion)
            if acc > best_acc:
                best_acc = acc
                best_model_wts = copy.deepcopy(model.state_dict())
            elif acc == best_acc and train_loss < best_trainloss:
                best_trainloss = train_loss
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
