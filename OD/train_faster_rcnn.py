import time
import os
import argparse

import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from sds_dataset import SDSDataset
from predict_faster_rcnn import generate_prediction_file


def collate_fn(batch):
    return tuple(zip(*batch))


def log_print(message, file_name, create_log):
    print(message)
    if create_log:
        with open(file_name + '.txt', 'a') as of:
            of.write(message + '\n')


def get_model_dir(name, resize):
    time_id = time.strftime('%Y_%m_%d-%H_%M')
    return os.path.join('Trained Models', '{}_FasterRCNN_{}_{}'.format(time_id, name, resize))


def main():

    # Set directory paths
    train_data_dir = 'Datasets/seadronesea_august_splitted/images/train'
    train_annotation_dir = 'Datasets/seadronesea_august_splitted/annotations/instances_train.json'
    test_data_dir = 'Datasets/seadronesea_august_splitted/images/test'
    test_annotation_dir = 'Datasets/seadronesea_august_splitted/annotations/instances_test.json'

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', help='Choose resnet18, 50 or 101',
                        default='resnet18', type=str)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--image_size', default='720x1280', type=str, help='[height]x[width]')
    parser.add_argument('--create_prediction_file', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--create_log', action='store_true')
    args = parser.parse_args()

    # Get model directory name
    model_dir = get_model_dir(args.backbone, args.image_size)

    # Check if Cuda is available
    log_print(f'Cuda available: {torch.cuda.is_available()}', model_dir, args.create_log)
    if torch.cuda.is_available():
        # If yes, use GPU
        device = torch.device('cuda')
    else:
        # If no, use CPU
        device = torch.device('cpu')

    resize = (int(args.image_size.split('x')[0]), int(args.image_size.split('x')[1]))
    log_print(f'Images resized to: {args.image_size}', model_dir, args.create_log)

    # Create Datasets
    train_dataset = SDSDataset(train_data_dir, train_annotation_dir, resize)
    test_dataset = SDSDataset(test_data_dir, test_annotation_dir, resize)

    # Create Dataloader
    data_loader_train = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   collate_fn=collate_fn)
    data_loader_test = DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  collate_fn=collate_fn)

    # Use ResNet-Versions as Backbone
    log_print(f'Using {args.backbone} as CNN backbone', model_dir, args.create_log)
    if args.backbone == 'resnet18':
        modules = list(resnet18(weights=None).children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 512
    elif args.backbone == 'resnet50':
        modules = list(resnet50(weights=None).children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    elif args.backbone == 'resnet100':
        modules = list(resnet101(weights=None).children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048

    # Create Anchor Generator
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # Initialize FasterRCNN with Backbone and AnchorGenerator
    model = FasterRCNN(backbone=backbone,
                       rpn_anchor_generator=anchor_generator,
                       num_classes=train_dataset.num_classes)

    # Load checkpoint (optional)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(
            os.path.join('Trained Models', args.checkpoint)),
            strict=False)

    # Send model to device
    model.to(device)

    # Define learning rate, optimizer and scheduler
    learning_rate = 0.00001
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=learning_rate)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    # Initialize other hyperparameter
    num_epochs = 500
    early_stopping_tolerance = 5
    early_stopping_counter = 0

    # Start Training Process
    model.train()
    for epoch in range(num_epochs):

        # Training
        for images, targets in data_loader_train:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # print(targets)

            loss_dict = model(images, targets)
            sum_loss = sum(loss for loss in loss_dict.values())

            sum_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluation
        with torch.no_grad():
            for images, targets in data_loader_test:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                average_loss = sum(loss for loss in loss_dict.values()) / args.batch_size

        if epoch == 0:
            best_average_loss = average_loss

        # If model improved, save weights
        if best_average_loss >= average_loss:
            best_average_loss = average_loss
            early_stopping_counter = 0
            torch.save(
                {'model_state_dict': model.state_dict()},
                model_dir
            )

        # Otherwise, reduce learning rate
        else:
            early_stopping_counter += 1
            lr_scheduler.step()

        time_id = time.strftime('%Y_%m_%d-%H_%M')
        log_print(f'[{time_id}] '
                  f'Epoch {epoch} of {num_epochs} - Loss: {average_loss} - LR: {str(lr_scheduler.get_last_lr()[0])} '
                  f'- Early Stopping: {early_stopping_counter}/{early_stopping_tolerance}',
                  model_dir, args.create_log)

        if early_stopping_tolerance == early_stopping_counter:
            break

    log_print(f'Training stopped', model_dir, args.create_log)

    if args.create_prediction_file:
        # Create prediction file in coco format:
        val_data_dir = 'Datasets/seadronesea_august_splitted/images/val'
        val_annotation_dir = 'Datasets/seadronesea_august_splitted/annotations/instances_val.json'

        val_dataset = SDSDataset(val_data_dir, val_annotation_dir, resize)

        data_loader_val = DataLoader(val_dataset,
                                     batch_size=1,
                                     shuffle=True,
                                     collate_fn=collate_fn)

        generate_prediction_file(model, data_loader_val, device, resize)


if __name__ == '__main__':
    main()




