import os
import argparse

from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sds_dataset import SDSDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def main():

    # Set directory paths
    test_data_dir = 'Datasets/seadronesea_august_splitted/images/val'
    test_annotation_dir = 'Datasets/seadronesea_august_splitted/annotations/instances_val.json'

    # Parse arguments
    parser = argparse.ArgumentParser(description='Test Faster R-CNN on SeaDroneSee')
    parser.add_argument('--backbone', help='backbone of the Faster R-CNN', default='resnet18', type=str)
    parser.add_argument('--image_size', default='720x1280', type=str, help='[height]x[width]')
    parser.add_argument('--checkpoint', type=str, default=None, required=True)
    args = parser.parse_args()

    # Check if Cuda is available
    print(f'Cuda available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        # If yes, use GPU
        device = torch.device('cuda')
    else:
        # If no, use CPU
        device = torch.device('cpu')

    resize = (int(args.image_size.split('x')[0]), int(args.image_size.split('x')[1]))
    print(f'Images resized to: {args.image_size}')

    # Crate Dataset and Dataloader
    test_dataset = SDSDataset(test_data_dir, test_annotation_dir, resize)
    data_loader_test = DataLoader(test_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  collate_fn=collate_fn)

    # Use ResNet50 as Backbone
    print(f'Using {args.backbone} as CNN backbone')
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
                       num_classes=test_dataset.num_classes)

    # Load checkpoint (optional)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(
            os.path.join('Trained Models', args.checkpoint)),
            strict=False)

    # Send model to device
    model.to(device)
    model.train()

    # Evaluation
    with torch.no_grad():
        for images, targets in data_loader_test:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            average_loss = sum(loss for loss in loss_dict.values()) / args.batch_size


if __name__ == '__main__':
    main()



