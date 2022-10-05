import time
import json
import argparse
import os

import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torch import nn
from torch.utils.data import DataLoader

from sds_dataset import SDSDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def generate_prediction_file(model, data_loader, device, resize):
    model.eval()

    # Create List for Predictions
    pred_list = list()

    with torch.no_grad():
        for images, targets in data_loader:

            images = list(image.to(device) for image in images)
            pred_dict = model(images)

            # Get image id and original size
            img_id = targets[0]['image_id'].item()
            org_width = targets[0]['org_w'].item()
            org_height = targets[0]['org_h'].item()

            # For every prediction:
            for box, label, score in zip(pred_dict[0]['boxes'],
                                         pred_dict[0]['labels'],
                                         pred_dict[0]['scores']):

                # Create Dictionary with
                pred_dict_coco = dict()
                pred_dict_coco['image_id'] = img_id
                # Predicted Label
                pred_dict_coco['category_id'] = label.item()
                # Confidence Score
                pred_dict_coco['score'] = score.item()
                # Predicted Bounding Box
                xmin = box[0].item() * (org_width/resize[1])
                ymin = box[1].item() * (org_height/resize[0])
                width = (box[2].item() - box[0].item()) * (org_width/resize[1])
                height = (box[3].item() - box[1].item()) * (org_height/resize[0])
                pred_dict_coco['bbox'] = [xmin, ymin, width, height]
                # And append Dictionary to List
                pred_list.append(pred_dict_coco)

    time_id = time.strftime('%Y_%m_%d-%H_%M')
    with open(os.path.join('Prediction Files', 'prediction{}.json'.format(time_id)), 'w') as f:
        json.dump(pred_list, f, ensure_ascii=False, indent=4)


def main():
    # Set directory paths
    data_dir = 'Datasets/SeaDroneSee/images/val'
    annotation_dir = 'Datasets/SeaDroneSee/annotations/instances_val.json'

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', help='backbone of the Faster R-CNN', default='resnet18', type=str)
    parser.add_argument('--image_size', default='720x1280', type=str, help='[height]x[width]')
    parser.add_argument('--checkpoint', type=str, default=None,
                        required=True)
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
    dataset = SDSDataset(data_dir, annotation_dir, resize)
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=collate_fn)

    # Use ResNet-Versions as Backbone
    print(f'Using {args.backbone} as CNN backbone')
    if args.backbone == 'resnet18':
        modules = list(resnet18(weights=None).children())[:-2]
        # print(modules)
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 512
    elif args.backbone == 'resnet50':
        modules = list(resnet50(weights=None).children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
    elif args.backbone == 'resnet101':
        modules = list(resnet101(weights=None).children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048

    # Create Anchor Generator
    anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # Initialize FasterRCNN with Backbone and AnchorGenerator
    model = FasterRCNN(backbone=backbone,
                       rpn_anchor_generator=anchor_generator,
                       num_classes=dataset.num_classes)

    # Load checkpoint
    if args.checkpoint is not None:
        print(f'Load checkpoint: {args.checkpoint}')
        model.load_state_dict(torch.load(
            os.path.join('Trained Models', args.checkpoint)),
            strict=True)
    else:
        print('No checkpoint selected!')

    # Send model to device
    model.to(device)

    # Generate prediction file
    generate_prediction_file(model, data_loader, device, resize)


if __name__ == '__main__':
    main()
