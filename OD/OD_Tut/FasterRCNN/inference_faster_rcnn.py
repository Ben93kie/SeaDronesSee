import os
import argparse

from PIL import Image, ImageDraw

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
    test_data_dir = 'Datasets/SeaDroneSee/images/val'
    test_annotation_dir = 'Datasets/SeaDroneSee/annotations/instances_val.json'

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', help='backbone of the Faster R-CNN', default='resnet18', type=str)
    parser.add_argument('--image_size', default='720x1280', type=str, help='[height]x[width]')
    parser.add_argument('--checkpoint', type=str, default=None, required=True) 
    parser.add_argument('--random_series', action='store_true')
    parser.add_argument('--score_threshold', default=0.5)
    parser.add_argument('--show_ground_truth', action='store_true')
    parser.add_argument('--image_number', default=10, type=int)
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
                                  shuffle=args.random_series,
                                  collate_fn=collate_fn)

    # Use ResNet-Versions as Backbone
    print(f'Using {args.backbone} as CNN backbone')
    if args.backbone == 'resnet18':
        modules = list(resnet18(weights=None).children())[:-2]
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
                       num_classes=test_dataset.num_classes)

    # Load checkpoint
    if args.checkpoint is not None:
        print(f'Load checkpoint: {args.checkpoint}')
        model.load_state_dict(torch.load(
            os.path.join('Trained Models', args.checkpoint)),
            strict=True)
    else:
        print('No checkpoint selected!')

    # Create Inference directory
    dir_path = os.path.join('Inference', args.checkpoint)
    os.mkdir(dir_path)

    # Send model to device
    model.to(device)
    model.eval()

    cntr = 0

    with torch.no_grad():
        for images, targets in data_loader_test:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            pred_dict = model(images)

            # Get image id and original size
            img_id = targets[0]['image_id'].item()
            org_width = targets[0]['org_w'].item()
            org_height = targets[0]['org_h'].item()

            # Open image with id
            path = test_dataset.coco.loadImgs(img_id)[0]['file_name']
            image = Image.open(os.path.join(test_data_dir, path))

            draw = ImageDraw.Draw(image)

            # Draw ground truth bounding boxes and labels (optional)
            if args.show_ground_truth:
                for box, label in zip(targets[0]['boxes'], targets[0]['labels']):
                    # Rescale bounding box coordinates
                    xmin = box[0].item() * (org_width / resize[1])
                    ymin = box[1].item() * (org_height / resize[0])
                    xmax = box[2].item() * (org_width / resize[1])
                    ymax = box[3].item() * (org_height / resize[0])

                    cat = test_dataset.coco.loadCats(label.item())[0]['name']

                    draw.rectangle(((xmin, ymin), (xmax, ymax)), width=3, outline='green')
                    draw.text((xmin - 10, ymin - 10),
                              'Category: {}'.format(cat),
                              fill='green', align='left')

            # Draw predicted bounding boxes, labels and scores
            for box, label, score in zip(pred_dict[0]['boxes'],
                                         pred_dict[0]['labels'],
                                         pred_dict[0]['scores']):
                if score.item() < float(args.score_threshold):
                    continue

                # Rescale bounding box coordinates
                xmin = box[0].item() * (org_width / resize[1])
                ymin = box[1].item() * (org_height / resize[0])
                xmax = box[2].item() * (org_width / resize[1])
                ymax = box[3].item() * (org_height / resize[0])

                cat = test_dataset.coco.loadCats(label.item())[0]['name']

                draw.rectangle(((xmin, ymin), (xmax, ymax)), width=3, outline='red')
                draw.text((xmin-10, ymin-10),
                          'Category: {}, Score:{}'.format(cat, str(score.item())[0:5]),
                          fill='red', align='left')

            # Save image with predictions
            image.save(os.path.join(dir_path, '{}.png'.format(img_id)))

            # Stop inference
            cntr += 1
            if cntr == args.image_number:
                break


if __name__ == '__main__':
    main()


