import os
from PIL import Image

from torchvision.transforms import Compose, ToTensor, Resize
import pycocotools.coco as pyco
import torch
from torch.utils.data import Dataset


class SDSDataset(Dataset):
    def __init__(self, root, annotation_file, resize):
        self.root = root
        self.coco = pyco.COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.num_classes = len(self.coco.cats)
        self.resize = resize
        self.transform = Compose([
            Resize(resize),
            ToTensor()
            ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco

        # Image ID of the input image
        img_id = self.ids[index]
        # Annotation IDs from coco
        ann_ids = coco.getAnnIds(img_id)
        # Load Annotation for the input image
        coco_annotation = coco.loadAnns(ann_ids)
        # Get path for the input image
        path = coco.loadImgs(img_id)[0]['file_name']

        # Open input image
        org_image = Image.open(os.path.join(self.root, path))

        # Get size of input image
        org_height = org_image.height
        org_width = org_image.width

        # Apply transformation (resize) to input image
        image = self.transform(org_image)

        # Get number of objects in the input image
        num_objects = len(coco_annotation)

        # Get bounding boxes and category labels
        # Coco format: bbox = [xmin, ymin, width, height]
        # Pytorch format: bbox = [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in range(num_objects):
            # Convert and resize boxes
            xmin = coco_annotation[i]['bbox'][0] / (org_width/self.resize[1])
            ymin = coco_annotation[i]['bbox'][1] / (org_height/self.resize[0])
            xmax = xmin + coco_annotation[i]['bbox'][2] / (org_width/self.resize[1])
            ymax = ymin + coco_annotation[i]['bbox'][3] / (org_height/self.resize[0])
            labels.append(coco_annotation[i]['category_id'])
            boxes.append([xmin, ymin, xmax, ymax])

        # Convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([img_id])

        # Get (rectangular) size of bbox
        areas = []
        for i in range(num_objects):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # Get Iscrowd
        iscrowd = torch.zeros((num_objects,), dtype=torch.int64)

        # Create annotation dictionary
        annotation = dict()
        annotation['boxes'] = boxes
        annotation['labels'] = labels
        annotation['image_id'] = img_id
        annotation['area'] = areas
        annotation['iscrowd'] = iscrowd

        # Save width and height of the original image to rescale bounding boxes later on
        annotation['org_h'] = torch.as_tensor(org_height, dtype=torch.int64)
        annotation['org_w'] = torch.as_tensor(org_width, dtype=torch.int64)

        return image, annotation
