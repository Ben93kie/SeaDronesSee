import os
import argparse
import numpy as np
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Evaluator:
    def __init__(self, annotation_dir):

        self.annotation_dir = annotation_dir

        self.coco = COCO(annotation_dir)
        self.image_ids = list(self.coco.imgs.keys())
        self.annotations = self.get_annotations()

        self.predictions = {
            "images": self.annotations["images"].copy(),
            "categories": self.annotations["categories"].copy(),
            "annotations": None
        }

    def get_annotations(self):
        with open(self.annotation_dir, 'r') as f:
            data = json.load(f)

        for d in data['annotations']:
            d['iscrowd'] = 0

        return data

    def get_predictions(self, preds):
        with open(os.path.join('Prediction Files', preds), 'r') as f:
            data = json.load(f)

        for new_id, d in enumerate(data, start=1):
            d['id'] = new_id
            d['iscrowd'] = 0
            d['area'] = d['bbox'][2] * d['bbox'][3]

        return data

    def evaluate(self, pred_file, n_imgs=-1):

        self.predictions["annotations"] = self.get_predictions(pred_file)

        coco_ds = COCO()
        coco_ds.dataset = self.annotations
        coco_ds.createIndex()

        coco_dt = COCO()
        coco_dt.dataset = self.predictions
        coco_dt.createIndex()

        imgIds = sorted(coco_ds.getImgIds())

        if n_imgs > 0:
            imgIds = np.random.choice(imgIds, n_imgs)

        cocoEval = COCOeval(coco_ds, coco_dt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.params.useCats = True
        cocoEval.params.iouType = "bbox"
        cocoEval.params.iouThrs = np.array([0.4])

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        return cocoEval



def main():

    # Set annotation paths
    annotation_dir = 'Datasets/SeaDroneSee/annotations/instances_val.json'

    # Parse arguments
    parser = argparse.ArgumentParser(description='Test Faster R-CNN on SeaDroneSee')
    parser.add_argument('--image_size', default='720x1280', type=str, help='[height]x[width]')
    parser.add_argument('--prediction_file', type=str, default=None, required=True)
    args = parser.parse_args()

    evaluator = Evaluator(annotation_dir)

    evaluator.evaluate(args.prediction_file)


if __name__ == '__main__':
    main()



