import argparse
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import time
import os

ROOT_DIR = './'

def evaluate_coco(file_name):
    # load results in COCO evaluation tool
    coco_true = COCO(os.path.join(ROOT_DIR, 'GROUND_TRUTH_COCO_JSON.json'))
    coco_pred = coco_true.loadRes(file_name)

    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print(list(coco_eval.stats[:3]) + list(coco_eval.stats[6:8]))

    return list(coco_eval.stats[:3]) + list(coco_eval.stats[6:8])


def get_args():
    parser = argparse.ArgumentParser(
        'Evaluation script for the Object Detection task.')
    parser.add_argument('--file_name', type=str, help='Path to the predictions file.')

    return parser.parse_args()


if __name__ == '__main__':
    opt = get_args()

    evaluate_coco(opt.file_name)
