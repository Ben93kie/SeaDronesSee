import argparse
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import time
import os
import json

ROOT_DIR = './'

def evaluate_coco(file_name, categories, ground_truth):
    # load results in COCO evaluation tool
    coco_true = COCO(ground_truth)
    coco_pred = coco_true.loadRes(file_name)
    catList = []
    modelValue = [0, 0, 0, 0, 0]
    for cat in categories:
      coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
      coco_eval.params.catIds = cat;
      coco_eval.evaluate()
      coco_eval.accumulate()
      coco_eval.summarize()
      catList.append(list(coco_eval.stats[:3]) + list(coco_eval.stats[6:8]))
      modelValue[0] += coco_eval.stats[0]
      modelValue[1] += coco_eval.stats[1]
      modelValue[2] += coco_eval.stats[2]
      modelValue[3] += coco_eval.stats[6]
      modelValue[4] += coco_eval.stats[7]
    modelValue = [x/len(categories) for x in modelValue]
    ap95 = [x[0] for x in catList]
    return modelValue + list(ap95)

# rewrite labels for binary classification (water and non-water)
# non-water: category_id = 1
def rewrite_label(file_name):
    resultFile = open(file_name)
    data = json.load(resultFile)
    for bbox in data:
        bbox['category_id'] = 0
    resultFile.close()
    out_file = open(file_name + "_non-water", "w")
    json.dump(data, out_file)
    out_file.close()
    return(file_name + "_non-water")

def get_args():
    parser = argparse.ArgumentParser(
        'Evaluation script for the Object Detection task.')
    parser.add_argument('--file_name', type=str, help='Path to the predictions file.')
    parser.add_argument('--v2', action='store_true', default=False, help='Is it SeaDronesSee Object Detection v2?')
    parser.add_argument('--skip_sub_1', action='store_true', default=False, help='Skip SeaDronesSee Object Detection v2 subtrack 1')
    return parser.parse_args()

def v1_od():
    # swimmer, person, boat, swimmer on boat, floater on boat, life jacket
    categories = [1, 2, 3, 4, 5, 6]
    ground_truth = os.path.join(ROOT_DIR, 'annotations', 'instances_' + 'val_iscrowd' + '.json')
    return evaluate_coco(opt.file_name, categories, ground_truth)

def v2_od_subtrack1():
    # swimmer, boat, jetski, life_saving_appliances, buoy
    categories = [1, 2, 3, 4, 5]
    ground_truth = os.path.join(ROOT_DIR, 'annotations', 'od_v2_annotations', 'instances_' + 'val_iscrowd' + '.json')
    return evaluate_coco(opt.file_name, categories, ground_truth)

def v2_od_subtrack2(filename):
    # non-water
    categories = [0]
    ground_truth = os.path.join(ROOT_DIR, 'annotations', 'od_v2_annotations', 'instances_' + 'val_iscrowd' + '_non_water' + '.json')
    return evaluate_coco(filename, categories, ground_truth)

if __name__ == '__main__':
    opt = get_args()
    if(opt.v2):
      output = []
      if(opt.skip_sub_1):
        output.append([])
        filename = opt.file_name
      else:
        output.append(v2_od_subtrack1())
        filename = rewrite_label(opt.file_name)
      output.append(v2_od_subtrack2(filename))
      print(output)
    else:
      print(v1_od())
