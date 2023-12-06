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
    # categories = [1, 2, 3, 4, 5, 6]
    # categories = [1, 2, 3, 6]
    categories = [0]
    #MOT
    # categories = [0]
    # [{"supercategory": "person", "id": 1, "name": "swimmer"},
    #  {"supercategory": "person", "id": 2, "name": "swimmer with life jacket"},
    #  {"supercategory": "boat", "id": 3, "name": "boat"},
    #  {"supercategory": "lifejacket", "id": 6, "name": "life jacket"}],
    # ground_truth = os.path.join(ROOT_DIR, 'annotations', 'instances_' + 'val_iscrowd' + '.json')
    # ground_truth = '/home/kiefer/PycharmProjects/3D_Tracking/gt_single_classs.json'
    # ground_truth = '/home/kiefer/TCML/yolov7_tiny/yolov7_tiny_mit_early_stopping/result/gt_single_class.json'
    # ground_truth = '/cshome/share/avalon/dataset/lake_constance_v2021_tracking/annotations/instances_test_objects_in_water_gt_iscrowd.json'
    # ground_truth = '/home/kiefer/TCML/yolov7_tiny/yolov7_tiny_mit_early_stopping/result/gt_single_class.json'
    ground_truth = '/cshome/share/avalon/dataset/seadronessee_august_splitted_challenge/annotations/instances_test.json'
    print("using ground truth: ", ground_truth)
    return evaluate_coco(opt.file_name, categories, ground_truth)

def v2_od_subtrack1():
    # swimmer, boat, jetski, life_saving_appliances, buoy
    # categories = [1, 2, 3, 4, 5]
    categories = [0]
    ground_truth = os.path.join(ROOT_DIR, 'annotations', 'od_v2_annotations', 'instances_' + 'val_iscrowd' + '.json')
    ground_truth = '/home/kiefer/Documents/SeaDronesSee/OD/evaluation of OD V2/required files/new_instances_shuffled_test_iscrowd.json'
    ground_truth = '/home/kiefer/PycharmProjects/3D_Tracking/gt_single_classs.json'
    return evaluate_coco(opt.file_name, categories, ground_truth)

def v2_od_subtrack2(filename):
    # non-water
    categories = [0]
    ground_truth = os.path.join(ROOT_DIR, 'annotations', 'od_v2_annotations', 'instances_' + 'val_iscrowd' + '_non_water' + '.json')
    ground_truth = '/home/kiefer/Documents/SeaDronesSee/OD/evaluation of OD V2/required files/new_instances_test_iscrowd_non_water.json'
    ground_truth = '/home/kiefer/PycharmProjects/3D_Tracking/gt_single_classs.json'
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
