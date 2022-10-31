import json
import argparse
import od


if __name__ == '__main__':

    def get_args():
        parser = argparse.ArgumentParser(
            'Evaluation script for the Object Detection task.')
        parser.add_argument('--ground_truth', type=str, help='Path to the ground truth file.')
        parser.add_argument('--file_name', type=str, help='Path to the predictions file.')
        parser.add_argument('--altitude',nargs='+', default=None,help='Altitude to run the evaluation script with [0,50],[50,100],[100,150]')
        return parser.parse_args()
    
    opt = get_args()

    low, high = opt.altitude


    ground_truth = open(opt.ground_truth)
    ground_truth = json.load(ground_truth)

    images_in_chosen_altitude = []
    for img in ground_truth['images']:
        if img['meta']:
            try:
                img_altitude = img['meta']['height_above_takeoff(meter)']
            except KeyError:
                img_altitude = img['meta']['altitude']
            finally:
                if img_altitude > int(low) and img_altitude <= int(high):
                    images_in_chosen_altitude.append(img)

    ground_truth['images']=images_in_chosen_altitude
    
    images_annotations_in_chosen_altitude = []
    for annots in ground_truth['annotations']: 
        for img in ground_truth['images']:
            if annots['image_id'] == img['id']:
                images_annotations_in_chosen_altitude.append(annots)
                
    ground_truth['annotations'] = images_annotations_in_chosen_altitude
    truth_output_pth = "truth.json"
    truth_output = open(truth_output_pth, "w")
    json.dump(ground_truth, truth_output)

    with open(opt.file_name) as f:
        pred_file = json.load(f)
    
    preds_lst = []
    for img in ground_truth['images']:
        for entry in pred_file:
            if entry['image_id'] == img['id']:
                preds_lst.append(entry)
    pred_file = preds_lst
    preds_output_pth = "preds.json"
    preds_output= open(preds_output_pth, "w")
    json.dump(pred_file,preds_output)

    preds_output.close()
    truth_output.close()

    od.evaluate_coco(file_name=preds_output_pth,categories=[1, 2, 3, 4, 5],ground_truth=truth_output_pth)
    
