# Creates a new metadata file after converting a darklabel txt file to a json and replaces the "annotaions" entry.
import json
import time
import argparse

categories = {"0":"ignored", "1":"swimmer", "2":"boat","3":"jetski","4":"life_saving_appliances","5":"buoy"}

if __name__ == '__main__':
    scriptTime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', type=str, default=None, help='Path to the txt file')
    parser.add_argument('--annotation', type=str, default=None, help='Path to the annotation json file')
    parser.add_argument('--output', type=str, default=None, help='Path to the output json txt file')
    options = parser.parse_args()
    if (options.txt == None):
        raise ValueError("Please enter a valid path to the txt file")
    if options.annotation == None:
        raise ValueError("Please enter a valid path to the original annotation json file")
    if options.output == None:
        raise ValueError("Please enter a valid path to the output json file")
    
    reversed_categories = {v: k for k, v in categories.items()}

    with open(options.txt,'r') as t:
        lines = t.readlines()

    annot_list = []
    for line in lines:
        file_name,ctg,bbox_id,x1,y1,w,h = line.split(',')
        x1,y1,w,h = int(x1), int(y1), int(w), int(h)
        t_image_id = int(file_name.split('.')[0])
        t_ctg = int(reversed_categories[ctg])
        t_id = int(bbox_id)
        t_bbox = [x1,y1,w,h]
        t_area = w * h
        annot =  {'id': t_id, 'image_id': t_image_id,'bbox': t_bbox,'area': t_area,'category_id': t_ctg}
        annot_list.append(annot)

    with open(options.annotation) as f:
        data = json.load(f)
    
    output = data
    output['annotations'] = annot_list

    with open(options.output, 'w') as o:
        json.dump(output,o,ensure_ascii=False)
    
    scriptRunTime = time.time() - scriptTime
    print('\nScript run time in seconds:',scriptRunTime)