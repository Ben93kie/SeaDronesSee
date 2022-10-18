#Converts a json annotation file to txt that works with darklabel
import argparse
import json
import time

categories = {"0":"ignored", "1":"swimmer", "2":"boat","3":"jetski","4":"life_saving_appliances","5":"buoy"}

if __name__ == '__main__':
    scriptTime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str, default=None, help='Path to the annotaion json file')
    parser.add_argument('--output', type=str, default=None, help='Path to the output txt file')
    options = parser.parse_args()
    if options.annotation == None:
        raise ValueError("Please enter a valid path to the annotation json file")
    if options.output == None:
        raise ValueError("Please enter a valid path to the output txt file")

    with open(options.annotation) as f:
        file = json.load(f)

    for img in file['images']:
        id = img['id']
        img_file_name = img['file_name']
        img_id = id
        img_annots = [annots for annots in file['annotations'] if annots['image_id'] == img_id]  # list of dics
        for annots in img_annots:    
            img_bbox = annots['bbox']
            img_annots_id = str(annots['id'])
            x1,y1,w,h = [str(x) for x in img_bbox]
            img_category_id = str(annots['category_id'])
            label = categories[img_category_id]
            line = [img_file_name,label,img_annots_id,x1,y1,w,h]
            line = ",".join(line) 
            with open(options.output, "a") as o:
                o.write(line+'\n')

    scriptRunTime = time.time() - scriptTime
    print('\nScript run time in seconds:',scriptRunTime)
