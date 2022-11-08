# Visualize certain images or images within a given range
import numpy as np
import cv2
import json 
import argparse
import os 

colors = {0:(0,100,100),1: (0,255,0), 2: (255,255,51), 3: (0,255,255), 4: (255,0,0), 5: (255, 102, 255), 6: (51, 153, 255)}



if __name__ == '__main__':
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--annotations', default=None, type=str, help='Path to the annotations file.')
        parser.add_argument('--image_folder', default=None, type=str, help='Path to the images folder.')
        parser.add_argument('--images',nargs='+',default=None, help="Images IDs e.g. 4331 2333")
        parser.add_argument('--range',nargs='+',default=None, help="Images ID range e.g. 100 200")
        parser.add_argument('--outdir', default=None,type=str,help="Output directory")
        return parser.parse_args()

    opt = get_args()
    imgs = opt.images

    with open(opt.annotations) as f:
        data = json.load(f)
    images = data['images']
    
    
    if imgs: 
        images_to_generate = [str(x) + '.jpg' for x in imgs]

    if opt.range is not None:
        low, high = opt.range
        low, high = int(low), int(high)
        if low >= 0 and high > low: 
            images_to_generate = [str(x) + '.jpg' for x in range(low,high)]
        if low and imgs:
            raise("Can not run script with images and range, please choose only one at a time")
    
    idxs=[]
    idx =0

    for img in images:
        if img['file_name'] in images_to_generate:
            idxs.append(idx)
        idx += 1

    for i in idxs:
        image = images[i]
        img = cv2.imread(opt.image_folder + '/' +image['file_name'])
        img_width = data['images'][0]['width']
        img_height = data['images'][0]['height']
        for annotation in data['annotations']:
            if annotation['image_id'] == image['id']:
                    topLeftX = annotation['bbox'][0] if annotation['bbox'][0] > 0 else 0
                    topLeftY = annotation['bbox'][1]
                    width = annotation['bbox'][2] if annotation['bbox'][2] < img_width else img_width
                    height = annotation['bbox'][3] 
                    for category in data['categories']:
                        if category['id'] == annotation['category_id']:
                            label = category['name']
                    cv2.rectangle(img, (topLeftX, topLeftY), (topLeftX + width, topLeftY + height), colors[annotation['category_id']], 3)
        cv2.imwrite(opt.outdir + image['file_name'], img)
   



