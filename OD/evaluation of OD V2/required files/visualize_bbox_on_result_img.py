import argparse
import json

import cv2
import time
import os

'''
one color for each category (excluding category nr 7)
    1: swimmer
    2: floater
    3: boat
    4: swimmer on boat
    5: floater on boat
    6: life jacket
    7: ignored
'''
colors = {0: (139, 0, 0), 1: (0,255,0), 2: (0,0,255), 3: (255,255,51), 4: (0,255,255), 5: (255, 102, 255), 6: (51, 153, 255)}

if __name__ == '__main__':
    scriptTime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str, default=None, help='path to the annotations json file')
    parser.add_argument('--pictures', type=str, default=None, help='path to the pictures')
    parser.add_argument('--output', type=str, default=None, help='Path to output')
    options = parser.parse_args()
    if (options.annotation == None):
        print("Please enter a valid path to the annotation json file ")
        exit()
    if (options.pictures == None):
        print("Please enter a valid path to folder where the pictures are located")
        exit()
    if (options.output == None):
        print("Please enter a valid path for the output directory")
        exit()

    # create Output Folder, avoid overriding any existing ones
    num = 0
    outputPath = options.output + "/Output" + str(num)
    while os.path.exists(outputPath):
        num += 1
        outputPath = options.output + "/Output" + str(num)
    os.mkdir(outputPath)

    with open(options.annotation) as f:
        data = json.load(f)

        # needed to show the current progression
        numberOfImages = len(data)
        currentImage = 0

        for image in data:
            currentImage += 1
            if not os.path.exists(options.pictures + "/" + str(image['image_id']) + ".jpg"):
                continue
            if os.path.exists(outputPath + "/" + str(image['image_id']) + ".jpg"):
                img = cv2.imread(outputPath + "/" + str(image['image_id']) + ".jpg")
            else:
                img = cv2.imread(options.pictures + "/" + str(image['image_id']) + ".jpg")
            start = time.time()
            # draw all boundingBoxes, which belong to one image, on it with the corresponding colors of the categories
            topLeftX = int(image['bbox'][0])
            topLeftY = int(image['bbox'][1])
            width = int(image['bbox'][2])
            height = int(image['bbox'][3])
            cv2.rectangle(img, (topLeftX, topLeftY), (topLeftX + width, topLeftY + height), colors[image['category_id']], 3)
            cv2.imwrite(outputPath + "/" + str(image['image_id']) + ".jpg", img)
            print('Picture generation took {0} second! Finished: {1}%'.format(time.time() - start, round((currentImage / numberOfImages) * 100, 2)))
        scriptRunTime = time.time() - scriptTime