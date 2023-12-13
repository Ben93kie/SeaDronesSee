import argparse
import json

import cv2
import time
import os
import pyshine as ps
import cvzone


colors = {1: (0,255,0), 2: (0,0,255), 3: (255,255,51), 4: (0,255,255), 5: (255, 102, 255), 6: (51, 153, 255)}

if __name__ == '__main__':
    scriptTime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str, default=None, help='Path to the annotations json file')
    parser.add_argument('--images', type=str, default=None, help='Path to the images')
    parser.add_argument('--output', type=str, default=None, help='Path to output')
    options = parser.parse_args()
    if options.annotation == None:
        raise ValueError("Please enter a valid path to the annotation json file")
    if options.images == None:
        raise ValueError("Please enter a valid path to folder where the images are located")
    if not os.path.isdir(options.output):
        raise ValueError("The output dir was not found.")
    
    drone_65 = "drone_65.png"
    drone_65= cv2.imread(drone_65,-1)
    drone_65= cv2.resize(drone_65,(150,150))
    drone_35 = "drone_35.png"
    drone_35= cv2.imread(drone_35,-1)
    drone_35= cv2.resize(drone_35,(150,150))
    drone_35_65 = "drone_35_65.png"
    drone_35_65= cv2.imread(drone_35_65,-1)
    drone_35_65= cv2.resize(drone_35_65,(150,150))
    with open(options.annotation) as f:
        data = json.load(f)
        # needed to show the current progression
        numberOfImages = len(data['images'])
        currentImage = 0
        # currently the video has fixed 30 frames per second
        img_width = data['images'][0]['width']
        img_height = data['images'][0]['height']
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fps = 30
        count = 0
        videoName = options.output + "\\video_" + str(fps) + "fps_" + str(count) + ".mp4"
        # avoid overriding existing videos
        # is not really needed because a new output folder will be generated each time the script runs
        while os.path.isfile(videoName):
            count += 1
            videoName = options.output + "\\video_" + str(fps) + "fps_" + str(count) + ".mp4"
        out = cv2.VideoWriter(videoName, fourcc, fps, (img_width, img_height))
        for image in data['images']:
            currentImage += 1
            img = cv2.imread(options.images + '/' +image['file_name'])
            start = time.time()
            altitude = round(image['meta']['altitude'])
            saltitude = 'altitude= ' + str("%.2f" % (image['meta']['altitude'])) + ' m'
            speed = round(image['meta']['speed'])
            sspeed = 'speed= ' + str("%.2f" % (image['meta']['speed'])) + ' km/h'
            sangle = 'angle= ' + str("%.2f" %(image['meta']['gimbal_pitch'])) + ' degrees'
            angle = round(image['meta']['gimbal_pitch'])
            # draw all boundingBoxes, which belong to one image, on it with the corresponding colors of the categories
            for annotation in data['annotations']:
                if annotation['image_id'] == image['id']:
                    topLeftX = annotation['bbox'][0]
                    topLeftY = annotation['bbox'][1]
                    width = annotation['bbox'][2]
                    height = annotation['bbox'][3]
                    for category in data['categories']:
                        if category['id'] == annotation['category_id']:
                            label = category['name']
                    cv2.rectangle(img, (topLeftX, topLeftY), (topLeftX + width, topLeftY + height), colors[annotation['category_id']], 3)
                    ps.putBText(img,text=saltitude,text_offset_x=300,text_offset_y=10,vspace=10,hspace=10,font_scale=3.0,background_RGB=(228,225,222),text_RGB=(0,0,0),thickness=2,alpha=0.5)
                    ps.putBText(img,text=sspeed,text_offset_x=round((img_width/2)-500),text_offset_y=10,vspace=10,hspace=10,font_scale=3.0,background_RGB=(228,225,222),text_RGB=(0,0,0),thickness=2,alpha=0.5)
                    ps.putBText(img,text=sangle,text_offset_x=img_width-1200,text_offset_y=10,vspace=10,hspace=10,font_scale=3.0,background_RGB=(228,225,222),text_RGB=(0,0,0),thickness=2,alpha=0.5)
                    # draw lineal
                    cv2.line(img,(15,0),(15,2160),(0,0,0),thickness=2)
                    cv2.line(img,(215,0),(215,2160),(0,0,0),thickness=2)
                    d = 0
                    for i in range(15,-1,-1):
                        v_lines_distance = 200
                        cv2.line(img,(15,150+d),(215,150+d),(0,0,0),thickness=2)
                        cv2.putText(img,str(i*10)+" m",(240,150+d+10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
                        d =d +120
                        
                    drone_y_position = 2160 - 210 - round(drone_35_65.shape[0]/2) - round(altitude*120/10)

                    if angle < 35:
                        img = cvzone.overlayPNG(img, drone_35,[40,drone_y_position])
                    if angle >= 35 and angle <65:
                        img = cvzone.overlayPNG(img, drone_35_65,[40, drone_y_position])
                    if angle >= 65:
                        img = cvzone.overlayPNG(img, drone_65,[40,drone_y_position])
            out.write(img)
            print('Finished: {0}%'.format(round((currentImage / numberOfImages) * 100, 2)))
        # save video
        out.release()
        scriptRunTime = time.time() - scriptTime
        print('\nScript run time in seconds: {0}'.format(scriptRunTime))
        print("mean time per image: {0}".format(scriptRunTime / numberOfImages))