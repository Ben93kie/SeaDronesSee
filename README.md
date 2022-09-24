<head>
  <meta name="google-site-verification" content="IrTBOul_GquPsD_pSvsgGIenX7VTrGIM9fFXq7ebajc" />
</head>  

# SeaDronesSee
Vision Benchmark for Maritime Search and Rescue


<div align="center">
  <img src="./cover.gif" width="600"/>
 </div>
<br>
SeaDronesSee is a large-scale data set aimed at helping develop systems for Search and Rescue (SAR) using Unmanned Aerial Vehicles (UAVs) in maritime scenarios. Building highly complex autonomous UAV systems that aid in SAR missions requires robust computer vision algorithms to detect and track objects or persons of interest. This data set provides three sets of tracks: object detection, single-object tracking and multi-object tracking. Each track consists of its own data set and leaderboard. 

This repository contains evaluation kits, code and model samples and links to the SeaDronesSee benchmark and evaluation webserver.


## Data Sets

You can find the data sets on our webserver [here](https://seadronessee.cs.uni-tuebingen.de/dataset). Currently, we provide the following data sets:

Object Detection: 5,630 train images, 859 validation images, 1,796 testing images

Single-Object Tracking: 58 training video clips, 70 validation video clips and 80 testing video clips

Multi-Object Tracking: 22 video clips with 54,105 frames

Multi-Spektral Object Detection: 246 train images, 61 validation images, 125 testing images

Boat-MNIST (toy data set): Image Classification: 3765 train images, 1506 validation images, 2259 testing image

For Object Detection, Single-Object Tracking, Multi-Object Tracking we do not hand out the test set labels but only the inputs. To evaluate your model, you may upload your prediction on our webserver, which runs an automatic evaluation protocol that yields the performance, which can be compared on a leaderboard. For Boat-MNIST, we withold the entire test set (images and labels) and you can upload an ONNX-model that will be run on the webserver to yield the accuracy.

Soon, we will update the SeaDronesSee benchmark and add additional data - stay tuned.

### Object Detection

 You need so submit a json-file in COCO format. That is, a list of dictionaries, each containing a single prediction of the form

```
{ "image_id": 6503, "category_id": 4, "score": 0.6774558424949646, 
"bbox": [ 426.15203857421875, 563.6422119140625, 43.328399658203125, 18.97894287109375 ] }
```

The predictions are evaluated on AP50:0.05:95, AP50, AP75, AR1, AR10. You can upload your prediction [here](https://seadronessee.cs.uni-tuebingen.de/upload) upon registration. If you need, you can test your json-style COCO prediction with the evaluation script [here](OD/od.py). You have to replace GROUND_TRUTH_COCO_JSON.json with the corresponding json file from the webserver.


### Single-Object Tracking

 There are 80 testing video clips on which the performance is measured. The protocol is based on the implementation of [PyTracking](https://github.com/visionml/pytracking). Note that this is a short-term Single-Object Tracking task, meaning that clips only feature objects that are present in the video during the whole video and do not disappear and reappear.

You need to submit a zip-file containing exactly 80 text-files, each corresponding to the respective clip. Each text file has to be named j.txt where j is the number corresponding to the respective clip (1,...,80). Each text file has as many rows as its corresponding clip has frames. Each row has 4 comma separated numbers (x,y,w,h), where x is the left-most pixel value of the tracked object's bounding box, y the top-most pixel value and w and h the width and height of the bounding box in pixels.

Via the link for "Single-Object Tracking" above, you will find three json-files (SeaDronesSee_train.json, SeaDronesSee_val.json,SeaDronesSee_test.json). It is a dict of dicts, where for each track/clip number, you find the corresponding frames that need to be taken from the Multi-Object Tracking link.

For example, in the following you see the first track starting with the frame 000486.png, with corresponding path lake_constance_v2021_tracking/images/test/486.png, followed by frame 000487.png and so on. Afterwards, we have clip 2 starting with frame 000494.png and so on:

{"1": {"000486.png": "lake_constance_v2021_tracking/images/test/486.png", "000487.png": "lake_constance_v2021_tracking/images/test/487.png", "000488.png": "lake_constance_v2021_tracking/images/test/488.png",... "000636.png": "lake_constance_v2021_tracking/images/test/636.png"}, "2": {"000494.png": "lake_constance_v2021_tracking/images/test/494.png",...

Furthermore, you find three folder: train_annotations, val_annotations, test_annotations_first_frame. For the train and val case you find the corresponding annotations for the respective clip in the respective train or val set. Each clip has its own text file with each line corresponding to the bounding box for that frame. The test folder contains text files for each clip as well but only contains the bounding box ground truth for the very first frame and dummy values for the succeeding frames.

See also the compressed folder [sample_submission.zip](SOT/sample_submission.zip). This zip-archive could be uploaded right away but will naturally yield bad results.

The predictions are evaluated on precision and success numbers. 

### Multi-Object Tracking

 There are 22 video clips in the data on which you can test your trained tracker. The mapping from images to video clips can be done via the files 'instances_test_objects_in_water.json' and 'instances_test_swimmer.json', respectively. They can be found via the link above.
For example, '410.png' from the test set can be assigned to video clip 'DJI_0057.MP4' because its entry in the annotation file looks like this:

{'id': 410, 'file_name': '410.png', 'height': 2160, 'width': 3840, 'source': {'drone': 'mavic', 'folder_name': 'DJI_0057', 'video': 'DJI_0057.MP4', 'frame_no': 715}, 'video_id': 0, 'frame_index': 715, 'date_time': '2020-08-27T14:18:35.823800', 'meta': {'date_time': '2020-08-27T12:18:36', 'gps_latitude': 47.671949, 'gps_latitude_ref': 'N', 'gps_longitude': 9.269724, 'gps_longitude_ref': 'E', 'altitude': 8.599580615665955, 'gimbal_pitch': 45.4, 'compass_heading': 138.2, 'gimbal_heading': 140.9, 'speed': 0.6399828341528834, 'xspeed': -0.39998927134555207, 'yspeed': 0.39998927134555207, 'zspeed': 0.299991953509164}}

The submission format is similar to the one for the [MOT-challenge](https://motchallenge.net/). To submit your results, you have to upload a zip file containing one [video_id].txt file for each video clip in its top-level domain. The ID of each video can be obtained from the .json file. Information about each video there looks like this:

{'id': 0, 'height': 2160, 'width': 3840, 'name:': '/data/input/recordings/mavic/DJI_0057.MP4'}

Inside any of the .txt files there has to be one line per object per frame. Each line is formatted like: [frame_id],[object_id],x,y,w,h
frame_id and object_id are supposed to be integers, the rest of the numbers may be floats. The frame_id can be obtained from the .json file while the object_id can be assigned by your tracker. Coordinates x and y are the upper left coordinate of the bounding box while w and h are its width and height, respectively. All of these are expressed in pixels. 

### Multi-Spektral Object Detection

Currently, there is no challenge or leaderboard for this track.

### Boat-MNIST

This is a toy data set for the task of binary image classification. It aims at providing a simple hands-on benchmark to test small neural networks. There are the following two classes: 

1 - if the image contains any watercraft instance including boats, ships, surfboards, ... ON the water

0 - all the rest, i.e. just water or anything on the land (could also be boats)

Naturally, there may be edge cases (e.g. boats at the verge of the water and the shore).

As metrics, we employ the prediction accuracy (number of correctly predicted images divided by number of all images) and the number of parameters of the model. For this benchmark, you can upload your trained ONNX model to be ranked on the leaderboard. For that, please refer to [this sample script](Boat_MNIST/challenge_nn.py). It trains a simple single-layer perceptron architecture on this data set upon saving and exporting the Pytorch model as an ONNX file. Make sure the exported model uses the transformation provided in this code, as this is the transformation used for the webserver evaluation.

You can also find some sample solutions of some groups in the folders under Boat-MNIST with their respective group number.


#### Citation

If you find the data set, the benchmark or this repo useful, please consider citing

```
@inproceedings{varga2022seadronessee,
title={Seadronessee: A maritime benchmark for detecting humans in open water},
author={Varga, Leon Amadeus and Kiefer, Benjamin and Messmer, Martin and Zell, Andreas},
booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
pages={2260--2270},
year={2022} } 
```




