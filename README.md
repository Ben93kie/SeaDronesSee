# SeaDronesSee
Vision Benchmark for Maritime Search and Rescue

SeaDronesSee is a large-scale data set aimed at helping develop systems for Search and Rescue (SAR) using Unmanned Aerial Vehicles (UAVs) in maritime scenarios. Building highly complex autonomous UAV systems that aid in SAR missions requires robust computer vision algorithms to detect and track objects or persons of interest. This data set provides three sets of tracks: object detection, single-object tracking and multi-object tracking. Each track consists of its own data set and leaderboard. 

This repository contains evaluation kits, code and model samples and links to the SeaDronesSee benchmark and evaluation webserver.


## Data Set

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




#### Citation

If you find the data set, the benchmark our this repo useful, please consider citing

```
@inproceedings{varga2022seadronessee,
title={Seadronessee: A maritime benchmark for detecting humans in open water},
author={Varga, Leon Amadeus and Kiefer, Benjamin and Messmer, Martin and Zell, Andreas},
booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
pages={2260--2270},
year={2022} } 
```




