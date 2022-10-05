# Hardware Requirements

A Nvidia GPU that supports CUDA is recommended. 

# Setup

1. Clone the SeaDroneSee repository:
    ```git clone https://github.com/Ben93kie/SeaDronesSee.git```

2. Download and Install Anaconda on your device: <br>
https://www.anaconda.com/products/distribution <br> 

3. Create a virtual environment: <br>
    ```conda create --name <env_name>``` <br>
    For additional information visit: <br>
    https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

4. Activate the virtual environment: <br>
    ```conda activate <env_name>``` <br>

5. Install Pytorch (https://pytorch.org/): <br>
    ```conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch``` <br>

6. Install Pycocotools: <br>
    ``` conda install -c conda-forge pycocotools``` <br>

# Training 

Download the SeaDroneSee Dataset from our official website and move it to the 'Datasets' directory. <br>
The 'Dataset' directory should look like this: 
```
/SeaDroneSee
├── /annotations
└── /images
	├── /val
	└── /train
```

To start the training, run ```python train_faster_rcnn.py```. <br>
To create a file that saves the console output, add ```--create_log```. <br>
To create a prediction-file on the validation set after the training, add ```--create_prediction_file```. <br>

Additional input arguments are:
- ```backbone```: Determines CNN-backbone used in the Faster R-CNN.
- ```batch_size```
- ```image_size```: Rescales all images to the given size.
- ```checkpoint```: Loads checkpoint for training.

The model checkpoints are saved in the 'Trained Models' directory.
 
# Evaluation, Visualization

To evaluate the model, run: ```python test_faster_rcnn.py --checkpoint <checkpoint_name>```.  <br>
The evaluation script TODO <br>
To visualize the model results, run: ```python inference_faster_rcnn.py --checkpoint <checkpoint_name>```. <br>
The script saves images with the predicted bounding boxes and labels in the 'Inference' directory.
Make sure to specify a checkpoint for evaluation or inference. <br>

# Creating a Prediction File:

To participate in the challenge you have to hand in a json-file in COCO format. That means a list of dictionaries, each containing one prediction: <br>
```
{ "image_id": 6503, "category_id": 4, "score": 0.6774558424949646, 
"bbox": [ 426.15203857421875, 563.6422119140625, 43.328399658203125, 18.97894287109375 ] }
```
To create a prediction file, run: ```python predict_faster_rcnn.py --checkpoint <checkpoint_name>```. <br>
Make sure to specify a checkpoint. <br>
The prediction file is stored in the 'Prediction Files' directory. <br>
