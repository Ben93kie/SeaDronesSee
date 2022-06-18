from PIL import Image
import os
import json
path = os.path.join("dataset","Boat-MNIST",)
imagedir_path = os.path.join(path,"train")
json_path = os.path.join(path,"boat_mnist_labels_trainval.json")
img_list = os.listdir(imagedir_path)

labels_dict = {}
with open(json_path) as file:
    labels_dict = json.load(file)

for img in img_list:
    
    label = labels_dict[img]
    img_name = img.split('.')[0]
    vertical_path = os.path.join(imagedir_path,img_name+"_vertical.png")
    horizontal_path = os.path.join(imagedir_path,img_name+"_horizontal.png")
    labels_dict[vertical_path] = label
    labels_dict[horizontal_path] = label
    original_img = Image.open(os.path.join(imagedir_path,img))
    vertical_img = original_img.transpose(method=Image.FLIP_TOP_BOTTOM)
    vertical_img.save(vertical_path)
    horizontal_img = original_img.transpose(method=Image.FLIP_LEFT_RIGHT)
    horizontal_img.save(horizontal_path)
    original_img.close()
    vertical_img.close()
    horizontal_img.close()

with open(os.path.join(path,"boat_mnist_labels_trainval_augmented.json"), "w") as outfile:
    json.dump(labels_dict, outfile)

#vertical_img = original_img.transpose(method=Image.FLIP_TOP_BOTTOM)