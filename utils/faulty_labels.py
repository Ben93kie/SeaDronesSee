# Creates a txt file containing imgs' IDs with displaced and missed bounding boxes (faulty_labels.txt)
# See below if wish to create seperate files for each case (displaced.txt and missed.txt)

import json
import argparse

if __name__ == '__main__':
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--original', default=None, type=str, help='Path to the original annotations file.')
        parser.add_argument('--fixed', default=None,type=str,help="Path to the fixed annotations file.")
        return parser.parse_args()

    opt = get_args()

    file1 = opt.original
    file2 = opt.fixed

    with open(file1) as f1:
        data1 = json.load(f1)

    with open(file2) as f2:
        data2 = json.load(f2)

    annotations1 = data1['annotations']
    annotations2 = data2['annotations']

    missed = []
    displaced = []
    for i in range(len(data1['images'])):
        img = data1['images'][i]
        img_id = img['id']
        img_annots1 = [annots for annots in annotations1 if annots['image_id'] == img_id]  # list of ditcs
        img_annots2 = [annots for annots in annotations2 if annots['image_id'] == img_id]  # list of ditcs
        if len(img_annots2) > len(img_annots1):
            missed.append(img_id)
            continue
        for j in range(len(img_annots2)):
            if img_annots1[j]['bbox'] != img_annots2[j]['bbox']:
                displaced.append(img_id)
                break
    faulty_labels =  missed + displaced

    # # Uncomment if you wish to have a sperate file for img ids with missed bounding boxes
    # with open('missed.txt', "w") as o:
    #     for i in missed:
    #         o.write("%s\n" % i)

    # # Uncomment if you wish to have a sperate file for img ids with displaced bounding boxes
    # with open('displaced.txt', "w") as o:
    #     for i in displaced:
    #         o.write("%s\n" % i)

    with open('faulty_labels.txt', "w") as o:
        for i in faulty_labels:
            o.write("%s\n" % i)

    print("Number of missed bounding boxes:",len(missed))
    print("Number of displaced bounding boxes:",len(displaced))
    print("Total:",len(faulty_labels))
