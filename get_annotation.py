import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

from pycocotools.coco import COCO

# set the directories of coco datasets
dataDir = '../Datasets/coco/'
dataType = 'val2014'
annFiles = dataDir + 'annotations/instances_' + dataType + '.json'
ImageDir = dataDir + 'images/' + dataType + '/'

def main():
    # initialize COCO api for instance annotations
    coco = COCO(annFiles)
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n\n', ' '.join(nms))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n', ' '.join(nms))

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    imgId = imgIds[np.random.randint(0, len(imgIds))]
    img = coco.loadImgs(imgId)[0]
    print('Image ID:\n', imgId)

    # load the selected image
    I = cv2.imread(ImageDir + img['file_name'])
    cv2.imshow('window', I)
    cv2.waitKey(0)

    # load the annotation in selected image
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    try:
        for ann in anns:
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                mask_color = (random.randrange(255), random.randrange(255), random.randrange(255))
                pts = np.array(poly, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(I, [pts], color=mask_color)

        cv2.imshow('mask', I)
        cv2.waitKey(0)

    except cv2.error:
        print('cv2 Error')
    except ValueError:
        print('numpy error')

    return 0


if __name__ == '__main__':
    main()
