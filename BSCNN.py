import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


import torch #基本モジュール
from torch.autograd import Variable #自動微分用
import torch.nn as nn #ネットワーク構築用
import torch.optim as optim #最適化関数
import torch.nn.functional as F #ネットワーク用の様々な関数
import torch.utils.data #データセット読み込み関連
import torchvision #画像関連
from torchvision import datasets, models, transforms #画像用データセット諸々

data_dir = '../dataset/coco/'
data_type_val = 'val2014'
val_annFiles = data_dir + '2014/annotations/instances_' + data_type_val + '.json'
valImageDir = data_dir + '2014/valimages/'

data_type_train = 'train2014'
train_annFiles = data_dir + '2014/annotations/instances_' + data_type_train + '.json'
trainImageDir = data_dir + '2014/trainimages/'


def main():
    ## COCOのデータセットを用いる。（人物のラベルのみを利用し、14x14にリサイズを行う。）
    # val data の読み込み
    # COCO api の初期化
    coco = COCO(val_annFiles)

    # COCO の categories と supercategories の表示
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n\n', ' '.join(nms))
    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n', ' '.join(nms))

    # 指定したカテゴリーに含まれる画像
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)

    # 格納するための配列を用意
    val_images = []
    val_anns = []
    train_images = []
    train_anns = []

    # データを読み込み、配列へ格納
    for img_number in range(0, int(len(imgIds)/2)):
        imgId = imgIds[img_number]
        img = coco.loadImgs(imgId)[0]
        print('Image ID:\n', imgId)

        # load the selected image
        I = cv2.imread(valImageDir + img['file_name'])
        #cv2.imshow('window', I)
        #cv2.waitKey(0)

        # load the annotation in selected image
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        try:
            # 画像の幅と高さを取得
            # カラーとグレースケールで場合分け
            if len(I.shape) == 3:
                height, width, channels = I.shape[:3]
            else:
                height, width = I.shape[:2]
                channels = 1
            # セグメンテーション結果をのせるための黒画像を用意する
            blank_img = np.zeros((height, width, 1), np.uint8)
            for ann in anns:
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    mask_color = 255
                    pts = np.array(poly, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(blank_img, [pts], color=mask_color)

            # 画像のりサイズ
            resized_ann = cv2.resize(blank_img, (14, 14), interpolation=cv2.INTER_AREA)

            #cv2.imshow('mask', blank_img)
            #cv2.waitKey(0)

        except cv2.error:
            print('cv2 Error')
            continue
        except ValueError:
            print('numpy error')
            continue

        # 配列へ格納
        val_images.append(I)
        val_anns.append(resized_ann)

    cv2.imshow('color', val_images[10])
    cv2.waitKey(0)
    cv2.imshow('annotation', val_anns[10])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('val_images length:', len(val_images))
    print('val_anns:', len(val_anns))

    ## COCOのデータセットを用いる。（人物のラベルのみを利用し、14x14にリサイズを行う。）
    # train data の読み込み
    # COCO api の初期化
    coco = COCO(train_annFiles)

    # COCO の categories と supercategories の表示
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n\n', ' '.join(nms))
    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n', ' '.join(nms))

    # 指定したカテゴリーに含まれる画像
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)

    # データを読み込み、配列へ格納
    for img_number in range(0, int(len(imgIds)/2)):
        imgId = imgIds[img_number]
        img = coco.loadImgs(imgId)[0]
        print('Image ID:\n', imgId)

        # load the selected image
        I = cv2.imread(trainImageDir + img['file_name'])
        # cv2.imshow('window', I)
        # cv2.waitKey(0)

        # load the annotation in selected image
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        try:
            # 画像の幅と高さを取得
            # カラーとグレースケールで場合分け
            if len(I.shape) == 3:
                height, width, channels = I.shape[:3]
            else:
                height, width = I.shape[:2]
                channels = 1
            # セグメンテーション結果をのせるための黒画像を用意する
            blank_img = np.zeros((height, width, 1), np.uint8)
            for ann in anns:
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    mask_color = 255
                    pts = np.array(poly, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(blank_img, [pts], color=mask_color)

            # 画像のりサイズ
            resized_ann = cv2.resize(blank_img, (14, 14), interpolation=cv2.INTER_AREA)

            # cv2.imshow('mask', blank_img)
            # cv2.waitKey(0)

        except cv2.error:
            print('cv2 Error')
            continue
        except ValueError:
            print('numpy error')
            continue

        # 配列へ格納
        train_images.append(I)
        train_anns.append(resized_ann)

    cv2.imshow('color', train_images[10])
    cv2.waitKey(0)
    cv2.imshow('annotation', train_anns[10])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('train_images length:', len(train_images))
    print('train_anns:', len(train_anns))


    return 0


if __name__ == '__main__':
    main()