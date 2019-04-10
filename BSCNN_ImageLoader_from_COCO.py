import numpy as np
import cv2
from pycocotools.coco import COCO

# data_dir = '../dataset/coco/'
# data_type_val = 'val2014'
# annFiles = data_dir + '2014/annotations/instances_' + data_type_val + '.json'
# ImageDir = data_dir + '2014/valimages/'

# data_type_train = 'train2014'
# annFiles = data_dir + '2014/annotations/instances_' + data_type_train + '.json'
# ImageDir = data_dir + '2014/trainimages/'


# 入力:(総ブロック数, 読み込むブロックの番号)
# 出力:画像を格納したNumpyの４次元配列
def ImageLoaderFromCOCO(minibatch_num, minibatch_count, annFiles, ImageDir):
    ## COCOのデータセットを用いる。（人物のラベルのみを利用し、14x14にリサイズを行う。）
    # val data の読み込み
    # COCO api の初期化
    coco = COCO(annFiles)

    # COCO の categories と supercategories の表示
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    #print('COCO categories: \n\n', ' '.join(nms))
    nms = set([cat['supercategory'] for cat in cats])
    #print('COCO supercategories: \n', ' '.join(nms))

    # 指定したカテゴリーに含まれる画像
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)

    # 格納するための配列を用意
    images_list = []
    anns_list = []

    # mini batchの数に応じて読み込むデータ数を変える
    loading_num = int(len(imgIds) / minibatch_num)
    start_num = minibatch_count * loading_num
    end_num = start_num + loading_num
    if len(imgIds) < end_num:
        end_num = len(imgIds)

    # データを読み込み、配列へ格納
    for img_number in range(start_num, end_num):
        imgId = imgIds[img_number]
        img = coco.loadImgs(imgId)[0]
        #print('Image ID: ', imgId)

        # load the selected image
        I = cv2.imread(ImageDir + img['file_name'])
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
            resized_img = cv2.resize(I, (448, 448))
            resized_ann = cv2.resize(blank_img, (14, 14), interpolation=cv2.INTER_AREA)

            #cv2.imshow('mask', blank_img)
            #cv2.waitKey(0)

        except cv2.error:
            #print('cv2 Error')
            continue
        except ValueError:
            #print('numpy error')
            continue

        # 配列へ格納
        images_list.append(resized_img)
        anns_list.append(resized_ann)

    # Numpyへの変換 & メモリの開放
    images_np = np.array(images_list, dtype=np.float32)
    del images_list
    anns_np = np.array(anns_list, dtype=np.float32)
    anns_np = anns_np[:, :, :, np.newaxis]
    del anns_list

    # # 形状確認
    # print('images_shape:', images_np.shape)
    # print('anns_shape:', anns_np.shape)

    # 0~1に正規化
    images_np /= 255
    anns_np /= 255

    # Pytorchに合わせて軸の入れ替え & メモリ解放
    transposed_images_np = np.transpose(images_np, (0, 3, 1, 2))
    del images_np
    transposed_anns_np = np.transpose(anns_np, (0, 3, 1, 2))
    del anns_np

    # print('transposed_images_shape:', transposed_images_np.shape)
    # print('transposed_anns_shape:', transposed_anns_np.shape)

    # # 10番目の画像確認
    # cv2.imshow('color', val_images_np[10])
    # cv2.waitKey(0)
    # cv2.imshow('annotation', val_anns_np[10])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return transposed_images_np, transposed_anns_np


if __name__ == '__main__':
    image, anns = ImageLoaderFromCOCO(20, 2, annFiles=annFiles, ImageDir=ImageDir)
