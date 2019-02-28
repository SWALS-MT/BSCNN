# このプログラムでは、人物のセグメンテーション結果が含まれる画像と、
# それと同数の人物が存在しない画像を抽出し、出力形式の14x14に
# リサイズするものである（リサイズ方法はNearestらしい）

import numpy as np
import glob
import os

import PIL
from PIL import Image

# バージョン確認
print('PIL VERSION: ', PIL.__version__)

# ディレクトリの設定
load_dir = 'Data/VOCdevkit/VOC2012/SegmentationClass/'
save_dir = 'Data/VOCdevkit/VOC2012/ConvertedSegmentationClass/'

def main():
    # ラベル設定
    class Label:
        GROUND = 0
        AEROPLANE = 1
        BICYCLE = 2
        BIRD = 3
        BOAT = 4
        BOTTLE = 5
        BUS = 6
        CAR = 7
        CAT = 8
        CHAIR = 9
        COW = 10
        DINING_TABLE = 11
        DOG = 12
        HORSE = 13
        MOTORBIKE = 14
        PERSON = 15
        POTTED_PLANT = 16
        SHEEP = 17
        SOFA = 18
        TRAIN = 19
        TV = 20
        VOID = 255

    # フォルダ内の画像の名前の読み込み
    files = glob.glob(load_dir + '*')

    # PERSON以外の保存枚数を設定するための変数
    person_count = 0
    other_count = 0

    for fname in files:
        print(os.path.basename(fname))
        # PIL で読み込む。
        pil_index_img = Image.open(fname)
        # numpy 配列に変換する。
        img = np.asarray(pil_index_img)
        # # 形状確認
        # print(img.shape)
        # # 画像行列確認
        # print(img)

        # # PERSONラベルのみ表示
        # plt.imshow(np.where(img == Label.PERSON, 100, 0))
        # plt.axis('off')
        # plt.show()

        # PERSONラベルのみを残す
        only_person_image = np.where(img == Label.PERSON, 255, 0)

        if 1 <= np.sum(only_person_image > 0):
            print('save person', person_count)
            # PILに変換
            only_person_image_PIL = Image.fromarray(np.uint8(only_person_image))
            # ネットワーク出力形状に変換(14x14)
            resized_img = only_person_image_PIL.resize((14, 14), Image.LANCZOS)
            # numpy 配列に変換
            resized_img_np = np.asarray(resized_img)
            # 画像として保存
            resized_img.save(save_dir + os.path.basename(fname))
            # # 形状確認
            # print(resized_img_np.shape)
            # # 画像行列確認
            # print(resized_img_np)
            # # 表示
            # plt.imshow(resized_img_np)
            # plt.axis('off')
            # plt.show()
            person_count += 1

        else:
            if other_count < person_count:
                print('save other', other_count)
                # PILに変換
                only_person_image_PIL = Image.fromarray(np.uint8(only_person_image))
                # ネットワーク出力形状に変換(14x14)
                resized_img = only_person_image_PIL.resize((14, 14), Image.LANCZOS)
                # numpy 配列に変換
                resized_img_np = np.asarray(resized_img)
                # 画像として保存
                resized_img.save(save_dir + os.path.basename(fname))

                other_count += 1

    return 0

if __name__ == '__main__':
    main()
