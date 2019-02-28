# このプログラムは、ImageGenerator.pyにて保存した人物のセグメンテーション結果の画像に
# 対応する生画像（JPEG形式）をディレクトリから探して保存するものである。

import numpy as np
import glob
import os
from PIL import Image

# ディレクトリの設定
load_dir = 'Data/VOCdevkit/VOC2012/JPEGImages/'  # JPEG画像（選別対象）のディレクトリ
browse_dir = 'Data/VOCdevkit/VOC2012/ConvertedSegmentationClass/'  # セグメンテーション後の画像が保存されているディレクトリ
save_dir = 'Data/VOCdevkit/VOC2012/CorrespondingImages/'  # 保存先のディレクトリ

def main():
    # ディレクトリ内のセグメンテーション画像の名前を取得
    files = glob.glob(browse_dir + '*')
    # すべてのファイルに対し動作を行う
    for fname in files:
        # セグメンテーション画像は拡張子が異なるので、ファイル名のみを取得
        jpeg_fname, ext = os.path.splitext(os.path.basename(fname))
        # ファイル名からパスの文字列を生成し、画像を読み込む
        img = Image.open(load_dir + jpeg_fname + ".jpg")
        # 同様に保存先のパスを生成し、画像を保存
        img.save(save_dir + jpeg_fname + '.png')
        # 確認用
        print(os.path.basename(fname))

    return 0

if __name__ == '__main__':
    main()
