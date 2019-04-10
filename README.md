# BSCNN
● セグメンテーション画像を大雑把に捉えることで特徴マップを高速に抽出しようとしている。 

# BSCNN_ImageLoader_from_COCO.py
● MicroSoft COCOのデータセットからデータの読み込みを行う。（ラベルはPersonに設定してある）  
 （Torchvisionで取得した場合、アノテーション情報の14x14への変換を行うことができないため）  
● 返り値1で画像のNumpy行列、返り値2でアノテーションのNumpy行列を得る。（4次元：枚数xチャンネル数x横幅x縦幅）  
● 入力にアノテーションデータと生画像のディレクトリを格納する変数を追加。

# BSCNN_Model.py
● CNNのモデルをここに設定。  
● ネットワーク構造はVGG16に類似しているが、出力がクラス分類ではなく特徴マップであるため、出力層チャンネル数は1  
● Batch Normalizationを各ブロックごとに実施

# BSCNN_CreateModel.py
● 学習を行うことができる。  
● 学習後はLossグラフ表示、Lossのnpzファイルとモデルの保存も行う。（check.pyにて再度グラフ表示可能）  
● TrainデータとValidationデータに対するLossとAccuracyを求めるパートを追加。
