# BSCNN
セグメンテーション画像を大雑把に捉えることで特徴マップを高速に抽出しようとしている。 
![Original_screenshot_10 04 2019](https://user-images.githubusercontent.com/47411597/55931351-0f9bdf00-5c60-11e9-9bf1-73c39991b4bb.png)![Output_screenshot_10 04 2019](https://user-images.githubusercontent.com/47411597/55931356-13c7fc80-5c60-11e9-8b4c-9b45fbcef5cb.png)

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

# Result
MicroSoft COCO2014のデータセットを用いて学習を行った。  
**Loss**  
  10エポックをピークにValidationのLossが増加傾向にある。  
![BSCNN_Loss](https://user-images.githubusercontent.com/47411597/55868230-03137a00-5bbf-11e9-8f3a-087b18ab6bdf.png)  
**Accuracy**  
  計算方法：出力と入力の差分を取り、（誤差5%未満の要素数）／（配列の全要素数）によって求めた。　　
  なぜかLossが下がり始める10エポックを超えてもAccuracyが増加傾向に。  
![BSCNN_Accuracy](https://user-images.githubusercontent.com/47411597/55868224-fc850280-5bbe-11e9-9ab8-01689becf725.png)
