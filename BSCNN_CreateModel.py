import sys
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data

# The Python file in the same directory
from BSCNN_ImageLoader_from_COCO import ImageLoaderFromCOCO
from BSCNN_Model import Net

## VERSION ##
# Python
print('Python: ', sys.version)

# pytorch
print('Pytorch: ', torch.__version__)
print('torchvision: ', torchvision.__version__)

##__Initialize__##
# Hyperparameters
epochs = 100
minibatch_size = 10
learning_rate = 0.001
miniblocks = 20
global_step = 0
# modelの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
model = model.to(device)
print(model)
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 損失関数
criterion = nn.BCELoss()
# 学習率のスケジューラー
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
# グラフ作成用配列
train_loss_list = []
train_acc_list = []
val_loss_list =[]
val_acc_list = []
# データをランダムに取り出すためのList(0~miniblock_numまで)を作成
number_list = list(range(miniblocks))
# Datasetディレクトリ
data_dir = '../dataset/coco/'
data_type_train = 'train2014'
annFiles_train = data_dir + '2014/annotations/instances_' + data_type_train + '.json'
ImageDir_train = data_dir + '2014/trainimages/'

data_type_val = 'val2014'
annFiles_val = data_dir + '2014/annotations/instances_' + data_type_val + '.json'
ImageDir_val = data_dir + '2014/valimages/'


##__学習用関数（Epochごとに呼び出し）__##
def train():
    ##__Initialize__##
    model.train()
    runnning_loss = 0.0
    correct = 0
    data_count = 0
    # データの読み込む順番をランダムに変更
    random.shuffle(number_list)
    # データを少量ずつ取得し学習を行う。(メモリ削減のため)
    for miniblock_count in range(miniblocks):

        # 読み込むデータを設定
        miniblock_num = number_list[miniblock_count]
        # データ取得(少量ずつ)
        print('Loading Images now... ( ' + str(miniblock_count + 1) +
              ' / ' + str(miniblocks) + ' )')
        images, anns = ImageLoaderFromCOCO(miniblocks, miniblock_num, annFiles=annFiles_train, ImageDir=ImageDir_train)
        # Pytorch用データセットの作成
        images = torch.from_numpy(images)
        anns = torch.from_numpy(anns)
        dataset_train = \
            data.TensorDataset(images, anns)
        dataloader_train = \
            data.DataLoader(dataset=dataset_train, batch_size=minibatch_size, shuffle=True)

        # ミニバッチ学習
        for step, (images, anns) in enumerate(dataloader_train, 1):
            # 合計のステップ数（Global Step）を保存
            global global_step
            global_step += 1
            if global_step == 1:
                print('Train Image Size:', images.size())
                print('Output Image Size:', anns.size())

            # データをGPUに対応させる
            images, anns = images.to(device), anns.to(device)

            # Optimizerの初期化
            optimizer.zero_grad()

            # 出力と誤差計算、最適化によるパラメータ更新
            output = model(images)
            loss = criterion(output, anns)
            loss.backward()
            optimizer.step()

            # 誤差の保存
            runnning_loss += loss.item()
            # Accuracyの導出
            subtrain_np = output.to('cpu').detach().numpy() - anns.to('cpu').detach().numpy()
            subtrain_sum = np.sum((subtrain_np < 0.05) & (subtrain_np > -0.05))
            subtrain_max = subtrain_np.size
            correct += subtrain_sum / subtrain_max
            data_count += 1

    train_loss = runnning_loss / data_count
    train_acc = correct / data_count

    return train_loss, train_acc

def eval():
    model.eval()
    running_loss = 0.0
    correct = 0.0
    data_count = 0
    # Train Stop
    with torch.no_grad():
        # データの読み込む順番をランダムに変更
        random.shuffle(number_list)
        # データを少量ずつ取得し学習を行う。(メモリ削減のため)
        for miniblock_count in range(miniblocks):
            # 読み込むデータを設定
            miniblock_num = number_list[miniblock_count]
            # データ取得(少量ずつ)
            print('Loading Images now... ( ' + str(miniblock_count + 1) +
                  ' / ' + str(miniblocks) + ' )')
            images, anns = ImageLoaderFromCOCO(miniblocks, miniblock_num, annFiles=annFiles_val, ImageDir=ImageDir_val)
            # Pytorch用データセットの作成
            images = torch.from_numpy(images)
            anns = torch.from_numpy(anns)
            dataset_val = \
                data.TensorDataset(images, anns)
            dataloader_val = \
                data.DataLoader(dataset=dataset_val, batch_size=minibatch_size, shuffle=True)

            for step, (images, anns) in enumerate(dataloader_val, 1):
                images = images.to(device)
                anns = anns.to(device)

                output = model(images)
                loss = criterion(output, anns)
                running_loss += loss.item()
                # Accuracyの導出
                subval_np = output.to('cpu').detach().numpy() - anns.to('cpu').detach().numpy()
                subval_sum = np.sum((subval_np < 0.05) & (subval_np > -0.05))
                subval_max = subval_np.size
                correct += subval_sum / subval_max
                data_count += 1

    val_loss = running_loss / data_count
    val_acc = correct / data_count

    return val_loss, val_acc

if __name__ == '__main__':
    # エポック数だけ学習（main loop）
    for epoch in range(epochs):
        train_loss, train_acc = train()
        val_loss, val_acc = eval()
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f' % (epoch + 1, train_loss, val_loss, val_acc))

    # modelとグラフの保存
    np.savez('train_loss_acc_backup.npz', loss=np.array(train_loss_list), acc=np.array(train_acc_list))
    np.savez('val_loss_acc_backup.npz', loss=np.array(val_loss_list), acc=np.array(val_acc_list))
    torch.save(model.state_dict(), 'BSCNN_COCO.pth')
