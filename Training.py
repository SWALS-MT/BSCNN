import sys

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data

# The Python file in same directory
from BSCNN_ImageLoader_from_COCO import ImageLoaderFromCOCO
from BSCNN_Model import Net

## VERSION ##
# Python
print('Python: ', sys.version)

# pytorch
print('Pytorch: ', torch.__version__)
print('torchvision: ', torchvision.__version__)

global_step = 0

def train():

    ## initialize ##
    # ハイパーパラメータ
    epoch = 50
    miniblock_num = 20
    learning_rate = 1e-4
    minibatch_size = 100
    # modelの設定
    device = 'cuda'
    model = Net()
    model = model.to(device)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 損失関数
    criterion = nn.NLLLoss2d()
    # 学習率のスケジューラー
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # Epoch数分同じデータを学習する。
    for epoch_num in range(epoch):

        model.train()
        scheduler.step()
        running_loss = 0.0
        print('Epoch Number: ' + str(epoch_num + 1) + ' / ' + str(epoch))
        print('lr: ', optimizer.param_groups[0]['lr'])

        # データを少量ずつ取得し学習を行う。(メモリ削減のため)
        for miniblock_count in range(miniblock_num):

            # データ取得(少量ずつ)
            images, anns = ImageLoaderFromCOCO(miniblock_num, miniblock_count)
            # Pytorch用データセットの作成
            images = torch.from_numpy(images)
            anns = torch.from_numpy(anns)
            dataset_train = \
                data.TensorDataset(images, anns)
            dataloader_train = \
                data.DataLoader(dataset=dataset_train, batch_size=minibatch_size, shuffle=True)

            # ミニバッチ学習
            steps = len(dataset_train)//minibatch_size
            for step, (images, anns) in enumerate(dataloader_train, 1):
                # 合計のステップ数（Global Step）を保存
                global global_step
                global_step += 1

                # データをGPUに対応させる
                images, anns = images.to(device), anns.to(device)

                # Optimizerの初期化
                optimizer.zero_grad()

                # 出力と誤差計算、最適化によるパラメータ更新
                output = model(images)
                loss = criterion(output, )
                loss.backward()
                optimizer.step()

                # 進行度合いの表示
                running_loss += loss.data[0]
                if step % 10 == 9: # print every 9 mini-batches
                    print('[' + str(epoch + 1) + ', ' + str(step + 1) + '] loss:'
                          + str(float(running_loss / 10)))
                    running_loss = 0.0

    # modelの保存
    torch.save(model.state_dict(), 'weight.pth')


if __name__ == '__main__':
    train()