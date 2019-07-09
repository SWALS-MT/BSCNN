import torch
from BSCNN_Model import Net

def TransposeModel(Weight_Path, Model, height=448, width=448, ch=3):
    device = torch.device("cuda")
    Model.to(device)
    Model.load_state_dict(torch.load(Weight_Path))
    # jit へ変換
    traced_net = torch.jit.trace(Model, torch.rand(1, ch, height, width).to(device))
    # 後の保存(Save the transposed Model)
    traced_net.save('BSCNN_h{}_w{}_mode{}_cuda.pt'.format(height, width, ch))
    print('BSCNN_h{}_w{}_mode{}_cuda.pt is exported.'.format(height, width, ch))


if __name__ == '__main__':
    # Set the model path
    Weight_Path = 'BSCNN_COCO.pth'
    # Set the model
    Model = Net()
    TransposeModel(Weight_Path, Model)