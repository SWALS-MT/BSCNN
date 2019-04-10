import numpy as np
import cv2
import torch
from BSCNN_Model import Net

model_path = 'BSCNN_COCO.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def implementation():
    print('Loading The Model...')
    BSCNN_model = Net()
    print(BSCNN_model)
    BSCNN_model.load_state_dict(torch.load(model_path))
    BSCNN_model.to(device)
    print('Finished Reading')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        frame = cv2.resize(frame, (448, 448))
        frame_float = np.array(frame, dtype=np.float32)
        frame_float /= 255
        frame_np = frame_float[np.newaxis, :, :, :]

        frame_np_transposed = np.transpose(frame_np, (0, 3, 1, 2))
        # print(frame_np_transposed.shape)

        frame_tensor = torch.from_numpy(frame_np_transposed)
        frame_tensor = frame_tensor.to(device)

        output = BSCNN_model(frame_tensor).to('cpu').detach().numpy()

        out_reshaped = output.reshape([14, 14, 1])
        out_reshaped = np.where(out_reshaped < 0.8, 0, out_reshaped)

        # print(out_reshaped.shape)
        out_reshaped /= out_reshaped.max()/255.0
        out_reshaped = cv2.resize(out_reshaped, (448, 448), interpolation=cv2.INTER_NEAREST)

        cv2.imshow('Original', frame)
        cv2.imshow('Output', out_reshaped)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    implementation()