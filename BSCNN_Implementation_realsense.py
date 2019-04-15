import pyrealsense2 as rs
import numpy as np
import torch
import cv2
import time

from BSCNN_Model import Net

model_path = 'BSCNN_COCO.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def implementation_realsense():
    ##__Initialize RealSense__##

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    ##__Initialize BSCNN__##

    # Loading the model
    print('Loading The Model...')
    BSCNN_model = Net()
    print(BSCNN_model)
    BSCNN_model.load_state_dict(torch.load(model_path))
    BSCNN_model.to(device)
    print('Finished Reading')

    ##__Streaming Loop__##

    try:
        while True:
            start_time = time.time()
            ## RealSense Part ##
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            color_image = np.asanyarray(color_frame.get_data())

            ## BSCNN Part ##
            frame = cv2.resize(color_image, (448, 448))
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
            out_reshaped /= out_reshaped.max() / 255.0
            out_reshaped = cv2.resize(out_reshaped, (448, 448), interpolation=cv2.INTER_NEAREST)

            cv2.imshow('Original', color_image)
            cv2.imshow('Output', out_reshaped)
            key = cv2.waitKey(1)
            if key == 27:
                break
            print("Time: {:.2f} s / img".format(time.time() - start_time))

    finally:
        pipeline.stop()


if __name__ == '__main__':
    implementation_realsense()
