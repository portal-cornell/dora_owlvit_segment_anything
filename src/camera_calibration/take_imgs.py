import pyrealsense2 as rs
import numpy as np
import cv2
import logging
from PIL import Image

import argparse

# from realtime_OD import load_owlvit, get_bounding_box

# parser = argparse.ArgumentParser("OWL-ViT Segment Anything", add_help=True)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# parser.add_argument("--video_path", "-v", type=str, required=True, help="path to video file")
# parser.add_argument("--view", type=str, required=True, help="view")
# parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
# parser.add_argument(
#     "--output_dir", "-o", type=str, default="outputs", help="output directory"
# )
# parser.add_argument('--owlvit_model', help='select model', default="owlvit-base-patch32", choices=["owlvit-base-patch32", "owlvit-base-patch16", "owlvit-large-patch14"])
# parser.add_argument("--box_threshold", type=float, default=0.05, help="box threshold")
# parser.add_argument('--get_topk', help='detect topk boxes per class or not', action="store_true")
# parser.add_argument('--device', help='select device', default="cuda:0", type=str)
# args = parser.parse_args()

# output_dir = args.output_dir
# box_threshold = args.box_threshold
# if args.get_topk:
#     box_threshold = 0.0
# text_prompt = args.text_prompt
# texts = [text_prompt.split(",")]
# # load OWL-ViT model
# model, processor = load_owlvit(checkpoint_path=args.owlvit_model, device=args.device)

# color_map = {
#     "ladle": tuple(np.random.randint(150, 255, size=3).tolist()),
#     "ketchup": tuple(np.random.randint(150, 255, size=3).tolist()),
#     "tartar": tuple(np.random.randint(150, 255, size=3).tolist()),
#     "blue tartar bottle": tuple(np.random.randint(150, 255, size=3).tolist()),
#     "pot": tuple(np.random.randint(150, 255, size=3).tolist()),
#     "black pot": tuple(np.random.randint(150, 255, size=3).tolist())
# }


# Configure depth and color streams...
# ...from Camera 1
ctx = rs.context()
devices = ctx.query_devices()
print(devices[0])
print(devices[1])
# quit()

pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('244622072611')
# config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config_1.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# ...from Camera 2
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device('244222077007')
# config_2.enable_stream(rs.stream.depth, 1920, 1080, rs.format.z16, 30)
config_2.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)


# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)

profile = pipeline_1.get_active_profile()
profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
print(profile)
print(profile.get_intrinsics())

profile = pipeline_2.get_active_profile()
profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
print(profile)
print(profile.get_intrinsics())
count = 0

try:
    while True:

        # Camera 1
        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_1.wait_for_frames()
        # depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()
        # if not depth_frame_1 or not color_frame_1:
        #     continue
        # # Convert images to numpy arrays
        # depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        # # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.03), cv2.COLORMAP_JET)

        # # Camera 2
        # # Wait for a coherent pair of frames: depth and color
        frames_2 = pipeline_2.wait_for_frames()
        # depth_frame_2 = frames_2.get_depth_frame()
        color_frame_2 = frames_2.get_color_frame()
        # if not depth_frame_2 or not color_frame_2:
        #     continue
        # # Convert images to numpy arrays
        # depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())
        # # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap_2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_2, alpha=0.03), cv2.COLORMAP_JET)

        # color_image_1 = cv2.flip(cv2.flip(color_image_1, 0), 1)
        # converted_img = cv2.cvtColor(color_image_1, cv2.COLOR_BGR2RGB) 
        # pil_image = Image.fromarray(converted_img) 
        # # color_image_1 = get_bounding_box(pil_image, args, model, processor, texts)
        # color_image_1 = cv2.flip(cv2.flip(color_image_1, 0), 1)
        # # Stack all images horizontally
        # flipped_images = [cv2.flip(image, 0) for image in [color_image_1, depth_colormap_1,color_image_2, depth_colormap_2]]
        # flipped_images = [cv2.flip(image, 1) for image in flipped_images]
        # images1 = np.hstack((flipped_images[:2]))
        # images2 =np.hstack((flipped_images[2:]))
        # images = np.vstack((images1, images2))

        # cur_image = images[0,0]
        

        # Show images from both cameras
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        # color_image_1 = cv2.flip(cv2.flip(color_image_1, 0), 1)
        cv2.imshow('RealSense', color_image_1)
        cv2.waitKey(1)

        # Save images and depth maps from both cameras by pressing 's'
        # board_size = (9,6)
        ch = cv2.waitKey(25)
        if ch==115:
            print('here')
            # gray = cv2.cvtColor(color_image_1, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # ret, corners = cv2.findChessboardCorners(gray, board_size, None)
            # print(ret)
            cv2.imwrite(f"calibration/right_april_{count}.jpg",color_image_1)
            cv2.imwrite(f"calibration/left_april_{count}.jpg",color_image_2)
            # cv2.imwrite(f"my_depth_1_{count}.jpg",depth_colormap_1)
            # cv2.imwrite(f"my_image_2_{count}.jpg",color_image_2)
            # cv2.imwrite(f"my_depth_2_{count}.jpg",depth_colormap_2)
            print("Save")
            count += 1
            # if ret == True:
            #     # objpoints.append(objp)
            #     corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            #     # imgpoints.append(corners2)
            #     # Draw and display the corners
            #     cv2.drawChessboardCorners(color_image_1, board_size, corners2, ret)
            #     cv2.namedWindow('chess', cv2.WINDOW_NORMAL)
            #     cv2.imshow('chess', color_image_1)
            #     cv2.waitKey(1500)


finally:

    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()