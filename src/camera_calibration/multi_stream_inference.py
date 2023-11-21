import pyrealsense2 as rs
import numpy as np
import cv2
import logging
from PIL import Image

import argparse

from detect_and_segment import load_owlvit, process_video_frame

from FastSAM.fastsam import FastSAM 

import time

parser = argparse.ArgumentParser("OWL-ViT Segment Anything", add_help=True)

# parser.add_argument("--video_path", "-v", type=str, required=True, help="path to video file")
# parser.add_argument("--view", type=str, required=True, help="view")
parser.add_argument("--text_prompt", "-t", type=str, default="blue tartar bottle", help="text prompt")
parser.add_argument(
    "--output_dir", "-o", type=str, default="outputs", help="output directory"
)
parser.add_argument('--owlvit_model', help='select model', default="owlvit-base-patch32", choices=["owlvit-base-patch32", "owlvit-base-patch16", "owlvit-large-patch14"])
parser.add_argument("--box_threshold", type=float, default=0.05, help="box threshold")
parser.add_argument('--get_topk', help='detect topk boxes per class or not', action="store_true")
parser.add_argument('--device', help='select device', default="cuda:0", type=str)
args = parser.parse_args()

output_dir = args.output_dir
# box_threshold = args.box_threshold
# if args.get_topk:
#     box_threshold = 0.0
args.box_threshold = 0.0
args.get_topk = True
text_prompt = args.text_prompt
texts = [text_prompt.split(",")]
# load OWL-ViT model
model, processor = load_owlvit(checkpoint_path=args.owlvit_model, device=args.device)

color_map = {
    "ladle": tuple(np.random.randint(150, 255, size=3).tolist()),
    "ketchup": tuple(np.random.randint(150, 255, size=3).tolist()),
    "tartar": tuple(np.random.randint(150, 255, size=3).tolist()),
    "blue tartar bottle": tuple(np.random.randint(150, 255, size=3).tolist()),
    "pot": tuple(np.random.randint(150, 255, size=3).tolist()),
    "black pot": tuple(np.random.randint(150, 255, size=3).tolist())
}

model_SAM = FastSAM('./FastSAM/weights/yolov8n-seg.pt')


# Configure depth and color streams...
# ...from Camera 1
ctx = rs.context()
devices = ctx.query_devices()
print(devices[0])
print(devices[1])
# quit()
FPS = 30
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('244222077007')
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)
# ...from Camera 2
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device('244622072611')
config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)
# 244622072611
# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)

try:
    while True:

        # Camera 1
        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_1.wait_for_frames()
        depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()
        if not depth_frame_1 or not color_frame_1:
            continue
        # Convert images to numpy arrays
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.03), cv2.COLORMAP_JET)

        # Camera 2
        # Wait for a coherent pair of frames: depth and color
        frames_2 = pipeline_2.wait_for_frames()
        depth_frame_2 = frames_2.get_depth_frame()
        color_frame_2 = frames_2.get_color_frame()
        if not depth_frame_2 or not color_frame_2:
            continue
        # Convert images to numpy arrays
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap_2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_2, alpha=0.03), cv2.COLORMAP_JET)
    
        
        converted_img = cv2.cvtColor(cv2.flip(cv2.flip(color_image_2, 0), 1), cv2.COLOR_BGR2RGB) 
        pil_image = Image.fromarray(converted_img) 
        t = time.time()
        result = process_video_frame(model, processor, texts, pil_image, args, color_map, model_SAM)
        print(time.time()-t)
        result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        # color_image_1 = get_bounding_box(pil_image, args, model, processor, texts)
        # Stack all images horizontally
        flipped_images = [cv2.flip(image, 0) for image in [color_image_1, depth_colormap_1,color_image_2, depth_colormap_2]]
        flipped_images = [cv2.flip(image, 1) for image in flipped_images]
        images1 = np.hstack((flipped_images[:2]))
        images2 =np.hstack((flipped_images[2:]))
        images = result


        # cur_image = images[0,0]
        

        # Show images from both cameras
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

        # Save images and depth maps from both cameras by pressing 's'
        ch = cv2.waitKey(25)
        if ch==115:
            cv2.imwrite("my_image_1.jpg",color_image_1)
            cv2.imwrite("my_depth_1.jpg",depth_colormap_1)
            cv2.imwrite("my_image_2.jpg",color_image_2)
            cv2.imwrite("my_depth_2.jpg",depth_colormap_2)
            print("Save")


finally:

    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()