import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# OwlViT Detection
from transformers import OwlViTProcessor, OwlViTForObjectDetection

import cv2
import numpy as np
import matplotlib.pyplot as plt

from FastSAM.fastsam import FastSAM, FastSAMPrompt 

import gc
import time

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def plot_boxes_to_image(image_pil, tgt, color_map):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    scores = tgt["scores"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)


    # draw boxes and masks
    for box, label, score in zip(boxes, labels, scores):
        # random color
        color = color_map[label]
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        draw.text((x0, y0), str(label) + ':' + str(round(score.item(), 3)), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label) + ':' + str(round(score.item(), 3)), font)
        else:
            w, h = draw.textsize(str(label) + ':' + str(round(score.item(), 3)), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label) + ':' + str(round(score.item(), 3)), fill="black")
        # draw.text((x1, y0), str(score.item()), fill='white')

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def load_owlvit(checkpoint_path="owlvit-large-patch14", device='cpu'):
    """
    Return: model, processor (for text inputs)
    """
    processor = OwlViTProcessor.from_pretrained(f"google/{checkpoint_path}")
    model = OwlViTForObjectDetection.from_pretrained(f"google/{checkpoint_path}")
    print(device)
    model.to(device)
    model.eval()
    
    return model, processor

def get_bounding_box(image, args, model, processor, texts):
    with torch.no_grad():
        inputs = processor(text=texts, images=image, return_tensors="pt").to(args.device)
        outputs = model(**inputs)
    
    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(outputs=outputs, threshold=args.box_threshold, target_sizes=target_sizes.to(args.device))
    scores = torch.sigmoid(outputs.logits)
    print(outputs.logits.shape)
    print(scores.shape)
    topk_scores, topk_idxs = torch.topk(scores, k=1, dim=1)
    
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    print(results[i]["labels"].shape)
    print(topk_scores)
    print(topk_idxs)
    if args.get_topk:    
        topk_idxs = topk_idxs.squeeze(1).tolist()
        topk_boxes = results[i]['boxes'][topk_idxs]
        topk_scores = topk_scores.view(len(text), -1)
        topk_labels = torch.tensor(list(range(4)), device=args.device).long()
        boxes, scores, labels = topk_boxes, topk_scores, topk_labels
    else:
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]


    # Getting the location of the bounding boxes within the video to pass into fastsam
    for result in results:
        boxes = result.boxes
    bounding_box = boxes.xyxy.tolist()[0]
    # print(bounding_box)

    # Print detected objects and rescaled box coordinates
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        # print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

    boxes = boxes.cpu().detach().numpy()
    normalized_boxes = copy.deepcopy(boxes)
    
    # # visualize pred
    size = image.size
    pred_dict = {
        "boxes": normalized_boxes,
        "size": [size[1], size[0]], # H, W
        "labels": [text[idx] for idx in labels],
        "scores": scores
    }

    cnt += 1
    # grounded results
    image_pil = Image.fromarray(frame)
    image_with_box = plot_boxes_to_image(image_pil, pred_dict, color_map)[0]
    return cv2.cvtColor(np.array(image_with_box), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("OWL-ViT Segment Aything", add_help=True)

    parser.add_argument("--video_path", "-v", type=str, required=True, help="path to video file")
    parser.add_argument("--view", type=str, required=True, help="view")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument('--owlvit_model', help='select model', default="owlvit-base-patch32", choices=["owlvit-base-patch32", "owlvit-base-patch16", "owlvit-large-patch14"])
    parser.add_argument("--box_threshold", type=float, default=0.05, help="box threshold")
    parser.add_argument('--get_topk', help='detect topk boxes per class or not', action="store_true")
    parser.add_argument('--device', help='select device', default="cuda:0", type=str)
    args = parser.parse_args()
    

    # cfg
    # checkpoint_path = args.checkpoint_path  # change the path of the model
    # image_path = args.image_path
    
    gif = []
    # make dir
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    if args.get_topk:
        box_threshold = 0.0
    os.makedirs(output_dir, exist_ok=True)
    text_prompt = args.text_prompt
    texts = [text_prompt.split(",")]
    # load OWL-ViT model
    model, processor = load_owlvit(checkpoint_path=args.owlvit_model, device=args.device)

    color_map = {
        "ladle": tuple(np.random.randint(150, 255, size=3).tolist()),
        "ketchup": tuple(np.random.randint(150, 255, size=3).tolist()),
        "tartar": tuple(np.random.randint(150, 255, size=3).tolist()),
        "blue tartar bottle": tuple(np.random.randint(150, 255, size=3).tolist()),
        "tartar bottle": tuple(np.random.randint(150, 255, size=3).tolist()),
        "pot": tuple(np.random.randint(150, 255, size=3).tolist()),
        "black pot": tuple(np.random.randint(150, 255, size=3).tolist())
    }

    import cv2 

    video = cv2.VideoCapture(args.video_path)
    video.set(cv2.CAP_PROP_FPS, 10)
    cnt = 0
    model_SAM = FastSAM('./FastSAM/weights/yolov8n-seg.pt')
    while video.isOpened():
        frame_is_read, frame = video.read()
        if not frame_is_read:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # load image & texts
        image = Image.fromarray(frame)
        # newsize = (320, 180)
        # image = image.resize(newsize, resample=Image.BILINEAR)
        
        # run object detection model
        # each forward pass takes about .06s
        with torch.no_grad():
            inputs = processor(text=texts, images=image, return_tensors="pt").to(args.device)
            outputs = model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = processor.post_process_object_detection(outputs=outputs, threshold=box_threshold, target_sizes=target_sizes.to(args.device))
        scores = torch.sigmoid(outputs.logits)
        topk_scores, topk_idxs = torch.topk(scores, k=1, dim=1)
        
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        if args.get_topk:    
            topk_idxs = topk_idxs.squeeze(1).tolist()
            topk_boxes = results[i]['boxes'][topk_idxs]
            topk_scores = topk_scores.view(len(text), -1)
            topk_labels = torch.tensor(list(range(4)), device=args.device).long()
            boxes, scores, labels = topk_boxes, topk_scores, topk_labels
        else:
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        

        # Print detected objects and rescaled box coordinates
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            # print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

        boxes = boxes.cpu().detach().numpy()
        normalized_boxes = copy.deepcopy(boxes)
        
        # # visualize pred
        size = image.size
        pred_dict = {
            "boxes": normalized_boxes,
            "size": [size[1], size[0]], # H, W
            "labels": [text[idx] for idx in labels],
            "scores": scores
        }

        input = image
        # each forward pass takes about .1s
        everything_results = model_SAM(
            input,
            device=args.device,
            retina_masks=True,
            imgsz=1024,
            conf=.4,
            iou=.9   
            )
        bboxes = boxes
        points = None
        point_label = None
        image_with_box = plot_boxes_to_image(image, pred_dict, color_map)[0]
        prompt_process = FastSAMPrompt(image_with_box, everything_results, device=args.device)
        ann = prompt_process.box_prompt(bboxes=bboxes.tolist())
        result = prompt_process.plot(
            annotations=ann,
            output_path=f'./tmp/{cnt}.png',
            bboxes = bboxes,
            points = points,
            point_label = point_label,
            withContours=False,
            better_quality=True,
        )

        cnt += 1
        # grounded results
        # image_with_box = plot_boxes_to_image(image, pred_dict, color_map)[0]
        # gif.append(image_with_box)
        gif.append(Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)))
        # result.save(os.path.join(f"./tmp/{cnt}.png"))
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    newsize = (320, 180)
    gif = [im.resize(newsize, resample=Image.BILINEAR) for im in gif]
    gif[0].save(os.path.join(f"./{output_dir}/{args.view}_{args.text_prompt}.gif"), save_all=True,optimize=True, append_images=gif[1:], loop=0)