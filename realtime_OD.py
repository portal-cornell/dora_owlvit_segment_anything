import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# OwlViT Detection
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
# FastSAM
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from ultralytics.models.fastsam import FastSAMPredictor, FastSAMPrompt, FastSAMValidator

import gc

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
    print(bounding_box)

    # Print detected objects and rescaled box coordinates
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

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
    while video.isOpened():
        frame_is_read, frame = video.read()
        if not frame_is_read:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # load image & texts
        image = Image.fromarray(frame)
        
        # run object detection model
        with torch.no_grad():
            inputs = processor(text=texts, images=image, return_tensors="pt").to(args.device)
            outputs = model(**inputs)
        
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = processor.post_process_object_detection(outputs=outputs, threshold=box_threshold, target_sizes=target_sizes.to(args.device))
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
            # topk_labels = results[i]["labels"][topk_idxs]
            # print(topk_labels.shape)
            # print(topk_labels)
            # import pdb; pdb.set_trace()
            topk_labels = torch.tensor(list(range(4)), device=args.device).long()
            boxes, scores, labels = topk_boxes, topk_scores, topk_labels
        else:
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        

        # Print detected objects and rescaled box coordinates
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

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
        gif.append(image_with_box)

        # release the OWL-ViT
        

        # run FastSAM
        # predictor = FastSAMPredictor({'checkpoint':"./sam_vit_h_4b8939.pth"})
        # image = cv2.imread(args.image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # predictor.set_image(image)
        
        # H, W = size[1], size[0]

        # for i in range(boxes.shape[0]):
        #     boxes[i] = torch.Tensor(boxes[i])

        # boxes = torch.tensor(boxes, device=predictor.device)

        # transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
        
        # masks, _, _ = predictor.predict_torch(
        #     point_coords = None,
        #     point_labels = None,
        #     boxes = transformed_boxes,
        #     multimask_output = False,
        # )
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box in boxes:
        #     show_box(box.numpy(), plt.gca())
        # plt.axis('off')
        # plt.savefig(f"./{output_dir}/right_table_items_owlvit_SAM_output.jpg")

        #defining a inference sources
        source = 'table_items.png'

        #creating FastSAM model
        model = FastSAM('FastSAM-s.pt')

        # Run inference on an image
        everything_results = model(source, device=args.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

        # Prepare a Prompt Process object
        prompt_process = FastSAMPrompt(source, everything_results, device=args.device)

        # # Everything prompt
        # ann = prompt_process.everything_prompt()

        #Bbox
        ann = prompt_process.box_prompt(bbox=get_bounding_box)
        prompt_process.plot(annotations=ann, output='./')


        # run segment anything (SAM)
        # predictor = SamPredictor(build_sam(checkpoint="./sam_vit_h_4b8939.pth"))
        # image = cv2.imread(args.image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # predictor.set_image(image)
        
        # H, W = size[1], size[0]

        # for i in range(boxes.shape[0]):
        #     boxes[i] = torch.Tensor(boxes[i])

        # boxes = torch.tensor(boxes, device=predictor.device)

        # transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
        
        # masks, _, _ = predictor.predict_torch(
        #     point_coords = None,
        #     point_labels = None,
        #     boxes = transformed_boxes,
        #     multimask_output = False,
        # )
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box in boxes:
        #     show_box(box.numpy(), plt.gca())
        # plt.axis('off')
        # plt.savefig(f"./{output_dir}/right_table_items_owlvit_SAM_output.jpg")
        cnt += 1
        # grounded results
        image_pil = Image.fromarray(frame)
        image_with_box = plot_boxes_to_image(image_pil, pred_dict, color_map)[0]
        gif.append(image_with_box)
        # image_with_box.save(os.path.join(f"./{output_dir}/left_table_items_owlvit_box_{cnt}.png"))
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # import pdb; pdb.set_trace()
    newsize = (320, 180)
    gif = [im.resize(newsize) for im in gif]
    gif[0].save(os.path.join(f"./{output_dir}/{args.view}_{args.text_prompt}.gif"), save_all=True,optimize=True, append_images=gif[1:], loop=0)