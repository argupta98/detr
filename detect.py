from PIL import Image
import requests
import matplotlib.pyplot as plt
import cv2

import torch
from torch import nn
from torchvision.models import resnet50
from models import build
import torchvision.transforms as T
from main import get_args_parser
import numpy as np

torch.set_grad_enabled(False)

parser = get_args_parser()

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model, transform, postprocessors, im_id):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    # Get segmentation and panoptic values 
    out = {}
    out["bboxes"] = bboxes_scaled
    targets = {}
    targets["size"] = torch.tensor(img.shape[2:][::-1])
    targets["orig_size"] = torch.tensor(im.size[::-1])
    # hack to get id tensor
    targets["image_id"] = torch.tensor(im_id)
    targets = [targets]
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    results = postprocessors['bbox'](outputs, orig_target_sizes)
    if 'segm' in postprocessors.keys():
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        out["segmentation"] = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
    if 'panoptic' in postprocessors.keys():
        out["panoptic"] = postprocessors["panoptic"](outputs, targets)

    return probas[keep], out

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        # text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, "detection", fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig('detection.png')
    plt.show()

def make_seg_img(out):
    masks = out["segmentation"][0]["masks"]
    labels = out["segmentation"][0]["labels"]
    scores = out["segmentation"][0]["scores"]
    # import IPython; IPython.embed()
    _, h, w = masks[0].shape
    image = np.zeros((h, w, 3))
    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]] * 100
    for idx in range(len(labels)):
        if scores[idx] < 0.7: continue
        color_mask = np.concatenate([(COLORS[labels[idx]][0] * masks[idx]) * 255, 
                                     (COLORS[labels[idx]][1] * masks[idx]) * 255,
                                     (COLORS[labels[idx]][2] * masks[idx]) * 255], axis=0)
        color_mask = np.transpose(color_mask, (1, 2, 0))
        image += color_mask
    print("image: max {} min {}".format(image.max(), image.min()))
    return image.astype(np.uint8)

def visualize_segmentation(out, im):
    pano_img = np.array(out["panoptic"][1][0]).astype(float)
    pano_img *= 255. / pano_img.max()
    og_img = np.array(im)
    pano_img = pano_img.astype(np.uint8)
    seg_img = make_seg_img(out)
    # seg_img = (og_img * 0.1 + seg_img * 0.9).astype(np.uint8)
    output = np.hstack([og_img, pano_img, seg_img])
    cv2.imshow("seg_img", output[:, :, ::-1])
    cv2.waitKey(0)
    
if __name__== "__main__":
    args = parser.parse_args()
    detr, criterion, postprocessors = build(args)
    state_dict = torch.hub.load_state_dict_from_url(
        url='https://dl.fbaipublicfiles.com/detr/detr-r50-panoptic-00ce5173.pth',
        map_location='cpu', check_hash=True)
    detr.load_state_dict(state_dict["model"])
    detr.eval()

    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    path = requests.get(url, stream=True).raw

    im = Image.open(path)

    scores, out = detect(im, detr, transform, postprocessors, im_id=39769)
    plot_results(im, scores, out["bboxes"])

    # get the segmentation image for visualization
    visualize_segmentation(out, im)