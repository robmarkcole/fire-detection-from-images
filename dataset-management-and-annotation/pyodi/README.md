## pyodi
* https://github.com/Gradiant/pyodi
* Requires annotations in COCO format, I exported these from Roboflow and placed at `/Users/robin/Documents/datasets/fireNET_coco/`

## Usage
* `python3 -m venv venv`
* `source venv/bin/activate`
* `pip3 install pyodi`

Then commands to use pyodi:

## 1. paint-annotations
Uses the annotation file to annotate images:

```
pyodi paint-annotations \
  /Users/robin/Documents/datasets/fireNET_coco/valid/_annotations.coco.json \
  /Users/robin/Documents/datasets/fireNET_coco/valid \
  /Users/robin/Documents/datasets/fireNET_coco/valid_annotated
```

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/dataset-management-and-annotation/pyodi/annotated_result.jpg" width="800">
</p>

### 2. ground-truth
This script can be used to explore the images and bounding boxes that compose an object detection dataset. The shape distribution of the images and bounding boxes and their locations are the key aspects to take in account when setting your training configuration.

```
pyodi ground-truth /Users/robin/Documents/datasets/fireNET_coco/train/_annotations.coco.json
```

* Creates 3 plots with plot.ly, bounding box shapes and centers and image shapes.
* The bounding boxes are well centered and tend towards smalle boxes, around 20% of the width and height of an image
* The average image shape is approx 280 wide by 180 high
* Note that Roboflow offers similar inspection in `Dataset Health Check`

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/dataset-management-and-annotation/pyodi/image_shapes.png" width="1200">
</p>

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/dataset-management-and-annotation/pyodi/bounding_box_shapes.png" width="1200">
</p>

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/dataset-management-and-annotation/pyodi/bounding_box_centers.png" width="800">
</p>

### 3. train-config
```
pyodi train-config generation \
  /Users/robin/Documents/datasets/fireNET_coco/train/_annotations.coco.json \
  --input-size [280,180] \
  --n-ratios 3 --n-scales 3
```

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/dataset-management-and-annotation/pyodi/anchor_generation.png" width="1200">
</p>

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/dataset-management-and-annotation/pyodi/config_evaluation.png" width="1200">
</p>