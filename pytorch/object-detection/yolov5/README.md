## Yolov5 pytorch
* https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/
* Again uses training script, but it is very clear what parameters we are using. Source https://github.com/ultralytics/yolov5
* Quickstart guide for [training on GCP VM](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) - 2 dollars/hour
* Optionally can edit the architecture
* Much nicer training and evaluation process than tensorflow obj det, also viz tensorboard in same notebook rather than seperate page
* During training, you want to be watching the `mAP@0.5` to see how your detector is performing
* Can plot multiple runs by renaming folders in `yolov5/runs`

## Experiment 1
* Dataset: `FireNET  2020-07-30 7:23am`
* Use notebook training defaults: `--img 416 --batch 16 --epochs 100`
* Tesla K80, train time: 19min 31s
* `mAP@.5` of 0.628, Precision of 0.299, recall of 0.775 - clearly room for improvement but have all the metrics to improve here
* All metrics were improving so we can train for more epochs
* Exported `best.pt` which [could be deployed to a jetson](https://blog.roboflow.com/deploy-yolov5-to-jetson-nx/)

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/pytorch/object-detection/yolov5/experiment1/metrics-expt1.png" width="700">
</p>

## Experiment 2
* As expt1 but 500 epochs
* Tesla P100-PCIE-16GB, train time: 33min 44s
* `mAP@.5` of 0.588, Precision of 0.569, recall of 0.641 - vs expt1 precision has improved but recall and mAP have fallen. Appears mAP has plateaued whilst recall is falling. Probably overfitting after 150 epochs

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/pytorch/object-detection/yolov5/experiment1/metrics-expt2.png" width="700">
</p>

## Experiment 3
* As expt2 but double batch size: `--batch 32 --epochs 500`
*  `mAP@.5` of 0.633, Precision of 0.594, recall of 0.7 - vs expt2 all metrics have improved, larger batch size has paid off slightly, but comparison of graphs indicate it is a very small improvement

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/pytorch/object-detection/yolov5/experiment1/metrics-expt3.png" width="700">
</p>

## Experiment 4
* As expt3 but double epochs: `--batch 32 --epochs 1000`
* `mAP@.5` of 0.607, Precision of 0.63, recall of 0.63 - vs expt3 precision has improved but other metrics are worse, clear overfitting. Optimum epochs around 150

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/pytorch/object-detection/yolov5/experiment1/metrics-expt4.png" width="700">
</p>
