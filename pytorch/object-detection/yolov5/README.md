## Yolov5 pytorch
* https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/
* Again uses training script, but it is very clear what parameters we are using. Source https://github.com/ultralytics/yolov5
* Quickstart guide for [training on GCP VM](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) - 2 dollars/hour
* Optionally can edit the architecture
* Much nicer training and evaluation process than tensorflow obj det, also viz tensorboard in same notebook rather than seperate page
* During training, you want to be watching the `mAP@0.5` to see how your detector is performing
* Can plot multiple runs by renaming folders in `yolov5/runs`
* Note on metrics reported below: the mAP etc reported are for the final batch, and subject to noise
* Relevant [repo](https://github.com/jshaffer94247/Counting-Fish) and [article](https://blog.roboflow.com/using-computer-vision-to-count-fish-populations/) on counting fish with yolov5

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
* `mAP@.5` of 0.588, Precision of 0.569, recall of 0.641 - vs expt1 precision has improved but recall and mAP have fallen. Appears mAP has plateaued whilst recall is falling.

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
* `mAP@.5` of 0.607, Precision of 0.63, recall of 0.63 - vs expt3 precision has improved but other metrics are worse. Optimum epochs around 150

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/pytorch/object-detection/yolov5/experiment1/metrics-expt4.png" width="700">
</p>

## Experiment 5
* Increase batch and reduce epochs: `--batch 64 --epochs 150`
* Slow training and poor results

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/pytorch/object-detection/yolov5/experiment1/metrics-expt5.png" width="700">
</p>

## Experiment 6
* Try small batch size and limited epochs: `--batch 8 --epochs 200`
* `mAP@.5` of 0.617, Precision of 0.4, recall of 0.768. Very comparable to expt1 & 2.

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/pytorch/object-detection/yolov5/experiment1/metrics-expt6.png" width="700">
</p>

## Summary of experiments 1 - 6
Increasing batch size results in slower training, and too many epochs may result in overfitting (hard to be certain). We really want to use a dedicated tool to optimise these parameters, but in general the upper limit of `mAP@.5` is approx 0.63. Worth looking at inference results to see if there are any obvious trends - e.g. poor detection on candle flames? Currently not using the test set. Next experiment with image augmentation.

## Experiment 7
* Apply -25% to + 25% brightness
* Add 5% of pixels noise
* Apply grayscale to 25% images
* Applying these augmentations generates dataset `FireNET  2020-09-27 8:09am`
* Use notebook training defaults: `--img 416 --batch 16 --epochs 100`
* `mAP@.5` of 0.622, Precision of 0.533, recall of 0.688. Comparable results to experiment 1 but with improved precision traded for recall and mAP, indicating these augmentation steps did change performance

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/pytorch/object-detection/yolov5/experiment1/dataset-expt7.png" width="1100">
</p>

## Experiment 8
* As expt7 but double epochs to 200
* `mAP@.5` of 0.657, Precision of 0.6, recall of 0.7. mAP is slowly improving but recall slowly degrading.

## Experiment 9
* As expt8 but increase epochs to 1000
* 3 hrs
* `mAP@.5` of 0.593, Precision of 0.728, recall of 0.58. Again more epochs improve precision but otherwise degrade the model.

## Final summary
In general the upper limit of `mAP@.5` is approx 0.66, and there appears to be a tradeoff between Precision and Recall, possibly due to overfitting. The default training parameters in experiment 1 resulted in a pretty respectable model, but some tweaking of parameters and application of image augmentation did result in a slightly improved model in experiment 8. Further performance improvement could likely be achieved by increasing the size of the training dataset, further experimentation with image augmentation, and a more robust technique for optimising training parameters.