## YOLOv4-tiny darknet
* https://blog.roboflow.com/train-yolov4-tiny-on-custom-data-lighting-fast-detection/
* YOLOv4 tiny is roughly 8X as fast at inference time as YOLOv4 and roughly 2/3 as performant. In practice you may see no degradation of performance.
* The primary difference between YOLOv4 tiny and YOLOv4 is that the network size is dramatically reduced. The number of convolutional layers in the CSP backbone are compressed. The number of YOLO layers are two instead of three and there are fewer anchor boxes for prediction.
* We may wish to slightly adjust network architecture based on the number of classes in our custom dataset
* Not pytorch but darknet training scripts, and have to do some compilation which depends on the hardware

## Experiment 1
* Tesla P100
* Get error: `cp: cannot stat 'train/_darknet.labels': No such file or directory` unable to proceed