## Yolov5 
* https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/
* Again uses training script, but it is very clear what parameters we are using
* Optionally can edit the architecture
* Much nicer training and evaluation process than tensorflow obj det, also viz tensorboard in same notebook rather than seperate page
* During training, you want to be watching the `mAP@0.5` to see how your detector is performing

## Experiment 1
* Dataset: `FireNET  2020-07-30 7:23am`
* Use notebook training defaults: `--img 416 --batch 16 --epochs 100`
* Tesla K80, train time: 19min 31s
* `mAP@.5` of 0.628, Precision of 0.299, recall of 0.775 - clearly room for improvement but have all the metrics to improve here