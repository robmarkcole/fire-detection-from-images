## Yolov4 pytorch
* https://models.roboflow.com/object-detection/yolov4-pytorch
* [opendatacam](https://github.com/opendatacam/opendatacam) uses yolov4
* Training script authored by Tianxiaomo

### Hyperparameters
The training script exposes full range of hyperparameters:
* -b batch size (you should keep this low (2-4) for training to work properly)
* -s number of subdivisions in the batch, this was more relevant for the darknet framework
* -l learning rate
* -g direct training to the GPU device
* pretrained invoke the pretrained weights that we downloaded above
* classes - number of classes
* epoch - how long to train for

## Experiment 1
* On running the training notebook  I hit error: `ValueError: invalid literal for int() with base 10: '(5)_jpg.rf.0197636708df2f500f855fc4776bfcdb.jpg'` have emailed support about this