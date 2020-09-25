## tflite object detection
* https://blog.roboflow.com/how-to-train-a-tensorflow-lite-object-detection-model/ use this notebook
* Uses tfrecord and currently tf1 obj det api. One issue is that tf2 is not yet supported but is required for future compatibility
* Tensorboard output (need to learn how to use)
* Relies on training scripts, which are tedious to use and debug
* Reported metrics are loss & DetectionBoxes_Precision/mAP
* **Summary: using these scripts is painful and feels overcomplicated. Prefer to try yolo/pytorch**

### Summary of steps
* Install TensorFlow object detection library and dependencies
* Import dataset from Roboflow in TFRecord format
* Write custom model configuration
* Start custom TensorFlow object detection training job
* Export frozen inference graph in .pb format
* Make inferences on test images to make sure our detector is functioning
* Converting `.pb` model to `.tflite` with the [command line converter](https://www.tensorflow.org/lite/convert)
* Deploy model to device, check out the official TensorFlow Lite Android Demo, iOS Demo, or Raspberry Pi Demo.

### Experiment 1: 25 Sept 2020
Dataset 2020-07-30 7:23am, 482 images, no augmentations, default training parameters

Completed after 35 mins with Loss for final step: 4.4343796:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.113
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.304
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.077
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.047
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.212
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.151
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.221
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.271
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.074
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.506
 ```