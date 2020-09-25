## tflite object detection
* https://blog.roboflow.com/how-to-train-a-tensorflow-lite-object-detection-model/
* Uses tfrecord and currently tf1 obj det api. One issue is that tf2 is not yet supported but is required for future compatibility
* Tensorboard output
* Relies on training scripts

### Summary of steps
* Install TensorFlow object detection library and dependencies
* Import dataset from Roboflow in TFRecord format
* Write custom model configuration
* Start custom TensorFlow object detection training job
* Export frozen inference graph in .pb format
* Make inferences on test images to make sure our detector is functioning
* Converting `.pb` model to `.tflite` with the command line converter
* Deploy model to device, check out the official TensorFlow Lite Android Demo, iOS Demo, or Raspberry Pi Demo.