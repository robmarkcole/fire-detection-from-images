# fire-detection-from-images
The purpose of this repo is to demonstrate a fire detection neural net model. In use this model will place a bounding box around any fire in an image.

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/images/fire-annotated.jpg" width="350">
</p>

## Best results
**Object detection:** After experimenting with various model architectures I settled on [Yolov5](https://github.com/ultralytics/yolov5) pytorch model (see `pytorch/object-detection/yolov5/experiment1/best.pt`). After a few hours of experimentation I generated a model of `mAP@.5` of 0.657, Precision of 0.6, Recall of 0.7, trained on 1155 images (337 base images + augmentation).

**Classification:** I have yet to train my own model, but 95% accuracy is reported using ResNet50

**Segmentation:** requires annotation

## Motivation and challenges
Traditional smoke detectors work by [detecting the physical presence of smoke particles](https://www.nfpa.org/Public-Education/Staying-safe/Safety-equipment/Smoke-alarms/Ionization-vs-photoelectric). However they are prone to false detections (e.g. from toasters) and do not localise the fire particularly well. In these situations a camera solution could complement a traditional detector, in order to improve response times or to provide additional metrics such as the size and location of a fire. With the location and nature of the fire identified, an automated intervention may be possible, e.g. via a sprinkler system or drone. Also data can be sent to fire services to provide otherwise non-existent situational awareness. Particular locations I am interested in are: kitchens & living rooms, garages and outbuildings, and areas where fires might already be present but spreading outside a desired zone e.g. fire pit.

There are a couple of significant challenges & open questions:
* For fast edge model what is 'best' architecture? Yolo3 is very popular for commecrial applications and can be implemented in keras or pytorch, baseline [Yolov5](https://github.com/ultralytics/yolov5) as it is currently SOTA and has deployment guide to Jetson. 
* Can the architecture be optimised since we are detecting only a single class?
* Baseline object detection, but is there benefit to classifier or segmentation? Obj models train on mAP and Recall metrics but for our application bounding box accuracy may not be top priority? However classification models work best on a nice shot containing only the target object but in real life fire scenarios the scene will not be as simple as this scenario.
* Tensorflow + google ecosystem or Pytorch + NVIDIA/MS? Tensorflow suffers from tf1 legacy
* Is a single 'super' model preferable, or several specialised models? Typical categories of fire include candle flame, indoor/outdoor, vehicle
* Gathering or locating a comprehensive, representative and balanced training dataset
* Handling different viewpoints, different camera manufacturers and settings, and different ambient lighting conditions.
* Since fires are so bright they can often wash out images and cause other optical disturbances, how can this be compensated for?
* Since we expect the model will have limitations, how do we make the model results interpretable?
* Fires can be a very wide range of sizes, from an candle flame to engulfing an entire forest - is this a small object & large object problem? Splitting dataset by fire class and training models for each class may give better results? Treat as a semantic segmentation problemn (requires reannotating dataset)?

Ideas:
* Preprocessing images, e.g. to remove background or apply filters
* Classifying short sequences of video, since the movement of fire is quite characteristic
* Simulated data, identify any software which can generate realistic fires and add to existing datasets
* Augmentations to simulate effect of different cameras and exposure settings
* Identify any relevant guidance/legislation on required accuracy of fire detection techniques
* Combine RGB + thermal for suppressing false positives? e.g. using https://openmv.io/blogs/news/introducing-the-openmv-cam-pure-thermal or cheaper [grideye](https://shop.pimoroni.com/products/adafruit-amg8833-ir-thermal-camera-breakout) or [melexsis](https://shop.pimoroni.com/products/mlx90640-thermal-camera-breakout?variant=12536948654163)

## Approach & Tooling
* Frames will be fed through neural net. On positive detection of fire metrics are extracted. Ignore smoke for MVP. Try various architectures & parameters to establish a 'good' baseline model.
* Develop a lower accuracy but fast model targeted at RPi and mobile, and a high accuracy model targeted at GPU devices like Jetson. Yolo present both options, yolo4 lite for mobile and yolo5 for GPU. Alternatively there is mobilenet and tf-object-detection-api. Higher accuracy GPU model is priority.
* Use Google Colab for training

## Articles & repos
* [Fire and smoke detection with Keras and Deep Learning by pyimagesearch](https://www.pyimagesearch.com/2019/11/18/fire-and-smoke-detection-with-keras-and-deep-learning/) - dataset collected by scraping Google images (provides link to dataset with  1315 fire images), binary Fire/Non-fire classification with tf2 & keras sequential CNN, achieve 92% accuracy, concludes that better datasets are required
* [Fire Detection from scratch using YOLOv3](https://medium.com/@b117020/fire-detection-using-neural-networks-4d52c5cd55c5) - discusses annotation using LabelImg, using Google drive and Colab, deployment via Heroku and viz using Streamlit [here](https://fireapp-aicoe.herokuapp.com/). Work by Devdarshan Mishra
* [fire-detect-yolov4](https://github.com/gengyanlei/fire-detect-yolov4) - training of Yolo v4 model
* [midasklr/FireSmokeDetectionByEfficientNet](https://github.com/midasklr/FireSmokeDetectionByEfficientNet) - Fire and smoke classification and detection using efficientnet, Python 3.7、PyTorch1.3, visualizes the activation map, includes train and inference scripts
* [arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection](https://github.com/arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection) - A Specialized Lightweight Fire & Smoke Detection Model for Real-Time IoT Applications (e.g. on RPi), accuracy approx. 95%. Paper https://arxiv.org/abs/1905.11922v2
* [tobybreckon/fire-detection-cnn](https://github.com/tobybreckon/fire-detection-cnn) - links to a couple of datasets
* [EmergencyNet](https://github.com/ckyrkou/EmergencyNet) - identify fire and other emergencies from a drone
* [Fire Detection using CCTV images — Monk Library Application](https://medium.com/towards-artificial-intelligence/fire-detection-using-cctv-images-monk-library-application-242df1fca2b9) - keras classifier on kaggle datasets, mobilenet-v2, densenet121 and densenet201
* [fire-detection-cnn](https://github.com/tobybreckon/fire-detection-cnn) - automatic detection of fire pixel regions in video (or still) imagery within real-time bounds. maximal accuracy of 0.93 for whole image binary fire detection (1), with 0.89 accuracy within our superpixel localization framework can be achieved
* [Early Fire detection system using deep learning and OpenCV](https://towardsdatascience.com/early-fire-detection-system-using-deep-learning-and-opencv-6cb60260d54a) - customized InceptionV3 and CNN architectures for indoor and outdoor fire detection. 980 images for training and 239 images for validation, training accuracy of 98.04 and a validation accuracy of 96.43, openCV used for live detection on webcam - code and datasets (already referenced here) on https://github.com/jackfrost1411/fire-detection
* [Smoke-Detection-using-Tensorflow 2.2](https://github.com/abg3/Smoke-Detection-using-Tensorflow-2.2) - EfficientDet-D0, 733 annotated smoke images, mentioned on the [Roboflow blog](https://blog.roboflow.com/fighting-wildfires/)
* [Aerial Imagery dataset for fire detection: classification and segmentation using Unmanned Aerial Vehicle (UAV)](https://github.com/AlirezaShamsoshoara/Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle) - binary classifier, 76% accuracy on test set
* [A Forest Fire Detection System Based on Ensemble Learning](https://www.mdpi.com/1999-4907/12/2/217) ->  Firstly, two individual learners Yolov5 and EfficientDet are integrated to accomplish fire detection process. Secondly, another individual learner EfficientNet is responsible for learning global information to avoid false positives
* [Fire Alert System with Multi-Label Classification Model Explained by GradCAM](https://medium.com/@wongsirikuln/fire-alert-system-with-multi-label-classification-model-explained-by-gradcam-bc18affe178c) -> use CAM to visualize which region of an image is responsible for a prediction, and uses synthetic data to fill in lacking classes to make class distribution balanced

## Datasets
* [FireNET](https://github.com/OlafenwaMoses/FireNET) - approx. 500 fire images with bounding boxes in pascal voc XML format. Repo contains trained Yolo3 model trained using [imageai](https://github.com/OlafenwaMoses/ImageAI), unknown performance. However small images, 275x183 pixels on average, meaning there are fewer textural features for a network to learn.
* [Fire Detection from CCTV on Kaggle](https://www.kaggle.com/ritupande/fire-detection-from-cctv) - images and video, images are extracted from video, relatively small dataset with all images only taken from 3-4 videos. Quite relevant to current task as have videos to test on. Dataset organised for classification task of normal/smoke/fire, no bounding box annotations
* [cair/Fire-Detection-Image-Dataset](https://github.com/cair/Fire-Detection-Image-Dataset) - This dataset contains many normal images and 111 images with fire. Dataset is highly unbalanced to reciprocate real world situations. Images are decent size but not annotated.
* [mivia Fire Detection Dataset](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/) - approx. 30 videos
* [USTC smoke detection](http://smoke.ustc.edu.cn/datasets.htm) - links to various sources that provide videos of smoke
* fire/not-fire dataset in the pyimagesearch article can be downloaded. Note that there are many images of fire scenes that do not contain actual fire, but burnt out homes for example.
* [FIRE Dataset on Kaggle](https://www.kaggle.com/phylake1337/fire-dataset) - 755 outdoor fire images and 244 non-fire images. Images are decent size but not annotated
* [Fire Image Data Set for Dunnings 2018 study](https://collections.durham.ac.uk/files/r2d217qp536#.X2rv1ZNKidb) - PNG still image set
* [Fire Superpixel Image Data Set for Samarth 2019 study](https://collections.durham.ac.uk/files/r10r967374q#.X2rv1pNKidb) - PNG still image set
* [Wildfire Smoke Dataset](https://public.roboflow.com/object-detection/wildfire-smoke) - 737 annotated (bounding boxed) images
* [Dataset by jackfrost1411](https://github.com/jackfrost1411/fire-detection) -> several hundred images sorted into fire/neutral for classification task. No bounding box annotations
* [fire-and-smoke-dataset on Kaggle](https://www.kaggle.com/dataclusterlabs/fire-and-smoke-dataset) -> 7000+ images, consisting of 691 flame only images, 3721 smoke only images, and 4207 fire {flame & smoke} images

## Fire safety references
* Locate reference covering the different kinds of fires in the home, common scenarios & interventions
* Safety/accuracy standards for fire detectors, including ROC characteristics

## Fires in the home
* Common causes including cigarettes left smouldering, candles, electrical failures, chip pan fires
* A large number of factors affect the nature of the fire, primarily the fuel and oxygenation, but also where the fire is, middle of the room/against a wall, thermal capacity of a room, the walls, ambient temperature, humidity, contaminants on the material (dust, oil based products, emollients etc)
* To put out a fire a number of retardants are considered - water (not on electrical or chip pan), foam, CO2, dry powder
* In electrical fires the electricity supply should first be isolated
* Reducing ventillation, e.g. by closing doors, will limit fire
* Smoke itself is a strong indicator of the nature of the fire
* Read https://en.m.wikipedia.org/wiki/Fire_triangle and https://en.m.wikipedia.org/wiki/Combustion

## Edge deployment
Our end goal of deployment to an edge device (RPi, jetson nano, android or ios) will influence decisions about architecture and other tradeoffs.
* [Deploy YOLOv5 to Jetson Xavier NX at 30FPS](https://blog.roboflow.com/deploy-yolov5-to-jetson-nx/) - inference at 30 FPS
* [How to Train YOLOv5 On a Custom Dataset](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/)
* [Train YOLOv4-tiny on Custom Data - Lightning Fast Object Detection](https://blog.roboflow.com/train-yolov4-tiny-on-custom-data-lighting-fast-detection/)
* [How to Train a Custom TensorFlow Lite Object Detection Model](https://blog.roboflow.com/how-to-train-a-tensorflow-lite-object-detection-model/) - colab notebook, MobileNetSSDv2, deploy to RPi
* [How to Train a Custom Mobile Object Detection Model with YOLOv4 Tiny and TensorFlow Lite](https://blog.roboflow.com/how-to-train-a-custom-mobile-object-detection-model/) - train YOLOv4 tiny Darknet and convert to tflite, demo on android, more steps than training straight for tflite
* [AI for AG: Production machine learning for agriculture](https://medium.com/pytorch/ai-for-ag-production-machine-learning-for-agriculture-e8cfdb9849a1) - complete workflow from training to deployment
* [Pytorch now officially supports RPi]()https://pytorch.org/blog/prototype-features-now-available-apis-for-hardware-accelerated-mobile-and-arm64-builds/
* [Hermes is a Wildfire detection system that utilizes Computer Vision and is accelerated using NVIDIA Deepstream](https://github.com/kn1ghtf1re/Hermes-Deepstream)

## Cloud deployment
We want a solution that could also be deployed to the cloud, with minimal changes vs the edge deployment. A couple of options:
* [Deploy as a lambda function](https://towardsdatascience.com/scaling-machine-learning-from-zero-to-hero-d63796442526) - in my experience response times are long, up to 45 seconds
* Deploy on a VM with custom code to handle queuing of requests
* [Use torchserve on sagemaker, runs on EC2 instance](https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-fastai-model-with-torchserve). Well documented but AWS specific.
* Use one of the cloud providers, e.g. AWS Rekognition will identify fire

## Image preprocessing and augmentation
Roboflow allows up to 3 types of augmentation per dataset, in addition to basic cropping. If we want to experiment with more augmentations we can checkout https://imgaug.readthedocs.io/en/latest/
* [Why Image Preprocessing and Augmentation Matters](https://blog.roboflow.com/why-preprocess-augment/)
* [The Importance of Blur as an Image Augmentation Technique](https://blog.roboflow.com/using-blur-in-computer-vision-preprocessing/)
* [When to Use Contrast as a Preprocessing Step](https://blog.roboflow.com/when-to-use-contrast-as-a-preprocessing-step/)
* [Data Augmentation in YOLOv4](https://blog.roboflow.com/yolov4-data-augmentation/)
* [Why to Add Noise to Images for Machine Learning](https://blog.roboflow.com/why-to-add-noise-to-images-for-machine-learning/)
* [Why and How to Implement Random Crop Data Augmentation](https://blog.roboflow.com/why-and-how-to-implement-random-crop-data-augmentation/)
* [When to Use Grayscale as a Preprocessing Step](https://blog.roboflow.com/when-to-use-grayscale-as-a-preprocessing-step/)

## ML metrics
* `Precision` is the accuracy of the predictions, calculated as `precision = TP/(TP+FP)` or "what % of predictions are correct?"
* `Recall` is the **true positive rate** (TPR), calculated as `recall = TP/(TP+FN)` or "what % of true positives does the model capture?"
* The `F1 score` (also called the F score or the F measure) is the harmonic mean of precision and recall, calculated as `F1 = 2*(precision * recall)/(precision + recall)`. It conveys the balance between the precision and the recall. [Ref](https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/)
* The **false positive rate** (FPR), calculated as `FPR = FP/(FP+TN)` is often plotted against recall/TPR in an [ROC curve](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) which shows how the TPR/FPR tradeoff varies with classification threshold. Lowering the classification threshold returns more true positives, but also more false positives
* mAP, IoU, precision and recall are all explained well [here](https://github.com/AlexeyAB/darknet#how-to-train-tiny-yolo-to-detect-your-custom-objects) and [here](https://github.com/jshaffer94247/Counting-Fish#Model-Metrics)
* IceVision returns the COCOMetric, specifically the `AP at IoU=.50:.05:.95 (primary challenge metric)`, from [here](https://cocodataset.org/#detection-eval), typically referred to as the "mean average precision" (mAP)
* `mAP@0.5`: the mean Average Precision or correctness of each label taking into account all labels. `@0.5` sets a threshold for how much of the predicted bounding box overlaps the original annotation, i.e. "50% overlap"

## Comments
* Firenet is a VERY common name for model, do not use

## Discussion
* [Thread I have started on the fast.ai forum](https://forums.fast.ai/t/yolo-v5-implementation-in-fastai2/79738)

## Demo
The best performing model can be used by running the demo app that created with [Gradio](https://gradio.app/). Note you must have the yolov5 repo cloned locally (`git clone https://github.com/ultralytics/yolov5`) and the path to it
* `python3 -m venv venv`
* `source venv/bin/activate`
* `pip3 install -r requirements.txt`
* `python3 demo.py`
* You are prompted to navigate to [http://127.0.0.1:7860/](http://127.0.0.1:7860/)

<p align="center">
<img src="https://github.com/robmarkcole/fire-detection-from-images/blob/master/images/demo.png" width="1100">
</p>
