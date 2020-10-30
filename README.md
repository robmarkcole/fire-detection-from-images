# fire-detection-from-images
The purpose of this repo is to demonstrate a fire detection neural net that can be deployed to an edge device as well as the cloud. It is to be connected to a camera that might be in a fixed location or on a roving drone e.g [Ring Always Home Cam](https://blog.ring.com/2020/09/24/introducing-ring-always-home-cam-an-innovative-new-approach-to-always-being-home/)

## Best results
**Object detection:** After experimenting with various model architectures I settled on Yolov5 pytorch model owing to its being SOTA (state of the art), relatively fast to train, and the availability of a well documented and easy to use notebook. After a few hours of experimentation I generated a model of `mAP@.5` of 0.657, Precision of 0.6, Recall of 0.7, trained on 1155 images (337 base images + augmentation).

**Classification:** I have yet to train my own model, but 95% accuracy is reported using the ResNet50 with the Monk tensorflow library.

## Motivation and challenges
Traditional smoke detectors work by [detecting the physical presence of smoke particles](https://www.nfpa.org/Public-Education/Staying-safe/Safety-equipment/Smoke-alarms/Ionization-vs-photoelectric). However they are prone to false detections (e.g. from toasters) and do not localise the fire particularly well. In these situations a camera solution could complement a traditional detector, in order to improve response times or to provide additional metrics such as the size and location of a fire. With the location and nature of the fire identified, an automated intervention may be possible, e.g. via a sprinkler system or drone. Also data can be sent to fire services to provide otherwise non-existent situational awareness. Particular locations I am interested in are: kitchens & living rooms, garages and outbuildings, and areas where fires might already be present but spreading outside a desired zone e.g. fire pit.

There are a couple of significant challenges & open questions:
* For fast edge model what is 'best' architecture? Yolo3 is very popular for commecrial applications and can be implemented in keras or pytorch, baseline [Yolov5](https://github.com/ultralytics/yolov5) as it is currently SOTA and has deployment guide to Jetson. 
* Can the architecture be optimised since we are detecting only a single class?
* Baseline object detection, but is there benefit to classifier or using both? Obj models train on mAP and Recall metrics but for our application bounding box accuracy may not be top priority? However classification models work best on a nice shot containing only the target object but in real life fire scenarios the scene will not be as simple as this scenario.
* Tensorflow + google ecosystem or Pytorch + NVIDIA/MS? Tensorflow suffers from tf1 legacy
* Is a single 'super' model preferable, or several specialised models? Typical categories of fire include candle flame, indoor/outdoor, vehicle
* Gathering or locating a comprehensive, representative and balanced training dataset
* Handling different viewpoints, different camera manufacturers and settings, and different ambient lighting conditions.
* Since fires are so bright they can often wash out images and cause other optical disturbances, how can this be compensated for?
* Since we expect the model will have limitations, how do we make the model results interpretable?

Ideas:
* Preprocessing images, e.g. to remove background or apply filters
* Classifying short sequences of video, since the movement of fire is quite characteristic
* Simulated data, identify any software which can generate realistic fires and add to existing datasets
* Augmentations to simulate effect of different cameras and exposure settings
* Identify any relevant guidance/legislation on required accuracy of fire detection techniques

## Approach & Tooling
* Frames will be fed through neural net. On positive detection of fire metrics are extracted. Ignore smoke for MVP. Try various architectures & parameters to establish a 'good' baseline model.
* Develop a lower accuracy but fast model targeted at RPi and mobile, and a high accuracy model targeted at GPU devices like Jetson. Yolo present both options, yolo4 lite for mobile and yolo5 for GPU. Alternatively there is mobilenet and tf-object-detection-api. Higher accuracy GPU model is priority.
* Use Google Colab for training and [Roboflow](https://app.roboflow.com/) for image dataset curation as allows easy export into common formats e.g. tfrecord. Once we get serious can use sagemaker or google equivalent (Deep Learning VM), and [weights & biases](https://www.wandb.com/)
* For cloud serving use Amazon SageMaker, e.g. [Serving PyTorch models in production with the Amazon SageMaker native TorchServe integration](https://aws.amazon.com/blogs/machine-learning/serving-pytorch-models-in-production-with-the-amazon-sagemaker-native-torchserve-integration/)
* [LabelImg for Labeling](https://blog.roboflow.com/labelimg/)

## Articles & repos
* [Fire and smoke detection with Keras and Deep Learning by pyimagesearch](https://www.pyimagesearch.com/2019/11/18/fire-and-smoke-detection-with-keras-and-deep-learning/) - dataset collected by scraping Google images (provides link to dataset with  1315 fire images), binary Fire/Non-fire classification with tf2 & keras sequential CNN, achieve 92% accuracy, concludes that better datasets are required
* [Fire Detection from scratch using YOLOv3](https://medium.com/@b117020/fire-detection-using-neural-networks-4d52c5cd55c5) - discusses annotation using LabelImg, using Google drive and Colab, deployment via Heroku and viz using Streamlit
* [fire-detect-yolov4](https://github.com/gengyanlei/fire-detect-yolov4) - training of Yolo v4 model
* [midasklr/FireSmokeDetectionByEfficientNet](https://github.com/midasklr/FireSmokeDetectionByEfficientNet) - Fire and smoke classification and detection using efficientnet, Python 3.7、PyTorch1.3, visualizes the activation map, includes train and inference scripts
* [arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection](https://github.com/arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection) - A Specialized Lightweight Fire & Smoke Detection Model for Real-Time IoT Applications (e.g. on RPi), accuracy approx. 95%. Paper https://arxiv.org/abs/1905.11922v2
* [tobybreckon/fire-detection-cnn](https://github.com/tobybreckon/fire-detection-cnn) - links to a couple of datasets
* [EmergencyNet](https://github.com/ckyrkou/EmergencyNet) - identify fire and other emergencies from a drone
* [Fire Detection using CCTV images — Monk Library Application](https://medium.com/towards-artificial-intelligence/fire-detection-using-cctv-images-monk-library-application-242df1fca2b9) - keras classifier on kaggle datasets, mobilenet-v2, densenet121 and densenet201
* [fire-detection-cnn](https://github.com/tobybreckon/fire-detection-cnn) - automatic detection of fire pixel regions in video (or still) imagery within real-time bounds. maximal accuracy of 0.93 for whole image binary fire detection (1), with 0.89 accuracy within our superpixel localization framework can be achieved
* [Early Fire detection system using deep learning and OpenCV](https://towardsdatascience.com/early-fire-detection-system-using-deep-learning-and-opencv-6cb60260d54a) - customized InceptionV3 and CNN architectures for indoor and outdoor fire detection. 980 images for training and 239 images for validation, training accuracy of 98.04 and a validation accuracy of 96.43, openCV used for live detection on webcam - code and datasets (already referenced here) on https://github.com/jackfrost1411/fire-detection

## Datasets
* [FireNET](https://github.com/OlafenwaMoses/FireNET) - approx. 500 fire images with bounding boxes. Repo contains trained Yolo3 model trained using [imageai](https://github.com/OlafenwaMoses/ImageAI), unknown performance.
* [Fire Detection from CCTV on Kaggle](https://www.kaggle.com/ritupande/fire-detection-from-cctv) - images and video, images are extracted from video, relatively small dataset. Quite relevant to current task as have videos to test on.
* [cair/Fire-Detection-Image-Dataset](https://github.com/cair/Fire-Detection-Image-Dataset) - This dataset contains normal images and images with fire, and is highly unbalanced to reciprocate real world situations.
* [mivia Fire Detection Dataset](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/) - approx. 30 videos
* [USTC smoke detection](http://smoke.ustc.edu.cn/datasets.htm) - links to various sources that provide videos of smoke
* fire/not-fire dataset in the pyimagesearch article can be downloaded. Note that there are many images of fire scenes that do not contain actual fire, but burnt out homes for example.
* [FIRE Dataset on Kaggle](https://www.kaggle.com/phylake1337/fire-dataset) - 755 outdoor fire images and 244 non-fire images. Many glossy images, representative?
* [Fire Image Data Set for Dunnings 2018 study](https://collections.durham.ac.uk/files/r2d217qp536#.X2rv1ZNKidb) - PNG still image set
* [Fire Superpixel Image Data Set for Samarth 2019 study](https://collections.durham.ac.uk/files/r10r967374q#.X2rv1pNKidb) - PNG still image set

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

## Cloud deployment
We want a solution that could also be deployed to the cloud, with minimal changes vs the edge deployment. A couple of options:
* [Deploy as a lambda function](https://towardsdatascience.com/scaling-machine-learning-from-zero-to-hero-d63796442526) - will memory be and performance issue? Scaling sorted.
* Deploy on a VM with code to handle queuing of requests, e.g. [Deepstack](https://deepquestai.com/). Will scaling be an issue?
* Use one of the cloud providers that host custom models, e.g. [AWS custom labels](https://aws.amazon.com/rekognition/custom-labels-features/). Dont know much about this approach, prefer vendor agnostic.
* [Use torchserve on sagemaker](https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-fastai-model-with-torchserve)

## Image preprocessing and augmentation
Roboflow allows up to 3 types of augmentation per dataset, in addition to basic cropping. If we want to experiment with more augmentations we can checkout https://imgaug.readthedocs.io/en/latest/
* [Why Image Preprocessing and Augmentation Matters](https://blog.roboflow.com/why-preprocess-augment/)
* [The Importance of Blur as an Image Augmentation Technique](https://blog.roboflow.com/using-blur-in-computer-vision-preprocessing/)
* [When to Use Contrast as a Preprocessing Step](https://blog.roboflow.com/when-to-use-contrast-as-a-preprocessing-step/)
* [Data Augmentation in YOLOv4](https://blog.roboflow.com/yolov4-data-augmentation/)
* [Why to Add Noise to Images for Machine Learning](https://blog.roboflow.com/why-to-add-noise-to-images-for-machine-learning/)
* [Why and How to Implement Random Crop Data Augmentation](https://blog.roboflow.com/why-and-how-to-implement-random-crop-data-augmentation/)
* [When to Use Grayscale as a Preprocessing Step](https://blog.roboflow.com/when-to-use-grayscale-as-a-preprocessing-step/)

## Metrics
* mAP, IoU, precision and recall are all explained well [here](https://github.com/AlexeyAB/darknet#how-to-train-tiny-yolo-to-detect-your-custom-objects) and [here](https://github.com/jshaffer94247/Counting-Fish#Model-Metrics)
* `mAP@0.5`: the mean Average Precision or correctness of each label taking into account all labels. `@0.5` sets a threshold for how much of the predicted bounding box overlaps the original annotation, i.e. "50% overlap"
* `Precision` is the accuracy of the positive predictions `(TP / TP + FP)` or "If you say it's a fire, what percentage of the time is it really a fire?"
* `Recall` is the true positive rate `(TP / TP + FN)` or "If there's a fire in there, what percentage of the time do you find it?"

## Comments
* Firenet is a VERY common name for model, do not use

## Discussion
* [Thread I have started on the fast.ai forum](https://forums.fast.ai/t/yolo-v5-implementation-in-fastai2/79738)
