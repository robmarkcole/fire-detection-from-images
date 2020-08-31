# fire-detection-from-images
The purpose of this repo is to demonstrate a fire detection neural nets model that can be deployed to an edge device, presumed to be connected to a CCTV camera.

## Motivation and challenges
Traditional smoke detectors work by [detecting the physical presence of smoke particles](https://www.nfpa.org/Public-Education/Staying-safe/Safety-equipment/Smoke-alarms/Ionization-vs-photoelectric). However they are not suitable or effective in all conditions, in particular in large well ventilated spaces and at long distances. In these situations a camera solution might complement a traditional detector, in order to improve response times or to provide additional metrics such as the size and location or a fire. Particular scenarios I am interested in are: outdoors, garages and outbuildings, shared communal areas where fires might already be present, e.g for cooking.

There are a couple of significant challenges:
* Gathering or locating a comprehensive, representative and balanced training dataset. The dataset should cover small/large/indoor/outdoor fires, at different viewpoints, for different camera manufacturers and settings, and under different lighting conditions.
* Since we expect the model will have limitations, how do we make the model results interpretable?

## Tooling and approach
* Frames will be fed through a fire/normal classifier. On positive detection of fire frames are fed to an object detection model to determine the size/severity of the fire. Ignore smoke for MVP. Try various architectures & parameters to establish a 'good' baseline model.
* Remain open to pytorch & tensorflow2
* Use Google Colab for training and host images on Google drive which has a nice UI, desktop apps with sync, easy auth. Additionally or alternatively use kaggle, any particular advantages vs colab?
* Identify any relevant guidance/legislation on required accuracy of fire detection techniques

## Articles & repos
* [Fire and smoke detection with Keras and Deep Learning by pyimagesearch](https://www.pyimagesearch.com/2019/11/18/fire-and-smoke-detection-with-keras-and-deep-learning/) - dataset collected by scraping Google images (provides link to dataset with  1315 fire images), binary Fire/Non-fire classification with tf2 & keras sequential CNN, achieve 92% accuracy, concludes that better datasets are required
* [Fire Detection from scratch using YOLOv3](https://medium.com/@b117020/fire-detection-using-neural-networks-4d52c5cd55c5) - discusses annotation using LabelImg, using Google drive and Colab, deployment via Heroku and viz using Streamlit
* [fire-detect-yolov4](https://github.com/gengyanlei/fire-detect-yolov4) - training of Yolo v4 model
* [midasklr/FireSmokeDetectionByEfficientNet](https://github.com/midasklr/FireSmokeDetectionByEfficientNet) - Fire and smoke classification and detection using efficientnet, pytorch, visualizes the activation map
* [arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection](https://github.com/arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection) - A Specialized Lightweight Fire & Smoke Detection Model for Real-Time IoT Applications (e.g. on RPi), accuracy approx. 95%. Paper https://arxiv.org/abs/1905.11922v2
* [tobybreckon/fire-detection-cnn](https://github.com/tobybreckon/fire-detection-cnn) - links to a couple of datasets
* [EmergencyNet](https://github.com/ckyrkou/EmergencyNet) - identify fire and other emergencies from a drone
* [Fire Detection using CCTV images — Monk Library Application](https://medium.com/towards-artificial-intelligence/fire-detection-using-cctv-images-monk-library-application-242df1fca2b9) - keras classifier on kaggle datasets, mobilenet-v2, densenet121 and densenet201

## Datasets
* [FireNET](https://github.com/OlafenwaMoses/FireNET) - approx. 500 fire images with bounding boxes
* [Fire Detection from CCTV on Kaggle](https://www.kaggle.com/ritupande/fire-detection-from-cctv) - images and video, images are extracted from video, relatively small dataset. Quite relevant to current task as have videos to test on.
* [cair/Fire-Detection-Image-Dataset](https://github.com/cair/Fire-Detection-Image-Dataset) - This dataset contains normal images and images with fire, and is highly unbalanced to reciprocate real world situations.
* [mivia Fire Detection Dataset](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/) - approx. 30 videos
* [USTC smoke detection](http://smoke.ustc.edu.cn/datasets.htm) - links to various sources that provide videos of smoke
* fire/not-fire dataset in the pyimagesearch article can be downloaded. Note that there are many images of fire scenes that do not contain actual fire, but burnt out homes for example.
* [FIRE Dataset on Kaggle](https://www.kaggle.com/phylake1337/fire-dataset) - 755 outdoor fire images and 244 non-fire images. Many glossy images, representative?

## Comments
* firenet is a VERY common name for model, do not use
