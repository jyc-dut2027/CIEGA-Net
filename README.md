# **CIEGA-Net:A Lightweight Approach for Rainfall Opportunistic Sensing**

## **Abstract**

Video-image-based rainfall measurement is one of the frontier research directions in opportunistic rainfall sensing. It can be generally summarized into two tightly coupled stages:  (1) **rain streak information extraction**, in which the motion blur of raindrops in imaging is characterized as rain streaks; and  (2) **rainfall intensity estimation**,which maps the extracted rain-streak features to rainfall intensity.

However, under limited computational resources and constrained processing environments, achieving a balance between computational efficiency and measurement accuracy remains a key challenge in opportunistic rainfall sensing. To address this issue, we propose a lightweight machine-learning approach based on image enhancement and progressive fusion, termed the **Coupled Image Enhancement and Gradual Aggregation Network (CIEGA-NET)**, for rainfall estimation.

The method uses a progressive temporal fusion strategy. It first coarsely fuses information from multiple video-frame groups and then refines the coarsely extracted information. Compared with direct processing of adjacent multi-frame images, this coarse-to-fine strategy uses temporal information more efficiently and avoids time-consuming optical-flow estimation. Combined with image enhancement, this strategy provides high execution efficiency while maintaining good measurement accuracy. The model performs well across different scenarios and especially during hazardous heavy rainfall. It therefore addresses the key conflict in rainfall opportunistic sensing between
computational efficiency and measurement accuracy.

## **Dependencies**

> conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts

## Data Preparation

You can download the [RainSynLight25、RainSynComplex25](https://github.com/flyywh/J4RNet-Deep-Video-Deraining-CVPR-2018)、 [Jiangdata](https://figshare.com/collections/Advancing_opportunistic_sensing_in_hydrology_a_novel_approach_to_measuring_rainfall_with_ordinary_surveillance_cameras/4437791)、[Yindata](https://figshare.com/articles/dataset/Estimating_rainfall_intensity_with_a_high_spatiotemporal_resolution_using_an_image-based_deep_learning_model/14709423/1?file=28249734) and [Zhengdata](https://figshare.com/articles/dataset/Towards_Improved_Real-Time_Rainfall_Intensity_Estimation_Using_Video_Surveillance_Cameras/22122500/1?file=39322703) from the attached link.

## Usage

The proposed algorithm is essentially a two-stage pipeline composed of two cascaded modules: a **rain streak extraction module** and a **rainfall intensity estimation module**. The overall workflow is described as follows:

First, the input video is converted into a sequence of consecutive frame images:

> video_to_img.py

Rain Streaks Extraction Stage:

You can first modify

> config.py

For train the model:

> python multi_train.py

For evaluate the model:

> python multi_evaluate.py

For gain the Rain Streaks:

> python absdiff.py

Rainfall Intensity Estimation Stage:

You can first modify

> config.py

For train the model:

> python model_train.py

For test the model:

> python model_test.py
