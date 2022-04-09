# Automatic Report Generation From Chest X-Rays
MTech Project thesis - Phase - I

## Business Problem
Clinical imaging captures enormous amounts of information but most radio-logic data are reported in qualitative and subjective terms. X-Rays are a form of Electromagnetic Radiation that is used for medical imaging. Analysis of X-ray reports is a very important task of radiologists and pathologists to recommend the correct diagnosis to the patients. In this project, we are tackling the image captioning problem for a data set containing Chest X-ray images with the help of the state of the art deep learning architecture and optimizing parameters of the architecture.

The problem statement here is to find the findings from the given chest X-Ray images. These images are of two types: Frontal and Lateral view of the chest. With these two types of images as input we need to find the impression for given X-Ray. To resolve this problem statement, we will be building a predictive model which involves both image and text processing to build a deep learning model.

## File Structure
> EDA
> Basic Model

## My Approach – Solution
Initially I will be doing the EDA and Preproccesing of the data with image as input and text as output. I could find the data imbalance, Images availability per patient, Type of images associated for each patient. After this step I will be implementing deep learning model with two different approach to find the improvement on one another.

## EDA & Pre-Processing

**Total Observations from EDA:**

* The dataset contains chest X-ray images and radiology text reports. Each image has been paired with four captions such as Impressions, Findings, Comparison and Indication that provide clear descriptions of the salient entities and events. All the raw texts from xml files are parsed and created the dataset.

* I consider finding features as Target variable, as it has much data volume and from sample data points, It is clear that this feature gives most of the information present in images.

* Images are in different shapes. All the X-Ray images are human upper body particularly about Chest part.

* Each patient have multiple x-rays associated with them. The maximum number of images associated with a report can be 5 while the minimum is 0. The highest frequency of being associated with a report are 2 images.

* Data is incomplete. Because all the features have few missing values except caption. We have to impute the missing values in data preprocessing step.

* In text features there are some unknown values like XXXX XXXXX these are replaced with empty string.

* We have total of 3955 records and Findings is our target variable.

* Most occurring words of diffrent features:
    
    > Findings: Pleural effusion

    > Impression: acute cardiopulmonary

* I created wordcloud, for 500 most frequent words of the feature. These are important words. Some of them are: acute, findings, disease, abnormality, high, right, impression, etc.


## For constructing baseline model :
- First I worked on getting structured data.

- limiting the data point to 2 images per data point, if we have 5 images, its 4+1 (all image + last image) so make it as 4 data points, I converted multiple images into two images using their projection. The projection dataset has been taken from kaggle.

- Dataset link : https://www.kaggle.com/raddar/chest-xrays-indiana-university

- I have used simple Encoder-Decoder architacture for basic modelling.

- Then extract the features from Images and did text tokenization and pass these tensors to model for training.

- This is basic model, we used LSTM model with pretrained-CheXNet model and I have tuned the model accordingly.

- I started with 0.001 as learning rate and gradually decrease it by ReduceLROnPlateau callback.

- CheXNet competition link : https://stanfordmlgroup.github.io/projects/chexnet/

- Publically Available weight : https://www.kaggle.com/theewok/chexnet-keras-weights

- Then I add dropout to learn more robust features in both encoder and decoder layers.

- I have also experimented with various batch sizes and observes lower batch size helps to achieve better test accuracy.

- Instead of dense layer I used TimeDistributedDense layer. as it applies a same dense to every time step during GRU/LSTM Cell unrolling. So the error function will be between predicted label sequence and the actual label sequence. (Which is normally the requirement for sequence to sequence labeling problems).

- The bleu score has got 0.541 which is somewhat satisfactory compared to other referred models.

- I have also used Cumulative N-Gram Scores(BLEU Score). Cumulative scores refer to the calculation of individual n-gram scores at all orders from 1 to n and weighting them by calculating the weighted geometric mean.
