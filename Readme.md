# Understanding Clouds from Satellite Images

## Sprint 1:
### Product Definition
#### 1) Product Mission
Build a model to classify cloud organization patterns from satellite images. If successful, it will help scientists to better understand how clouds will shape our future climate. This project will guide the development of next-generation models which could reduce uncertainties in climate projections.<br/>
#### 2) Target User(s)
•	 Climates scientists who want to better understand how clouds will shape our future climate.<br/><br/>
•	 Weather station that wants to provide more precise weather forecasting.<br/><br/>
•	 Companies that want to develop an APP that can forecast weather change for users by detecting a cloud image.<br/><br/>
#### 3) User Stories
As a climate scientist, I want to create a model to better classify cloud organization patterns from satellite images so I can project the future climate more precisely.<br/>
#### 4) MVP
**Major components:** A recognize pattern to learn and analyze cloud images.<br/><br/>
**Technology need to develop and analysis:** There are many ways in which clouds can organize, but the boundaries between different forms of organization are murky. This makes it challenging to build traditional rule-based algorithms to separate cloud features. <br/>

### Industry Review and Analysis
*Image Recognition*

#### Apple Photos App

a typical Apple stuff...

#### CatAndDogRecognizer
[https://github.com/klevis/CatAndDogRecognizer](https://github.com/klevis/CatAndDogRecognizer)

* Java
* Deep Neural Networks (Deeplearning4j)
* VGG - 16 architecture (convolution architecture, to simplify the training process)

##### License
[**EPL**](https://www.eclipse.org/legal/epl-v10.html)

Permissions:

- Commercial use
- Distribution
- Modification
- Patent use
- Private use

Limitations:

- Liability
- Warranty

Conditions:

- Disclose source
- License and copyright notice
- Same license

#### cntk-hotel-pictures-classificator
[https://github.com/karolzak/cntk-hotel-pictures-classificator](https://github.com/karolzak/cntk-hotel-pictures-classificator)

can recognize pictures of hotel stuffs (sink, bed, lamp, pillow), and classify them

* Python
* CNTK (Microsoft Cognitive Toolkit)
* FasterRCNN (Region-CNN)

##### License
[**MIT**](https://www.mit.edu/~amini/LICENSE.md)

Permissions:

- Commercial use
- Modification
- Distribution
- Private use

Limitations:

- Liability
- Warranty

Conditions:

- License and copyright notice

### Kaggle Notebooks Analysis
Some Ideas Shared on Kaggle

#### [EDA: Find Me In The Clouds](https://www.kaggle.com/ekhtiar/eda-find-me-in-the-clouds#Drawing-Clouds)
***Pretreatment of the Training Data***

Firstly, analyze the training data. 
Explore the correlation between different types of cloud. 

Then explore the data pictures. 
To know how many types of clouds we have per image normally. 

Next, mask. 
Draw cloud shapes on images. 
A mask not outlining the exact clouds but roughly the area with the same kind of patterns. 
This is to cut off the parts that we don't want to use in training, improving the accuracy of training.

Finally, visualize the result. 
TBH this step is not necessary for the Kaggle project. 
But it's valuable for those users who want to learn other things from the cloud patterns. 

* Python
* cv2
* Pandas
* NumPy

#### [Image Segmentation From scratch using Pytorch](https://www.kaggle.com/dhananjay3/image-segmentation-from-scratch-in-pytorch)
***To Build a Training Neural Network***

Use vanilla U-Net Architecture to mask images and training. 
Use GPU to make training be faster. 

* Python
* PyTorch
* CNN
  * Vanilla U-Net (or we can use LinkNet)
  * Deterministic behavior for reproducibility
