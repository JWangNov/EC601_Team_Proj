# Analysis of other solutions on Kaggle

Note: On Kaggle, the private leaderboard is calculated with approximately 75% of the test data. Public LB score is calculated with 25% of the test data.

## Solution 1:
Here is the link to the solution: [I'm an link]
https://github.com/pudae/kaggle-understanding-clouds
### Settings:
**Two models**: UNet with classification head, FPN or UNet, no classification head

**Dataset**: train vs val = 9:2

**Backbone**: resnet34, efficientnet-b1, resnext10132x8dwsl, resnext10132x16dwsl

**Augmentation**: hflip, vflip, shift/scale/rotate, grid distortion, channel shuffle, invert, to gray

### Special Ideas:

1) To solve the issue of unstable LB scores: the reason for the unstable LB score was because of poor segmentation performance. If there is a more powerful segmentation model, the effect of poor classification performance can be reduced.
<pre>
  TTA4, 1 seg with cls + 1 seg: CV 0.6560, Public LB 0.67395, Private LB 0.66495
  TTA4, 1 seg with cls + 3 seg: CV 0.6582, Public LB 0.67482, Private LB 0.66501
  TTA4, 1 seg with cls + 4 seg: CV 0.6587, Public LB 0.67551, Private LB 0.66604
  TTA4, 1 seg with cls + 7 seg: CV 0.6594, Public LB 0.67596, Private LB 0.66663
</pre>
2) To take advantage of the performance of the segmentation models, they used a mean of top K pixel probabilities as a classification probability.
segmentation models could predict empty masks more accurately than a classification model trained on labels 

<pre>
  cls_probabilities = np.sort(mask_probabilities.reshape(4, -1), axis=1)
  cls_probabilities = np.mean(cls_probabilities[:,-17500:], axis=1)
</pre>


3) All images in the train set have at least one type of cloud, so they treated the label of max probability in each image as a positive prediction.
<pre>
   cls_probabilities[np.argmax(cls_probabilities)] = 1
</pre>



## Solution 2:

### Ideas:
1) Pure segmentation models without false positive classifier:<br/> 
    After reaching public LB 0.6752 with segmentation model, this kaggler has trained a few classifiers using Resnet34, SE-ResNext-50 and EfficientNet-B4 but the performance is pretty unstable (+/- 0.003 ~ 0.010) in local cross validations of 10 folds. Thus, he discarded the classifiers and decided to stick with segmentation models. 

2) Network Architectures:<br/> 
    He has used the awesome implementations of various models from segmentation_models.pytorch, pretrained-models.pytorch , EfficientNet-PyTorch and Resnet34-ASPP from @hengck23. His final ensemble used 7 folds of EfficientNet-B4-FPN and 3 folds of Resnet34-ASPP as they have better performance and more stable in error convergence in my case after running rounds of experiments using various network architectures.
	
3) RAdam Optimizer:<br/> 
    RAdam helped to stabilize training error convergence as it is less sensitive to learning rate change in my case, thus minimizing the variance.  Flat threshold of 0.4 for all classes Threshold of 0.4 yielded the highest cross validation DICE score when compared in the range of [0.4, 0.5, 0.6], no further fine-tuning of threshold is done. 
	

