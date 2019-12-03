# Data Augmentation through Transform Class #
## Introduction ##
Annotating samples is expensive and time-consuming, and sample collection for special scenarios, such as the use of drones to capture the surface of a bridge, which is very difficult to obtain. Therefore, in the field of deep learning, data augumentation(DA) is a common skill. Pytorch provides an official package that includes 20 classes on PIL images and 3 classes on tensor for data augmentation.

## Requirements ##
•	Implement at least 5 different data augmentation methods by using torchvision.transforms.  
•	Choose a network and train it on 1,000 samples with each of the augmentation methods.
## Dependencies ##

> * Python 3.7.3
> * NVIDIA GeForce GTX 1080
> * NVIDIA GeForce GTX Titan X
> * PyTorch 1.0.1

## Results ##

|nets            |augmentation methods |  parameters |epochs        |accuracy   | 
|:--------------:|:-------------------:|:-----------:|:------------:|:---------:|
| MobileNetV2    |1:None                 |2236106      |      40      |89.51%   |
| MobileNetV2    |2:CenterCrop           |2236106      |      40      |70.57%   |
| MobileNetV2    |3:RandomCrop           |2236106      |      40      |84.74%   |
| MobileNetV2    |4:RandomVerticalFlip   |2236106      |      40      |82.55%   |
| MobileNetV2    |5:RandomHorizontalFlip |2236106      |      40      |85.62%   |
| MobileNetV2    |6:RandomPerspective    |2236106      |      40      |94.65%   |
| MobileNetV2    |7:RandomApply          |2236106      |      40      |61.74%   |

Note: for a fair comparison, training and testing related hyperparameters are all selected as the same.


##  Visualization for each Method ## 

![image](https://github.com/RAKIYOU/Data-Augmentation-through-Transforms-Class/blob/master/pic1.png)
The training ID for this sample is 811 and its label is 3.
## Conclusion ##
(1) A higher accuracy of 89.51% is already available without using any DA method.       
(2) After introducing CenterCrop, the accuracy was reduced to 70.57%. There may be two possible reasons casuing 
&emsp;&emsp;this.  
&emsp; i) CenterCrop does not increase the number of training samples.  
&emsp; ii) By cropping the central region of the 224\*224 from the 300\*300 image, the peripheral information is missing and   
&emsp; &emsp;the model has caused interference. Please see the following 7 examples.
                 
![image](https://github.com/RAKIYOU/Data-Augmentation-through-Transforms-Class/blob/master/pic2.png)                 
(3) For each epoch, model will get different input pictures after randomcrop. After 40 epochs, the number of training   
&emsp;samples will be greatly increased. Theoretically, the classification accuracy should be increased. But as we can see  
&emsp;from the following pictures, RandomCrop actually reduces the quality of the training samples, Therefore the decline   
&emsp;in accuracy is not hard to understand.
  
![image](https://github.com/RAKIYOU/Data-Augmentation-through-Transforms-Class/blob/master/pic3.png)                 
(4) Randomverticalflip and Randomhorizontalflip also reduce the quality of input samples. For example, 6 turns to be 9  
 &emsp;after flipping, but its label is still 6.    
(5) In this experiment, random perspective acheives the result of data augumentation.
 ![image](https://github.com/RAKIYOU/Data-Augmentation-through-Transforms-Class/blob/master/pic4.png)    
As seen in the pictures above, this operation does not reduce the quality of the input image. In addition, the model will get different input pictures for each single epoch, which is equivalent to greatly improving the number of training samples. 
 
## Specific Operation for each Data Augmentation Method ##
Basic transformation for test data:
```
transformnone=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
```
None: Basic transformation:
```
transform1 = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
```
CenterCrop:
```
transform2 = transforms.Compose([transforms.Resize((300,300)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
```
RandomCrop:
```
transform3 = transforms.Compose([transforms.Resize((300,300)), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
```
RandomVerticalFlip:
```
transform4 = transforms.Compose([transforms.Resize((224,224)), transforms.RandomVerticalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
```
RandomHorizontalFlip:
```
transform5 = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
```
RandomPerspective:
```
transform6 = transforms.Compose([transforms.Resize((224,224)), transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),  transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
```
transforms.RandomApply:

```

transformlist = [transforms.RandomVerticalFlip(p=0.5), transforms.RandomHorizontalFlip(p=0.5),  transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)]
transform7 = transforms.Compose([transforms.Resize(224), transforms.RandomApply(transformlist, p=0.5), transforms.ToTensor(),   transforms.Normalize((0.1307,), (0.3081,))])

````

## Implement mix-up on MNIST ##

I need a little bit more time for this part.
