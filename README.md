# Rover :mag: 
> Reverse engineer your CNNs, in style

<img src = "images/demo_1.gif">

Rover will help you break down your CNN and visualize the features from within the model. No need to write weirdly abstract code to visualize your model's features anymore. 

It supports pretty much any PyTorch model with an input of shape `[N, 3, H, W]` (even segmentation models/VAEs and all that fancy stuff) with imagenet normalization on input.


## Args
* `width` (`int`, optional): Width of image to be optimized 
* `height` (`int`, optional): Height of image to be optimized 
* `iters` (`int`, optional): Number of iterations, higher -> stronger visualization
* `lr` (`float`, optional): Learning rate
* `rotate (deg)` (`int`, optional): Max rotation in default transforms
* `scale max` (`float`, optional): Max image size factor. 
* `scale min` (`float`, optional): Minimum image size factor. 
* `translate (x)` (`float`, optional): Maximum translation factor in x direction
* `translate (y)` (`float`, optional): Maximum translation factor in y direction
* `weight decay` (`float`, optional): Weight decay for default optimizer. Helps prevent high frequency noise. 
* `gradient clip` (`float`, optional): Maximum value of the norm of gradient. 


## Run locally

Clone the repo
```
git clone https://github.com/Mayukhdeb/rover.git
```

install requirements
```
pip install -r requirements.txt
```

showtime

```
streamlit run explore.py
```
