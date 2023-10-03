# IAF-RCNN
Illumination-aware Faster R-CNN

Discover:
-	Pedestrian detection confidences from color or thermal images are correlated with illumination conditions.

Main:
-	an Illumination-aware Network is introduced to give an illumination measure of the input image.
-	adaptively merge color and thermal sub-networks via a gate function defined over the illumination value.

Archietecture:
-	Base on Faster R-CNN
-	IAF R-CNN is composed of three parts:
  -	Multispectral backbone. (Multispectral Faster R-CNN)
  -	Illumination Estimation.
  -	Gated fusion layer.
  -	the estimated illumination value iv ∈ [0, 1].
-	Illumination-aware Network (IAN)
  -	Compared with:
    -	A chains of convolutional→ 2 convolutional layers with 3x3 filters, each of which followed by a ReLU layer. 
    -	Max pooling layers. → A 2×2 max pooling layer.
    -	Fully-connected layers. → two subsequent fully-connected layers with 256 and 2 neurons respectively.
    -	A dropout layer with a ratio of 0.5 is inserted after the first fully-connected layer to alleviate over-fitting.
  -	How to train IAN?
    -	using the coarse day/night labels instead to train IAN.
  -	Input: color image. And resized to 56x56 pixels.
  -	Goal: estimate the illumination conditions.
