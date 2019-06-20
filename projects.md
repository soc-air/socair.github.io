---
layout: page
permalink: /projects/
---

Deep Learning PyTorch Tutorials
====================

<img src="https://raw.githubusercontent.com/krshrimali/krshrimali.github.io/master/assets/pytorch.jpg" width="250">

[PyTorch Tutorials](https://github.com/krshrimali/Deep-Learning-Libraries/tree/master/PyTorch/Chapters) - This GitHub repo contains PyTorch Tutorials and Notes from the book: https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-pytorch. Highly suggest reading this book to get a good head start in PyTorch. The following topis are covered:

- **Basics of PyTorch:** Tensors, Data Loading etc.
- **Convolutional Neural Networks**
- **Visualizing Outputs from Hidden Layers**
and more.

Computer Vision Projects
===================

<img src="/assets/opencv.png" width="250">

**Dimensionality-Calculation-using-Homography-Matrix-and-QR-Code-Detection** (C++ and Python)

OpenCV based dimensional measurement of a book cover using Homography and Ratio comparison.

- **What it does?**: Approximation of the dimensions of a cover page of a book using techniques: Homography Algorithms, (QR Code Detection using Zbar). 
- **How it does?**: QR Code generation using any online web service. [Example: https://www.qr-code-generator.com/]
    * Detection of the QR Code and Text generation [encoded in the QR code - assuming text or any hyperlink etc.] using zbar module in Python. Credits: learnopencv.com
    * Printing out the QR Code on a page, assuming it to be on a book - take the snap of it, and determine the approximate dimensions of the book cover, using the measured (manually) dimensions of the QR code.
    * Note: The QR Code detection has been shown in qr_code_detection.py file, although in the book dimension code - a text has been assumed instead of QR code because of some unavailability of the printing facilities. The version for QR code will be out soon.
    * Homography technique is used, feature detection, choosing the image of the QR code as the selected area.
- **Link**: https://github.com/krshrimali/Dimensionality-Calculation-using-Homography-Matrix-and-QR-Code-Detection

[**Implementation of No Reference Image Quality Assessment using BRISQUE**](https://github.com/krshrimali/No-Reference-Image-Quality-Assessment-using-BRISQUE-Model) (C++ and Python)

<img src="https://raw.githubusercontent.com/krshrimali/No-Reference-Image-Quality-Assessment-using-BRISQUE-Model/master/Images/Table_Comparison_BRISQUE.png"></img>

Implementation of NR IQA Method (BRISQUE) in OpenCV using C++ and Python. The project uses LIBSVM and OpenCV libraries. NumPy is used for vectorization.

[**Template Matching, Cartoonification and more**](https://github.com/krshrimali/OpenCV_Work/blob/master/Cartoonifier_Report.pdf) (C++ and Python)

<img src="/assets/blog/cartoonified.PNG">

Implementation of several OpenCV Algorithms like Template Matching and Cartoonification. [Code](https://github.com/krshrimali/OpenCV_Work) available on my GitHub.

[**Deep Learning based Edge Detection**](https://github.com/krshrimali/Deep-Learning-based-Edge-Detection) (Python)

<img src="https://raw.githubusercontent.com/krshrimali/Deep-Learning-based-Edge-Detection/master/testdata/comparison/output_flowers.png" height="400">

Deep Learning based Edge Detection using OpenCV's HED Implementation.

**Flask based Web App using OpenCV** (Python)

Implementation of OpenCV's Thresholding and Grayscaling on Realtime webcam interface using Flask and OpenCV. Code to be released soon.

[**Panorama Image Stitching using OpenCV**](https://github.com/krshrimali/Panorama-Image-Stitching-using-OpenCV) (Python and C++)

Panorama of two images using OpenCV.

Blogs
====================
[**PyTorch C++ API: Using PyTorch C++ API (VGG-16 Network on MNIST Dataset)**](https://krshrimali.github.io/PyTorch-C++-API/):

I discuss about using PyTorch C++ API for Digit Recognition using MNIST Dataset.

[**What's so special about Gaussian Distribution?**](https://krshrimali.github.io/Understanding-Gaussian-Distribution/):

I discuss about Gaussian Distribution and it's implementation. 

Video: https://www.youtube.com/watch?v=JaGEiePus-E&feature=youtu.be

Guest Blogs
====================
[**Convex Hull using Python and C++**](https://www.learnopencv.com/convex-hull-using-opencv-in-python-and-c/):

In this post, I explain how to find the Convex Hull of a shape (a group of points). I also explained the algorithm and then follow up with C++ and Python code implementation using OpenCV.

[**SVM using Scikit-Learn in Python**](https://www.learnopencv.com/svm-using-scikit-learn-in-python/):

This post explains the implementation of Support Vector Machines (SVMs) using Scikit-Learn library in Python.

[**Average Faces of FIFA World Cup 2018**](https://www.learnopencv.com/average-faces-of-fifa-world-cup-2018/):

<img src="https://www.learnopencv.com/wp-content/uploads/2018/06/fifa-players-with-country-names.png">Average Faces Generated</img>

[**Image Quality Assessment using BRISQUE**](https://www.learnopencv.com/image-quality-assessment-brisque/):

<img src="https://www.learnopencv.com/wp-content/uploads/2018/06/workflow-brisque-iqa.png">
