---
layout: post
title: "Setting up Jupyter Notebook (Xeus Cling) for Libtorch and OpenCV Libraries"
date: 2019-08-19
desc: "In this blog, we discuss setting up Xeus Cling for Libtorch and OpenCV Libraries"
keywords: "Blog, PyTorch, C++, Xeus Cling, Jupyter, Notebook, Libtorch, OpenCV"
categories: [Blog, PyTorch, OpenCV]
tags: [Blog, PyTorch, OpenCV]
blog: [OpenCV]
published: false
excerpt_separator: <!--more-->
images:
  - url: /assets/Cover-Xeus-Cling.jpg
icon: icon-html
---

##  Introduction to Xeus Cling

Today, we are going to run our C++ codes in the Jupyter Notebook. Sounds ambitious? Not much. Let's see how we do it using Xeus Cling. 

I'll quote the definition of Xeus Cling on the official <a href="https://xeus-cling.readthedocs.io/en/latest/#targetText=xeus%2Dcling%20is%20a%20Jupyter,of%20the%20Jupyter%20protocol%20xeus.">documentation website.</a>

> xeus-cling is a Jupyter kernel for C++ based on the C++ interpreter cling and the native implementation of the Jupyter protocol xeus.

Just like we use Python Kernel in the Jupyter Notebook, we can also use a C++ based interpreter cling combined with a Jupyter protocol called Xeus to reach closer to implementing C++ code in the notebook.

## Installing Xeus Cling using Anaconda

It's pretty straight forward to install Xeus Cling using Anaconda. I'm assuming the user has Anaconda installed.

`conda install -c conda-forge xeus-cling`

The conventional way to install any such library which can create conflicts with existing libraries, is to create an environment and then install it in the environment.

```
conda create -n cpp-xeus-cling
source activate cpp-xeus-cling

conda install -c conda-forge xeus-cling
```

Once setup, let's go ahead and get started with Jupyter Notebook. When creating a new notebook, you will see different options for the kernel. One of them would be `C++XX` where XX is the `C++` version. 

<img src="/assets/kernels-available.png"/>

Click on any of the kernel for C++ and let's start setting up environment for PyTorch C++ API.

You can try and implement some of the basic commands in C++.

<img src="/assets/Jupyter-Notebook-Sample.png"/>

This looks great, right? Let's go ahead and set up the Deep Learning environment.

## Setting up Libtorch in Xeus Cling

Just like we need to give path to `Libtorch` libraries in `CMakeLists.txt` or while setting up `XCode` (for OS X users) or `Visual Studio` (for Windows Users), we will also load the libraries in Xeus Cling.

We will first give the `include_path` of Header files and `library_path` for the libraries. We will also do the same for `OpenCV` as we need it to load images.

```cpp
#pragma cling add_library_path("/Users/krshrimali/Downloads/libtorch/lib/")
#pragma cling add_include_path("/Users/krshrimali/Downloads/libtorch/include/torch/csrc/api/include/")
#pragma cling add_library_path("/usr/local/Cellar/opencv/4.1.0_2/lib")
#pragma cling add_include_path("/usr/local/Cellar/opencv/4.1.0_2/include/opencv4")
```

For OS X, the libtorch libraries will be in the format of `.dylib`. Ignore the `.a` files as we only need to load the `.dylib` files. Similarly for Linux, load the libraries in `.so` format located in the `lib/` folder.

**For Mac**

```cpp
#pragma cling load("/Users/krshrimali/Downloads/libtorch/lib/libiomp5.dylib")
#pragma cling load("/Users/krshrimali/Downloads/libtorch/lib/libmklml.dylib")
#pragma cling load("/Users/krshrimali/Downloads/libtorch/lib/libc10.dylib")
#pragma cling load("/Users/krshrimali/Downloads/libtorch/lib/libtorch.dylib")
#pragma cling load("/Users/krshrimali/Downloads/libtorch/lib/libcaffe2_detectron_ops.dylib")
#pragma cling load("/Users/krshrimali/Downloads/libtorch/lib/libcaffe2_module_test_dynamic.dylib")
#pragma cling load("/Users/krshrimali/Downloads/libtorch/lib/libcaffe2_observers.dylib")
#pragma cling load("/Users/krshrimali/Downloads/libtorch/lib/libshm.dylib")
```

**For Linux**

```cpp
#pragma cling load("/opt/libtorch/lib/libc10.so")
#pragma cling load("/opt/libtorch/lib/libcaffe2_detectron_ops.so")
#pragma cling load("/opt/libtorch/lib/libcaffe2_module_test_dynamic.so")
#pragma cling load("/opt/libtorch/lib/libgomp-4f651535.so.1")
#pragma cling load("/opt/libtorch/lib/libtorch.so")
```

For OpenCV, the list of libraries is long.

**For Mac**

```cpp
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_datasets.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_aruco.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_bgsegm.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_bioinspired.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_calib3d.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_ccalib.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_core.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_dnn_objdetect.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_dnn.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_dpm.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_face.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_features2d.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_flann.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_freetype.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_fuzzy.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_gapi.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_hfs.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_highgui.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_img_hash.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_imgcodecs.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_imgproc.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_line_descriptor.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_ml.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_objdetect.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_optflow.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_phase_unwrapping.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_photo.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_plot.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_quality.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_reg.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_rgbd.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_saliency.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_sfm.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_shape.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_stereo.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_stitching.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_structured_light.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_superres.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_surface_matching.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_text.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_tracking.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_video.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_videoio.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_videostab.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_xfeatures2d.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_ximgproc.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_xobjdetect.4.1.0.dylib")
#pragma cling load("/usr/local/Cellar/opencv/4.1.0_2/lib/libopencv_xphoto.4.1.0.dylib")
```

**For Linux**

```cpp
#pragma cling load("/usr/local/lib/libopencv_aruco.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_bgsegm.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_bioinspired.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_calib3d.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_ccalib.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_core.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_datasets.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_dnn_objdetect.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_dnn.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_dpm.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_face.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_features2d.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_flann.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_freetype.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_fuzzy.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_gapi.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_hdf.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_hfs.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_highgui.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_imgcodecs.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_img_hash.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_imgproc.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_line_descriptor.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_ml.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_objdetect.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_optflow.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_phase_unwrapping.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_photo.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_plot.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_reg.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_rgbd.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_saliency.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_sfm.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_shape.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_stereo.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_stitching.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_structured_light.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_superres.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_surface_matching.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_text.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_tracking.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_videoio.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_video.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_videostab.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_xfeatures2d.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_ximgproc.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_xobjdetect.so.4.1.0")
#pragma cling load("/usr/local/lib/libopencv_xphoto.so.4.1.0")
```

Once done, run the cell and that's it. We have successfully setup the environment for `Libtorch` and `OpenCV`.

## Testing Xeus Cling Notebook

Let's go ahead and include the libraries.

<img src="/assets/Include-Libraries.png"/>

After successfully importing libraries, we can define functions, write code and use the utilities Jupyter provides. For testing, I have implemented Transfer Learning example that we discussed in the <a href="https://krshrimali.github.io/Applying-Transfer-Learning-Dogs-Cats/">previous blog</a>. 

<img src="/assets/Training-Image.png"/>

With this blog, I'm also happy to share a Notebook file with implementation of Transfer Learning using ResNet18 Model on Dogs vs Cats Dataset. The source code for Transfer Learning and the Notebook file can be found <a href="https://github.com/krshrimali/Transfer-Learning-Dogs-Cats-Libtorch.git">here</a>.

## References

1. <a href="https://www.learnopencv.com/xeus-cling-run-c-code-in-jupyter-notebook/">Xeus-Cling: Run C++ code in Jupyter Notebook by Vishwesh Ravi Shrimali</a>.
2. <a href="https://xeus-cling.readthedocs.io/en/latest">Documentation of Xeus Cling</a>.
