---
layout: post
title: "Deep Convolutional Generative Adversarial Networks: Review and Implementation using PyTorch C++ API"
date: 2019-09-15
desc: "In this blog, we discuss DCGAN and it's implementation using PyTorch C++ API."
keywords: "Blog, PyTorch, C++, Libtorch, OpenCV"
categories: [Blog, PyTorch, OpenCV]
tags: [Blog, PyTorch, OpenCV]
blog: [PyTorch]
published: false
excerpt_separator: <!--more-->
images:
  - url: /assets/Cover-DCGAN.jpg
icon: icon-html
---

I'm pleased to start a series of blogs on GANs and their implementation with PyTorch C++ API. We'll be starting with one of the initial GANs - DCGANs (Deep Convolutional Generative Adversarial Networks).

<img src="/assets/Cover-DCGAN.jpg"/>

The authors (Soumith Chintala, Radford and Luke Metz) in this Seminal Paper on DCGANs introduced DCGANs to the world like this:

> We introduce a class of CNNs called deep convolutional generative
adversarial networks (DCGANs), that have certain architectural constraints, and
demonstrate that they are a strong candidate for unsupervised learning. Training
on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to
scenes in both the generator and discriminator. Additionally, we use the learned
features for novel tasks - demonstrating their applicability as general image representations.

Even though, the introduction to DCGANs is quite lucid, and here are some points to note:

1. DCGANs are a class of Convolutional Neural Networks.
2. They are a strong candidate for Unsupervised Learning.
3. They are applicable as general image representations as well.

Let's go ahead and see what exactly is DCGAN?

## Introduction to DCGAN

At the time when this paper was released, there was quite a focus on Supervised Learning. The paper aimed at bridging the gap between Unsupervised Learning and Supervised Learning. DCGANs are a way to understand and extract important feature representations from a dataset and generate good image representations by training.

Any Generative Adversarial Network has 2 major components: a Generator and a Discriminator. The tasks for both of them are simple.

1. **Generator**: Generates Images similar to the data distribution such that Discriminator can not distinguish it with the original data.
2. **Discriminator**: Discriminator has a task on accurately distinguishing between the image from the generator and from the data distribution. It basically has to recognize an image as fake or real, correctly.

Both Generator and Discriminator tasks can be represented beautifully with the following equation:

<img src="http://www.sciweavers.org/tex2img.php?eq=%5Cmin_%7BG%7D%5Cmax_%7BD%7DL%28D%2C%20G%29%20%3D%20E_%7Bx%20%20%5Csim%20p_data%28x%29%7D%20%20%5Cbig%28%5Clog%20D%28x%29%20%5Cbig%29%20%20%2B%20E_%7Bx%20%5Csim%20p_z%28x%29%7D%20%5Cbig%28%5Clog%281%20-%20D%28G%28z%29%29%29%5Cbig%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\min_{G}\max_{D}L(D, G) = E_{x  \sim p_data(x)}  \big(\log D(x) \big)  + E_{x \sim p_z(x)} \big(\log(1 - D(G(z)))\big)" width="569" height="29" />

The above equation, shows how the Generator and Discriminator plays min-max game. 

1. The **Generator tries to minimize the loss function.** It follows up with two cases:
	1. **When the data is from the data distribution:** Generator has a task of forcing the Discriminator to predict the data as fake.
	2. **When data is from the Generator:** Generator has a task of forcing the Discriminator to predict the data as real.
2. The **Discriminator tries to maximize the loss function.** It follows up with two cases:
	1. **When the data is from the data distribution:** Discriminator tries to predict the data as real.
	2. **When the data is from the Generator:** Discriminator tries to predict the data as fake.

Fundamentally, the Generator is trying to fool the Discriminator and the Discriminator is trying not to get fooled with. Because of it's analogy, it's also called a police-thief game. (Police is the Discriminator and theif is the Generator).

We have good enough discussion on GANs, to kickstart discussion on DCGANs. Let's go ahead and see what changes they proposed on common CNNs:

Changes in the **Generator**:

1. Spatial Pooling Layers such as MaxPool Layers were replaced with Fractional-Strided Convolutions (a.k.a Transposed Convolutions). This allows the network to learn it's own spatial downsampling, instead of explicitly mentioning the downsampling parameters by Max Pooling.
2. Use BatchNorm in the Generator.
3. Remove Fully Connected layers for deeper architectures.
4. Use ReLU activation function for all the layers except the output layer (which uses Tanh activation function).

Changes in the **Discriminator**:

1. Spatial Pooling Layers such as MaxPool layers were replaced with Strided Convolutions.
2. Use BatchNorm in the Discriminator.
3. Remove FC layers for deeper architectures.
4. Use LeakyReLU activation function for all the layers in the Discriminator.

<img src="/assets/DCGAN-Generator.png"/>
<center>Generator of the DCGAN used for LSUN scene modeling.</center>

As you would note in the above architecture, there is absence of spatial pooling layers and fully connected layers. 

<img src="/assets/DCGAN-Discriminator.png"/>
<center>Discriminator of the DCGAN used for LSUN scene modeling.</center>

Notably again, there are no pooling and fully connected layers (except the last layer).

Let's start with defining the architectures of both Generators and Discriminators using PyTorch C++ API. I used the Object Oriented approach by making class, each for Generator and Discriminator. Note that each of them are a type of CNNs, and also inherit functions (or methods) from `torch::nn::Module` class.

As mentioned before, Generator uses Transposed Convolutional Layers and has no pooling and FC layers. It also uses ReLU Activation Function (except the last layer). The parameters used for the Generator include:

1. `dataroot`: (type: `std::string`) Path of the dataset's root directory. 
2. `workers`: (type: `int`) Having more `workers` will increase CPU memory usage. (Check this link <a href="https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/2">for more details</a>)
3. `batch_size`: (type: `int`) Batch Size to consider. 
4. `image_size`: (type: `int`) Size of the image to resize it to.
5. `nc`: (type: `int`) Number of channels in the Input Image.
6. `nz`: (type: `int`) Length of latent vector, from which the input image is taken.
7. `ngf`: (type `int`) Depth of feature maps carried through the generator.
8. `num_epochs`: (type `int`) Number of epochs for which the model is trained.
9. `lr`: (type `float`) Learning Rate for training. Authors described it to be 0.0002
10. `beta1`: (type: `float`) Hyperparameter for Optimizer used (Adam).
11. `ngpu`: (type: `int`) Number of GPUs available to use. (use `0` if no GPU available)

```cpp
class Generator : public torch::nn::Module {
private:
    std::string dataroot;
    int workers;
    int batch_size;
    int image_size;
    int nc;
    int nz;
    int ngf;
    int num_epochs;
    float lr;
    float beta1;
    int ngpu;
public:
    torch::nn::Sequential main;
    Generator(std::string dataroot_ = "data/celeba", int workers_ = 2, int batch_size_ = 128, int image_size_ = 64, int nc_ = 3, int nz_ = 100, int ngf_ = 64, int ndf_ = 64, int num_epochs_ = 5, float lr_ = 0.0002, float beta1_ = 0.5, int ngpu_ = 0) {
        
        // Set the arguments
        dataroot = dataroot_;
        workers = workers_;
        batch_size = batch_size_;
        image_size = image_size_;
        nc = nc_;
        nz = nz_;
        ngf = ngf_;
        ndf = ndf_;
        num_epochs = num_epochs_;
        lr = lr_;
        beta1 = beta1_;
        ngpu = ngpu_;
        
        main = torch::nn::Sequential(
                                   torch::nn::Conv2d(torch::nn::Conv2dOptions(nz, ngf*8, 4).stride(1).padding(0).with_bias(false).transposed(true)),
                                     torch::nn::BatchNorm(ngf*8),
                                     torch::nn::Functional(torch::relu),
                                     torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf*8, ngf*4, 4).stride(2).padding(1).with_bias(false).transposed(true)),
                                     torch::nn::BatchNorm(ngf*4),
                                     torch::nn::Functional(torch::relu),
                                     torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf*4, ngf*2, 4).stride(2).padding(1).with_bias(false).transposed(true)),
                                     torch::nn::BatchNorm(ngf*2),
                                     torch::nn::Functional(torch::relu),
                                     torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf*2, ngf, 4).stride(2).padding(1).with_bias(false).transposed(true)),
                                     torch::nn::BatchNorm(ngf),
                                     torch::nn::Functional(torch::relu),
                                     torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf, nc, 4).stride(2).padding(1).with_bias(false).transposed(true)),
                                     torch::nn::Functional(torch::tanh)
        );
    }
    
    torch::nn::Sequential main_func() {
    	// Returns Sequential Model of the Generator
        return main;
    }
};
```

Note how we used Transposed Convolution, by passing `.transposed(true)`. 

Similarly, we define the class for Discriminator.

```
class Discriminator : public torch::nn::Module {
private:
    std::string dataroot;
    int workers;
    int batch_size;
    int image_size;
    int nc;
    int nz;
    int ndf;
    int num_epochs;
    float lr;
    float beta1;
    int ngpu;
public:
    torch::nn::Sequential main;
    Discriminator(std::string dataroot_ = "data/celeba", int workers_ = 2, int batch_size_ = 128, int image_size_ = 64, int nc_ = 3, int nz_ = 100, int ngf_ = 64, int ndf_ = 64, int num_epochs_ = 5, float lr_ = 0.0002, float beta1_ = 0.5, int ngpu_ = 1) {
        
        dataroot = dataroot_;
        workers = workers_;
        batch_size = batch_size_;
        image_size = image_size_;
        nc = nc_;
        nz = nz_;
        ngf = ngf_;
        ndf = ndf_;
        num_epochs = num_epochs_;
        lr = lr_;
        beta1 = beta1_;
        ngpu = ngpu_;
        
        main = torch::nn::Sequential(
                                                           torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, ndf, 4).stride(2).padding(1).with_bias(false)),
                                                           torch::nn::Functional(torch::leaky_relu, 0.2),
                                                           
                                                           torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf, ndf*2, 4).stride(2).padding(1).with_bias(false)),
                                                           
                                                           torch::nn::BatchNorm(ndf*2),
                                                           
                                                           torch::nn::Functional(torch::leaky_relu, 0.2),
                                                           
                                                           torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*2, ndf*4, 4).stride(2).padding(1).with_bias(false)),
                                                           
                                                           torch::nn::BatchNorm(ndf*4),
                                                           
                                                           torch::nn::Functional(torch::leaky_relu, 0.2),
                                                           torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*4, ndf*8, 4).stride(2).padding(1).with_bias(false)),
                                                           
                                                           torch::nn::BatchNorm(ndf*8),
                                                           
                                                           torch::nn::Functional(torch::leaky_relu, 0.2),
                                                           
                                                           torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*8, 1, 4).stride(1).padding(0).with_bias(false)),
                                                           
                                                           torch::nn::Functional(torch::sigmoid)
                                                           );
    }
    
    torch::nn::Sequential main_func() {
        return main;
    }
};
```

We can initialize these networks as shown below:

```cpp
// Uses default arguments if no args passed
Generator gen = Generator()
Discriminator dis = Discriminator()

torch::nn::Sequential gen_model = gen.main_func()
torch::nn::Sequential dis_model = dis.main_func()
```

In case you are using a GPU, you can convert the models:

```cpp
torch::Device device = torch::kCPU;
if(torch::cuda::is_available()) {
    device = torch::kCUDA;
}

gen_model->to(device);
dis_model->to(device);
```

**Note on Data Loading**: In the past blogs, I've discussed on loading custom data. Please refer to the previous blogs for a quick review on loading data.

Let's go ahead and define optimizers and train our model. We use the parameters defined by the authors, for optimizer (Adam, `beta` = 0.5) and learning rate of `2e-4`.

```cpp
torch::optim::Adam gen_optimizer(gen_model->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
torch::optim::Adam dis_optimizer(dis_model->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
```