---
layout: post
title: "What's so special about Gaussian Distribution?"
date: 2019-02-14
desc: "This blog post talks about Gaussian Distribution, it's usage and
theory."
keywords: "Blog, Distributions, Gaussian"
categories: [Blog, Theory]
tags: [Blog, Theory]
icon: icon-html
---

If you have ever studied Linear Regression before, you would have heard about
the assumption it makes about the error - for it to be following Gaussian
Distribution. In this blog, I talk about Gaussian Distribution and answer some
of very important questions.

[![Understanding Gaussian Distribution and it's
Visualization](https://img.youtube.com/vi/JaGEiePus-E/0.jpg)](https://www.youtube.com/watch?v=JaGEiePus-E&feature=youtu.be)

Let's look at the code for Visualizing Gaussian Distribution:

**Step - 1: Import Necessary Libraries**


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

**Step - 2: Generate Normal Distribution**


```python
# define parameters for Normal Distribution
# Here: Standard Normal Distribution
mean, standard_deviation = 0.0, 1.0
```


```python
data = np.random.normal(mean, standard_deviation, 1000)
```

**Step - 3: Draw the probability density function**


```python
print(mean - np.mean(data))
print(standard_deviation - np.std(data, ddof=1))
```

    0.04169077132822318
    -0.000940436245006504



```python
# find freq of each point
counts, bins, patches = plt.hist(data, 30, normed=True) # density = True
```

    /home/kushashwa/.local/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "



![png](https://raw.githubusercontent.com/krshrimali/krshrimali.github.io/master/_posts/data/output_7_1.png)



```python
counts_before, bins, _p = plt.hist(data, 30, density=False)
counts_after, bins, _p = plt.hist(data, 30, density=True)
plt.plot(bins, \
         1/np.sqrt(2 * np.pi * standard_deviation**2) * np.exp((-1 * (bins - mean)**2)/(2 * standard_deviation**2)), \
        linewidth=2, color='r')
```

![png](https://raw.githubusercontent.com/krshrimali/krshrimali.github.io/master/_posts/data/output_8_1.png)


```python
counts_after, bins, _p = plt.hist(data, 30, density=True)
plt.plot(bins, \
         1/np.sqrt(2 * np.pi * standard_deviation**2) * np.exp((-1 * (bins - mean)**2)/(2 * standard_deviation**2)), \
        linewidth=2, color='r')
```

![png](https://raw.githubusercontent.com/krshrimali/krshrimali.github.io/master/_posts/data/output_9_1.png)


```python
print(help(plt.hist))
```
        
        density : boolean, optional
            If ``True``, the first element of the return tuple will
            be the counts normalized to form a probability density, i.e.,
            the area (or integral) under the histogram will sum to 1.
            This is achieved by dividing the count by the number of
            observations times the bin width and not dividing by the total
            number of observations. If *stacked* is also ``True``, the sum of
            the histograms is normalized to 1.
        
            Default is ``None`` for both *normed* and *density*. If either is
            set, then that value will be used. If neither are set, then the
            args will be treated as ``False``.
        
            If both *density* and *normed* are set an error is raised.

```python
print(counts_before) # when density=False
```

    [ 1.  0.  8. 10. 19. 20. 35. 38. 51. 54. 60. 80. 82. 87. 74. 66. 62. 76.
     49. 39. 28. 21. 15. 15.  2.  4.  2.  1.  0.  1.]



```python
print(counts_after) # when density=True
```

    [0.00492267 0.         0.03938138 0.04922672 0.09353077 0.09845345
     0.17229353 0.18706155 0.25105629 0.26582431 0.29536034 0.39381379
     0.40365913 0.42827249 0.36427775 0.32489637 0.30520568 0.3741231
     0.24121094 0.19198422 0.13783483 0.10337612 0.07384008 0.07384008
     0.00984534 0.01969069 0.00984534 0.00492267 0.         0.00492267]


**Next up** - Talking about Gaussian Distributions' Role in Datasets.

**References**

1. https://www.varsitytutors.com/hotmath/hotmath_help/topics/normal-distribution-of-data
2. https://www.youtube.com/watch?v=iYiOVISWXS4
