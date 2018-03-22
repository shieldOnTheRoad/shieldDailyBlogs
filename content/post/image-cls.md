---
title: "A Survey for Image Classification"
date: 2018-03-21T12:47:36+08:00
lastmod: 2018-03-23T12:47:36+08:00
draft: false
tags: ["image-cls"]
categories: ["Research"]
author: 'shield'

weight: 10

# You can also close(false) or open(true) something for this content.
# P.S. comment can only be closed
# comment: false
# autoCollapseToc: false
# reward: false
toc: false
mathjax: true
---

`图像分类`算法综述，结合`PASCAL VOC`、`ImageNet`两个数据集介绍近20年来有关`图像分类`的相关技术。包含：`手工特征`、`特征学习`两种主要方法。

<!--more-->
<br>
## #1 **概述**
&emsp;&emsp;`图像分类`任务即是识别一张图片是否含有某类物体，有关`图像分类`算法的研究已经有五十多年历史，随着一些重要数据（e.g., `PASCAL VOC`、`ImageNet`）的相继推出，`图像分类`算法有了长足的发展。研究内容主要集中于如何对图像的特征进行描述。一般来说，`图像分类`算法通过`手工特征`或者`特征学习方法`对整幅图像进行全局描述，然后利用分类器判断是否存在某类物体 <sup>[[1]](#ref01)</sup>。

### **1.1 手工特征**
&emsp;&emsp;基于词袋模型（Bag of Words）的图像分类算法是提取`手工特征`的主要算法。图像分类里的词袋模型和文本挖掘中的词袋模型非常相似，通过构建一个通用的词表（图像对应由图像局部特征构建的词表、文本对应由文本词汇构成的词表），统计每一个训练数据在词表中出现的个数或差值等等形成表达特征。基于词袋模型的图像分类算法主要包括以下四个方面：1）底层特征提取；2）特征编码；3）特征汇聚；4）分类器设计。

#### **&emsp;&emsp;1）底层特征提取**<br>
&emsp;&emsp;底层特征是图像分类的第一步，底层特征提取方式有两种：一种是基于兴趣点检测，另一种是采用密集提取的方式。兴趣点检测算法通过某种准则选择具有明确定义的、局部纹理特征比较明显的像素点、边缘、角点、区块等，并且通常能够获得一定的几何不变性，从而可以在较小的开销下得到更有意义的表达，最常用的兴趣点检测算子有Harris角点检测子、Shi-Tomasi角点检测子、FAST（Featuress from Accelerated Segment Test）算子、LoG（Laplacian of Gaussian）、DoG（Difference of Gaussian）等。

&emsp;&emsp;近年来图像分类领域使用更多的则是密集提取的方式，从图像中按固定的步长、尺度提取出大量的局部特征描述，大量的局部描述尽管具有更高的冗余度，但信息更加丰富，后面再使用词袋模型进行有效表达后通常可得到比兴趣点检测更好的性能。常用的局部特征包括SIFT（Scale Invariant Feature Transform，尺度不变特征转换）、HOG（Histogram of Oriented Gradient，方向梯度直方图）等。`PASCAL VOC`竞赛历年最好的图像分类算法都采用了多种特征，采样方式上密集提取与兴趣点检测相结合，底层特征描述也采用了多种特征描述子，这样做的好处是，在底层特征提取阶段，通过提取到大量的冗余特征，最大限度的对图像进行底层描述，防止丢失过多的有用信息，这些底层描述中的冗余信息主要靠后面的特征编码和特征汇聚得到抽象和简并。

#### **&emsp;&emsp;2）特征编码**
&emsp;&emsp;密集提取的底层特征中包含了大量的冗余与噪声，为提高特征表达的鲁棒性，需要使用一种特征变换算法对底层特征进行编码，从而获得更具区分性、更加鲁棒的特征表达，这一步对图像分类的性能具有至关重要的作用，因而大量的研究工作都集中在寻找更加强大的特征编码方法，重要的特征编码算法包括向量量化编码、稀疏编码、局部线性约束编码、VLAD （Vector of Locally Aggregated Descriptors）向量编码、Fisher向量编码等。

#### **&emsp;&emsp;3）特征汇聚**
&emsp;&emsp;空间特征汇聚是特征编码后进行的特征集整合操作，通过对编码后的特征，每一维都取其最大值或者平均值，得到一个紧致的特征向量作为图像的特征表达。这一步得到的图像表达可以获得一定的特征不变性，同时也避免了使用特征集进行图像表达的高额代价。最大值汇聚在绝大部分情况下的性能要优于平均值汇，其在图像分类中使用最为广泛。由于图像通常具有极强的空间结构约束，空间金字塔匹配SPM （Spatial Pyramid Matching）提出将图像均匀分块，然后每个区块里面单独做特征汇聚操作并将所有特征向量拼接起来作为图像最终的特征表达。

#### **&emsp;&emsp;4）分类器设计**
&emsp;&emsp;使用支持向量机等分类器进行分类。从图像提取到特征表达之后，一张图像可以使用一个固定维度的向量进行描述，接下来就是学习一个分类器对图像进行分类．这个时候可以选择的分类器就很多了，常用的分类器有支持向量机、犓近邻、神经网络、随机森林等。基于最大化边界的支持向量机是使用最为广泛的分类器之一，在图像分类任务上性能很好，特别是使用了核方法的支持向量。

### **1.2 特征学习方法**
&emsp;&emsp;最近几年，深度学习模型逐渐成为图像分类的首选算法，其基本思想是通过有监督或者无监督的方式学习层次化的特征表达，来对图像特征进行从底层图像语义到高层图像语义的描述。主流的深度学习模型主要包括自动编码器（Auto Encoder）、受限波尔兹曼机RBM （Restricted Boltzmann Machine）、深度信念网络DBN （Deep Belief Nets）卷积神经网络CNN （Convolutional Neural Networks）。从最近几年超大规模图像分类比赛`ImageNet`的冠军模型可以发现，卷积神经网络正在成为最主流的图像分类算法。2012年Hinton等人提出`AlexNet`，首次颠覆了以`手工特征`为核心的图像分类算法。`ImageNet`图像分类竞赛后续的几年中，相继涌现出分类性能良好的卷积神经网络，比如：`ZFNet`、`VGGNet`、`GoogleNet`、`Resnet`、`SENet`等。

<br>
## #2 **PASCAL VOC图像分类算法**
| 年份      | 底层特征            | 特征编码        | 空间约束  | 分类器    | 融合     |
| :-------: | :---------------:  | :------------: | :-------: |:--------:| :-------:|
| 2005      | 密集 SIFT          | 向量量化        | 无        | 线性SVM   | 特征拼接 |
| 2006      | 兴趣点检测＋密集提取 | 向量量化        | SPM      | 两层核SVM | 两层融合 |
| 2007      | 密集＋兴趣点，多特征 | 向量量化        | SPM      | 核SVM     | 通道加权 |
| 2008      | 密集＋兴趣点，多特征 | 软量化          | SPM      | 多分类器  | 多模型   |
| 2009      | 密集 SIFT          | GMM，LCC        | SPM      | 线性SVM   | 多特征   |
| 2010      | 密集＋兴趣点，多特征 | 向量量化        | SPM，检测 | 多分类器  | 多模型   |
| 2011      | 密集＋兴趣点，多特征 | 向量量化        | SPM，检测 | 多分类器  | 多模型   |
| 2012      | 密集＋兴趣点，多特征 | 向量量化，Fisher | SPM，检测 | 多分类器 | 多模型   |


<br>
## #3 **ImageNet图像分类算法**
| 年份      | 模型               | TOP5-ACC        | 备注  |
| :-------: | :---------------: | :---------: | :-------: |
| 2010      | 密集 HOG LBP+SVM  | 71.8%       |           |
| 2011      | Fisher+线性SVM    | 74.2%       |           |
| 2012      | AlexNet <sup>[[2]](#ref02)</sup>    | 84.6%       | ReLU、image flip、patch extractions、dropout       |
| 2013      | ZFNet <sup>[[3]](#ref03)</sup>      | 88.8%       | similar architecture to AlexNet、7x7 first filter  |
| 2014      | VGGNet <sup>[[4]](#ref04)</sup>     | 92.7%       | 3x3 sized filters、<font color="red">*Leaderboard Ranking 2*</font> |
| 2014      | GoogleNet <sup>[[5]](#ref05)</sup>  | 93.3%       | Inception modules                                  |
| 2015      | ResNet <sup>[[6]](#ref06)</sup>     | 96.4%       | residual block、Ultra-deep                          |

<br>
## #4 **关键技术**
&emsp;&emsp;`HOG` `SIFT` `BOF` `VLAD` `LeNet` `AlexNet` `VGGNet` `GoogleNet` `ResNet`

### **4.1 HOG** <sup>[[转载]](#ref07)</sup>
&emsp;&emsp;Histogram of oriented gradient, 梯度直方图。将图像分为小的细胞单元(cells)，每个细胞单元计算一个梯度方向(或边缘方向)直方图。为了对光照和阴影有更好的不变性，需要对直方图进行对比度归一化。将检测窗口中的所有块的HOG描述子组合起来就形成了最终的特征向量。
#### **&emsp;&emsp;1）灰度化**<br>
&emsp;&emsp;Hog特征提取的是纹理特征，颜色信息不起作用，所以现将彩色图转为灰度图。

#### **&emsp;&emsp;2）标准化**<br>
&emsp;&emsp;为了提高检测器对光照等干扰因素的鲁棒性，需要对图像进行Gamma校正，以完成对整个图像的归一化，目的是调节图像的对比度，降低局部光照和阴影所造成的影响，同时也可以降低噪音的干扰；（当r取1/2时，像素的取值范围就从0~255变换到0~15.97）。

#### **&emsp;&emsp;3）计算像素梯度**<br>
&emsp;&emsp;计算图像像素的梯度：根据下面的公式计算每个像素的水平方向和竖直方向的梯度，并计算每个像素位置的梯度大小和方向。图像在像素点（x,y）处的水平方向和垂直方向的梯度。随后利用所得梯度计算像素点（x,y）处的梯度幅值和梯度方向。
$$ G\_{x}(x, y) = G(x+1, y) - G(x-1, y) $$
$$ G\_{y}(x, y) = G(x, y+1) - G(x, y-1) $$
<br>
$$ \nabla G(x, y) = \sqrt{G\_{x}(x, y)^{2}+G\_{y}(x, y)^{2}} $$
$$ \theta (x, y) = tan^{-1}\frac{G\_{y}(x, y)}{G\_{x}(x, y)} $$

#### **&emsp;&emsp;4）统计cell直方图**<br>
&emsp;&emsp;将图像划分成小的Cell，将梯度方向映射到180度的范围内，将像素的梯度幅值作为权值进行投影，用梯度方向决定向哪一维进行投影，假如该像素的梯度方向为20度，梯度幅值为10，那么直方图的第二维就加10。下图是一个细胞单元内的方向梯度直方图，角度分辨率是在180度的范围内，以20度等分，即一个细胞单元的HOG特征是一个9维的向量。
{{% figure class="center" src="https://upload-images.jianshu.io/upload_images/3584856-6627b9fc1f097e4e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/345" title="一个细胞单元的梯度方向直方图" %}}

#### **&emsp;&emsp;5）统计block直方图**<br>
&emsp;&emsp;统计每个细胞单元内的梯度直方图，形成每个细胞单元的描述子，由cell组成更大的描述子，称为块，将一个块内四个cell的特征向量串联起来就构成了该块的梯度方向直方图，按照一个细胞单元是9维的HOG特征，则一个块的HOG特征为4x9=36维。由于局部光照的变化，以及前景背景对比度的变化，使得梯度强度的变化范围非常大，这就需要对梯度做局部对比度归一化。这里的策略是针对每个块进行对比度归一化，一般使用L2-norm。

#### **&emsp;&emsp;6）统计window直方图**<br>
&emsp;&emsp;只需要将窗口内所有块的Hog特征向量串联起来就得到了Window的HOG特征。

#### **&emsp;&emsp;7）统计图像直方图**<br>
&emsp;&emsp;一幅图像可以无重叠的划分为多个Window，这时将所有Window的特征向量串联起来就是整幅图像的HOG特征了，如果Window的大小和图像的大小相同，那么Window的HOG特征就是整幅图像的HOG特征。

### **&emsp;&emsp;`NOTE THAT !!!`**<br>
&emsp;&emsp;*Cell、Block、stride、细胞单元、块、步长、窗口、图像、Hog特证有什么联系？*

&emsp;&emsp;其实Cell就是细胞单元，Block指的就是块，Stride是步长，为了方便理解，下面给出了窗口、块、细胞单元、步长的相互关系图。

{{% figure class="center" src="https://upload-images.jianshu.io/upload_images/3584856-69d37207b9fa542f.png?imageMogr2/auto-orient/" title="图像、窗口、块、细胞单元、步长的关系图" %}}

&emsp;&emsp;可以看到4个相邻Cell就组成了一个Block（左上角所示），接下来Block在图像上按照一定stride（黄色箭头），按照从左至右、从上至下的顺序进行滑动对图像进行遍历，通过第4步的计算我们就可以得到一个Cell的方向梯度直方图了，那么将四个Cell的方向梯度直方图进行串联，就组成了一个Block的梯度方向直方图，接下来该Block对Window进行遍历，是不是就有许多新的Block了（不断地会有4个相邻的Cell组成新的Block），最后我们将所有Block的方向梯度直方图串联起来就是Window的Hog特征啦，一幅图像可以当作一个窗口，也可以无重叠的划分为多个不重叠的窗口，如果是这种情况，将所有Window的特征串联起来就是图像的HOG特征了。记住我们最后得到的是一个行向量（列向量）。

### **4.2 SIFT**
&emsp;&emsp;`coming soon...`

<br>
## # **参考资料**
1. <a id="ref01">[图像物体分类与检测算法综述](http://cjc.ict.ac.cn/online/cre/hkq-2014526115913.pdf)</a>
1. <a id="ref02">[Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)</a>
1. <a id="ref03">[Zeiler, M. D., & Fergus, R. (2014, September). Visualizing and understanding convolutional networks. In European conference on computer vision (pp. 818-833). Springer, Cham.](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53)</a>
1. <a id="ref04">[Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.](http://arxiv.org/pdf/1409.1556v6.pdf)</a>
1. <a id="ref05">[Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov & Rabinovich, A. (2015, June). Going deeper with convolutions. Cvpr.](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)</a>
1. <a id="ref06">[He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).](https://arxiv.org/pdf/1512.03385v1.pdf)</a>
1. <a id="ref07">[方向梯度直方图（HOG）](https://www.jianshu.com/p/6f69c751e9e7)</a>
1. <a id="ref08">[The 9 Deep Learning Papers You Need To Know About (Understanding CNNs Part 3)](https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)</a>

<br>