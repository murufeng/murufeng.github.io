本篇博文主要讲YOLOv1-YOLOv3的进化历程。
### YOLOv1
#### 1. 介绍
论文名称:You only look once unified real-time object detection
[论文链接](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)

#### 2. 基本思想
YOLOv1是典型的目标检测one stage方法，在YOLO算法中，***核心思想*** 就是把物体检测（object detection）问题处理成回归问题，用一个卷积神经网络结构就可以从输入图像直接预测bounding box和类别概率。用回归的方法去做目标检测，执行速度快，达到非常高效的检测，其背后的原理和思想也非常简单。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190828214633854.JPG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21yamt6aGFuZ21h,size_16,color_FFFFFF,t_70)

如上图所示，YOLOv1的算法思想就是把一张图片，首先reshape成448x448大小（由于网络中使用了全连接层，所以图片的尺寸需固定大小输入到CNN中），然后将划分成SxS个单元格（原文中S=7），以每个格子所在位置和对应内容为基础，来预测：

- 1）检测框，包含物体框中心相对其所在网格单元格边界的偏移（一般是相对于单元格左上角坐标点的位置偏移，以下用x，y表示）和检测框真实宽高相对于整幅图像的比例（注意这里w，h不是实际的边界框宽和高），每个格子预测B个检测框(原文中是2），如下图；
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190828222655640.JPG)

- 2）每个框的Confidence，这个confidence代表了预测框含有目标的置信度和这个预测框预测的有多准2重信息，公式和说明如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190828222849117.JPG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21yamt6aGFuZ21h,size_16,color_FFFFFF,t_70)

- 3）每个格子预测一共C个类别的概率分数，并且这个分数和物体框是不相关的，只是基于这个格子。
- 
**注意（重要细节）:** 
- x，y，w，h，confidence都被限制在区间[0,1]。
- 置信度confidence值只有2种情况，要么为0（边界框中不含目标，P(object)=0），要么为预测框与标注框的IOU，因为P(Object)只有0或1，两种可能，有目标的中心落在格子内，那么P(object)=1，否则为0，不存在（0，1）区间中的值。其他论文中置信度的定义可能跟YOLOv1有些不同，一般置信度指的是预测框中是某类别目标的概率，在[0,1]之间。
- 每个格子预测C个类别的概率分数，而不是每个每个检测框都需要预测C个类别的概率分数。


**具体实现见图：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190828223008439.JPG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21yamt6aGFuZ21h,size_16,color_FFFFFF,t_70)

综上所述：每个格子需要输出的信息维度是Bx(4+1)+C=Bx5+C。在YOLO的论文中，S=7，B=2，C是PASCAL VOC的类别数20，所以最后得到的关于预测的物体信息是一个7x7x30的张量。最后从这7x7x30的张量中提取出来的预测框和类别预测的信息经过NMS，就得到了最终的物体检测结构。


#### 3. 网络结构
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019082822342490.JPG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21yamt6aGFuZ21h,size_16,color_FFFFFF,t_70)
网络方面主要采用GoogLeNet，卷积层主要用来提取特征，全连接层主要用来预测类别概率和坐标。并且，作者将GoogleNet中的Inception模块换成了1x1卷积后接3x3卷积，最终网络结构由24个卷积层和4个最大池化层和2个全连接层组成。

***两个小细节***：1、作者先在ImageNet数据集上预训练网络，而且网络只采用fig3的前面20个卷积层，输入是224*224大小的图像。然后在检测的时候再加上随机初始化的4个卷积层和2个全连接层，同时输入改为更高分辨率的448*448。2、Relu层改为pRelu，即当x<0时，激活值是0.1*x，而不是传统的0。

### 4. 损失函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190828224413644.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21yamt6aGFuZ21h,size_16,color_FFFFFF,t_70)
**损失函数设计细节：**
- YOLOv1对位置误差，confidence误差，分类误差均使用了均方差作为损失函数。
- 三部分误差损失（位置误差，confidence误差，分类误差），在损失函数中所占权重不一样，位置误差权重系数最大，为5。
- 由于一副图片中没有目标的网格占大多数，有目标的网格占少数，所以损失函数中对没有目标的网格中预测的bbox的confidence误差给予小的权重系数，为0.5。
- 有目标的网格中预测的bbox的confidence损失和分类损失，权重系数正常为1。
- 由于相同的位置误差对大目标和小目标的影响是不同的，相同的偏差对于小目标来说影响要比大目标大，故作者选择将预测的bbox的w,h先取其平方根，再求均方差损失。
- 一个网格预测2个bbox，在计算损失函数的时候，只取与ground truth box中IoU大的那个预测框来计算损失。
- 分类误差，只有当单元格中含有目标时才计算，没有目标的单元格的分类误差不计算在内。

***Loss Function公式详细介绍:***

在原文中，如下图所示:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190828224638844.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21yamt6aGFuZ21h,size_16,color_FFFFFF,t_70)

**在loss function中**，前面两行表示localization error(即坐标误差)，第一行是box中心坐标(x,y)的预测，第二行为宽和高的预测。这里注意用宽和高的开根号代替原来的宽和高，这样做主要是因为相同的宽和高误差对于小的目标精度影响比大的目标要大。举个例子，原来w=10，h=20，预测出来w=8，h=22，跟原来w=3，h=5，预测出来w1，h=7相比，其实前者的误差要比后者小，但是如果不加开根号，那么损失都是一样：4+4=8，但是加上根号后，变成0.15和0.7。 

第三、四行表示bounding box的confidence损失，就像前面所说的，分成grid cell包含与不包含object两种情况。这里注意下因为每个grid cell包含两个bounding box，所以只有当ground truth 和该网格中的某个bounding box的IOU值最大的时候，才计算这项。

第五行表示预测类别的误差，注意前面的系数只有在grid cell包含object的时候才为1。

***所以具体实现的时候是什么样的过程呢？***

**训练的时候：** 输入N个图像，每个图像包含M个objec，每个object包含4个坐标（x，y，w，h）和1个label。然后通过网络得到7*7*30大小的三维矩阵。每个1*30的向量前5个元素表示第一个bounding box的4个坐标和1个confidence，第6到10元素表示第二个bounding box的4个坐标和1个confidence。最后20个表示这个grid cell所属类别。注意这30个都是预测的结果。然后就可以计算损失函数的第一、二 、五行。至于第二三行，confidence可以根据ground truth和预测的bounding box计算出的IOU和是否有object的0,1值相乘得到。真实的confidence是0或1值，即有object则为1，没有object则为0。 这样就能计算出loss function的值了。

**测试的时候**：输入一张图像，跑到网络的末端得到7*7*30的三维矩阵，这里虽然没有计算IOU，但是由训练好的权重已经直接计算出了bounding box的confidence。然后再跟预测的类别概率相乘就得到每个bounding box属于哪一类的概率。

#### 5. YOLOv1的缺点
1、位置精确性差，并且，由于每个单元格只预测2个bbox，然后每个单元格最后只取与gt_bbox的IOU高的那个最为最后的检测框，也只是说每个单元格最多只预测一个目标，若单个单元格有多个目标时，只能检测出其他的一个，导致小目标漏检，因此YOLOv1对于小目标物体以及物体比较密集的检测不是很好。 
2、YOLO虽然可以降低将背景检测为物体的概率，但同时导致召回率较低。
3、由于输出层为全连接层，因此在检测时，YOLO 训练模型只支持与训练图像相同的输入分辨率的图片。

<br>

### YOLOv2
#### 1. 介绍
YOLOv2又叫YOLO9000，其能检测超过9000种类别的物体，在VOC2007数据集中在76FPS的速度下，能达到76.8%的mAP，在40FPS的速度下，能达到78.6%的mAP，很好的达到速度与精度的平衡。

[论文链接](http://xxx.itp.ac.cn/pdf/1612.08242v1.pdf)

<br>
本篇论文是YOLO作者为了改进原有的YOLO算法所写的。由上可知，YOLO主要有两个大缺点：一个缺点在于定位不准确，另一个缺点在于和基于region proposal的方法相比召回率较低。因此YOLOv2主要是要在这两方面做提升。

<br>

####  2. 算法层面
- **Anchor:** 引入了Faster R-CNN中使用的Anchor，注意这里作者在YOLOv2中设计的Anchor并不是像Faster R-CNN中人为事先设计的尺寸和高宽比一级个数，作者通过在所有训练图像的所有边界框上运行k-means聚类来选择锚的个数和形状(k = 5，因此它找到五个最常见的目标形状)。因此，YOLO的锚是特定于您正在训练(和测试)的数据集的。k-means算法找到了将所有数据点划分聚类的方法。这里的数据点是数据集中所有真实边界框的宽度和高度。但5个锚是否最佳选择?我们可以在不同数量的聚类上多次运行k-means，并计算真实标签框与它们最接近的锚框之间的平均IOU。毫无疑问，使用更多质心(k值越大)平均IOU越高，但这也意味着我们需要在每个网格单元中使用更多的检测器，并使模型运行速度变慢。对于YOLO v2，他们选择了5个锚作为召回率和模型复杂度之间的良好折衷。![在这里插入图片描述](https://img-blog.csdnimg.cn/2019082922231368.JPG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21yamt6aGFuZ21h,size_16,color_FFFFFF,t_70)

- **坐标预测:** 在这里作者虽然引入了Faster R-CNN中类似的anchor，但是作者并没有像其意义，对bbox中心坐标的预测是基于anchor坐标的偏移量得到的，而是采用了v1中预测anchor中心点相对于对于单元格左上角位置的偏移，如下图：![在这里插入图片描述](https://img-blog.csdnimg.cn/20190829222349500.JPG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21yamt6aGFuZ21h,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190829222353621.JPG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21yamt6aGFuZ21h,size_16,color_FFFFFF,t_70)


#### 3. 损失函数
1. 在计算类概率误差时，YOLOv1中仅对每个单元格计算；而YOLOv2中对每一个anchor box都会计算类概率误差。
2. YOLOv1中使用w和h的开方来缓和box的尺寸不平衡问题，而在YOLOv2中则通过赋值一个和w，h相关的权重函数达到该目的。
3. 与YOLOv1不同的是修正系数的改变，YOLOv1中no_objects_loss和objects_loss分别是0.5和1，而YOLOv2中则是1和5


#### 4. 网络层面
- **Darknet19:**  与v1不同采用的是全卷积网络，取掉了v1中的全连接层，改用全局平均池化，去掉v1中最后一个池化层，增加特征的分辨率。网络共19个卷积层，5个最大池化层，具体结构见下图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190829222516280.JPG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21yamt6aGFuZ21h,size_16,color_FFFFFF,t_70)


#### 5. 训练检测方面

- **训练图像分辨率：** v1在ImageNet上预训练时用的224x224尺寸的图片，正式训练时用448x448,这需要模型适应新的分辨率。YOLOv2是直接使用448x448的输入训练，随着输入分辨率的增加，模型提高了4%的mAP。
使用了WordTree：通过WordTree来混合检测数据集与识别数据集之中的数据，使得这一网络结构可以实时地检测超过9000种物体分类。
- **联合训练算法**：使用这种联合训练技术同时在ImageNet和COCO数据集上进行训练。YOLO9000进一步缩小了监测数据集与识别数据集之间的代沟。联合训练算法的基本思路就是：同时在检测数据集和分类数据集上训练物体检测器（Object Detectors ），用检测数据集的数据学习物体的准确位置，用分类数据集的数据来增加分类的类别量、提升健壮性。分类信息学习自ImageNet分类数据集，而物体位置检测则学习自COCO检测数据集。
- **多尺度训练**：为了提高模型的鲁棒性，在训练的时候采用了多尺度的输入进行训练，由于网络的下采样因子是32，故输入尺寸选择32的倍数288，352，...，544
- **多尺度检测**，reorg层：作者将前一层的26*26的特征图做一个reorg操作，将其变成13*13但又不破坏其大特征图的特征，然后和本层的13*13的1特征图进行concat。

#### 6. 技巧方面
- **Batch Normalization：** **使用Batch Normalization对网络进行优化，让网络提高了收敛性，同时还消除了对其他形式的正则化（regularization）的依赖。** 通过对YOLO的每一个卷积层增加Batch Normalization，最终使得mAP提高了2%，同时还使model正则化。使用Batch Normalization可以从model中去掉Dropout，而不会产生过拟合。
- **High Resolution Classifier ：** 首先fine-tuning的作用不言而喻，现在基本跑个classification或detection的模型都不会从随机初始化所有参数开始，所以一般都是用预训练的网络来finetuning自己的网络，而且预训练的网络基本上都是在ImageNet数据集上跑的，一方面数据量大，另一方面训练时间久，而且这样的网络都可以在相应的github上找到。 
**原来的YOLO网络在预训练的时候采用的是224 * 224的输入**（这是因为一般预训练的分类模型都是在ImageNet数据集上进行的），**然后在detection的时候采用448 * 448的输入，这会导致从分类模型切换到检测模型的时候，模型还要适应图像分辨率的改变。而YOLOv2则将预训练分成两步：先用224 * 224的输入从头开始训练网络，大概160个epoch（表示将所有训练数据循环跑160次），然后再将输入调整到448*448，再训练10个epoch。注意这两步都是在ImageNet数据集上操作。最后再在检测的数据集上fine-tuning，也就是detection的时候用448*448的图像作为输入就可以顺利过渡了。作者的实验表明这样可以提高几乎4%的MAP。**

- **Convolutional With Anchor Boxes** :  **原来的YOLO是利用全连接层直接预测bounding box的坐标，而YOLOv2借鉴了Faster R-CNN的思想，引入anchor。** 首先将原网络的全连接层和最后一个pooling层去掉，使得最后的卷积层可以有更高分辨率的特征；然后缩减网络，用416*416大小的输入代替原来448*448。**这样做的原因在于希望得到的特征图都有奇数大小的宽和高，奇数大小的宽和高会使得每个特征图在划分cell的时候就只有一个center cell（比如可以划分成7*7或9*9个cell，center cell只有一个，如果划分成8*8或10*10的，center cell就有4个）。为什么希望只有一个center cell呢？因为大的object一般会占据图像的中心，所以希望用一个center cell去预测，而不是4个center cell去预测。网络最终将416*416的输入变成13*13大小的feature map输出，也就是缩小比例为32。** 
我们知道原来的YOLO算法将输入图像分成7*7的网格，每个网格预测两个bounding box，因此一共只有98个box，但是在YOLOv2通过引入anchor boxes，预测的box数量超过了1千（以输出feature map大小为13*13为例，每个grid cell有9个anchor box的话，一共就是13*13*9=1521个，当然由后面第4点可知，最终每个grid cell选择5个anchor box）。顺便提一下在Faster RCNN在输入大小为1000*600时的boxes数量大概是6000，在SSD300中boxes数量是8732。显然增加box数量是为了提高object的定位准确率。 
**作者的实验证明：虽然加入anchor使得MAP值下降了一点（69.5降到69.2），但是提高了recall（81%提高到88%）。**


<br>

### YOLOv3
#### 1.介绍
论文：YOLOv3: An Incremental Improvement 
[论文链接](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

**YOLOv3效果展示:**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190829233102858.JPG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21yamt6aGFuZ21h,size_16,color_FFFFFF,t_70)

 **YOLO算法的基本思想是：** 首先通过特征提取网络对输入图像提取特征，得到一定size的feature map，比如13*13，然后将输入图像分成13*13个grid cell，接着如果ground truth中某个object的中心坐标落在哪个grid cell中，那么就由该grid cell来预测该object，因为每个grid cell都会预测固定数量的bounding box（YOLO v1中是2个，YOLO v2中是5个，YOLO v3中是3个，这几个bounding box的初始size是不一样的），那么这几个bounding box中最终是由哪一个来预测该object？答案是：这几个bounding box中只有和ground truth的IOU最大的bounding box才是用来预测该object的。可以看出预测得到的输出feature map有两个维度是提取到的特征的维度，比如13*13，还有一个维度（深度）是B*（5+C），注：YOLO v1中是（B*5+C），其中B表示每个grid cell预测的bounding box的数量，比如YOLO v1中是2个，YOLO v2中是5个，YOLO v3中是3个，C表示bounding box的类别数（没有背景类，所以对于VOC数据集是20），5表示4个坐标信息和一个置信度（objectness score）。
 
 <br>
 
#### 改进点
- **网络**：Darknet53，一方面基本采用全卷积,另一方面采用简化的residual block 取代了原来 1×1 和 3×3的block; (其实就是加了一个shortcut，也是网络加深必然所要采取的手段)。这和上一点是有关系的，v2的darknet-19变成了v3的darknet-53，为啥呢？就是需要上采样啊，卷积层的数量自然就多了，另外作者还是用了一连串的3*3、1*1卷积，3*3的卷积增加channel，而1*1的卷积在于压缩3*3卷积后的特征表示。Darknet-53只是特征提取层，源码中只使用了pooling层前面的卷积层来提取特征，因此multi-scale的特征融合和预测支路并没有在该网络结构中体现，具体信息可以看源码：https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg。预测支路采用的也是全卷积的结构，其中最后一个卷积层的卷积核个数是255，是针对COCO数据集的80类：3*(80+4+1)=255，3表示一个grid cell包含3个bounding box，4表示框的4个坐标信息，1表示objectness score。模型训练方面还是采用原来YOLO v2中的multi-scale training。
 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190829234128202.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21yamt6aGFuZ21h,size_16,color_FFFFFF,t_70)

- **分类损失**：在YOLOv3中，每个框用多标签分类来预测边界框可能包含的类。该算法将v2中的softmax替换成了逻辑回归loss，在训练过程中使用二原交叉熵损失来进行类别预测。对于重叠的标签，多标签方法可以更好的模拟数据。

- **跨尺度预测**：YOLOv3采用多个尺度融合的方式做预测。原来YOLOv2中有一个层叫：passthrough layer，假设最后提取的特征图尺度是13*13，那么这个层的作用就是将前面一层的26*26的特征图和本层13*13的特征图进行连接，有点像ResNet。这样的操作是为了加强YOLO算法对小目标检测的精度。在YOLOv3中，作者采用了类似与FPN的上采样和融合做法（最后融合了3个尺度，其他2个尺度分别是26*26和52*52），在多给尺度的特征图上做预测，对于小目标的提升效果还是非常明显的。虽然在YOLOv3中每个网格预测3个边界框，比v2中的5个要少，但v3采用了多个尺度的特征融合，所以边界框的数量也比之前多很多。


**Darknet53与其他backbone对比（256×256的图片，并进行单精度测试。运行环境为Titan X）**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190829234625651.JPG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21yamt6aGFuZ21h,size_16,color_FFFFFF,t_70)

**尝试，但效果不好的工作**
- **Anchor box坐标的偏移预测。** 作者尝试了常规的Anchor box预测方法，比如利用线性激活将坐标x、y的偏移程度预测为边界框宽度或高度的倍数。但发现这种做法降低了模型的稳定性，且效果不佳。 用线性方法预测x,y，而不是使用逻辑方法。我们尝试使用线性激活来直接预测x，y的offset，而不是逻辑激活，还降低了mAP。

- **focal loss**。我们尝试使用focal loss，但使我们的mAP降低了2%。 对于focal loss函数试图解决的问题，YOLOv3从理论上来说已经很强大了，因为它具有单独的对象预测和条件类别预测。因此，对于大多数例子来说，类别预测没有损失？或者其他的东西？我们并不完全确定。

- **双IOU阈值和真值分配**。在训练期间，Faster RCNN用了两个IOU阈值，如果预测的边框与ground truth的IoU>0.7，那它是个正样本；如果在[0.3，0.7]之间，则忽略；如果和ground truth的IoU<0.3，那它就是个负样本。作者尝试了这种思路，但效果并不好。

### 总结
YOLO系列算法不断吸收目标检测同类算法的优点，如FPN，Faster-RCNN，ResNet，，将其应用与自身，不断进步，取得了较高的检测速度和检测精度，相比于其他算法更符合工业界对目标检测算法实时性的要求，简单易实现，对于嵌入式很友好，期待下一代的YOLO算法！



### 参考链接
1. [https://blog.csdn.net/u014380165/article/details/72616238](https://blog.csdn.net/u014380165/article/details/72616238)
2. [https://blog.csdn.net/u014380165/article/details/77961414](https://blog.csdn.net/u014380165/article/details/77961414)
3. [https://blog.csdn.net/u014380165/article/details/80202337](https://blog.csdn.net/u014380165/article/details/80202337)
4. [https://blog.paperspace.com/tag/series-yolo/](https://blog.paperspace.com/tag/series-yolo/)
