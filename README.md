# python_cnn
implement convolution neural network with python

一：项目组成简单介绍：
cnn_convolution.py
卷积层类
cnn_down_sample.py
下采样层类
cnn_full_connect.py
全连接层类（也可以做为独立的神经网络模型）
activators.py
激活函数
self_conv2d.py
自定义的卷积操作函数
cnn_test.py
卷积神经网络的训练测试（有点儿乱，还在调试中，随后进行改进）

二：目的：
1：主要为了自己的理解，这个是最首要的目的
2：稍微熟悉下python
3: 网上大多是matlab，c++的实现代码

三：卷积神经网络学习回顾：
1：网络上对各种CNN框架(LENET-5, ALEXNET, VGG)的介绍，熟悉术语，定义
2：tensorflow中实现简单的例子
3：阅读CNN的相关论文，深入理解其中的细节
4：阅读网上CNN实现的代码，主要是matlab代码
5：自己用python实现CNN

四：说明：
1：我在写的过程中，参考了下面网址的博主的博客：
https://www.cnblogs.com/charlotte77/p/7783261.html
2：我在博主的基础上实现了一个完整的CNN,包括卷积层，池化层和全连接层
3：公式来自论文《Notes on Convolutional Neural Networks》

五：运行的一些限制条件：
1：单个图片数据输入（开始的时候只想着按照算法实现，没考虑图片批量输入，后来用了一些笨方法，实现了批量图片输入）
2：我自己实现了一个python版的conv2d（matlab标准）函数 self_conv2d(input_array, kernel_array, str_shape)，str_shape("full","valid","same"),
  因此限制了自由的pading和stride取值
3：中间层feature_map与其后的卷积核的操作，为了简单操作，我采用了全连接形式
4：池化层仅支持(2 2)的max_pool

六：改进：
1：实现批量输入图片
2：实现feature_map贡献自学习
3：支持多种类的池化操作

七：学习交流
1：邮箱：baishiruyue@163.com， 
   qq:1032304268




