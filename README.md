# Image_Segmentation_Deep_Learning
Deep learning method for Image segmentation

The dataset1 can be downloaded from: 

https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing

Segmentation images are generated by 'labelme' which can be downloaded from: 

https://github.com/wkentaro/labelme

For label Images generation, please refer to cv_test.py file.

The problem of 'labelme':

1. 对不同的图片进行标定时，颜色标签都是从头开始的 (在生成标签图片时需要特别小心)

2. 每一张图片标定结束后都单独生成了文件夹，无法批量处理文件

3. labelme 程序依赖于imgviz 库 （这个库需要一同下载才能正常使用labelme）

imgviz 库的使用可参考： 

https://github.com/wkentaro/imgviz

The network architecture used in train.py and predict.py files is UNet. In the future I will implement UNet++ and PSPNet for performance comparison.

The loss function is Cross Entropy, but Focal Loss is recommended.

Focal loss can be availabe from:

https://github.com/umbertogriffo/focal-loss-keras  (My gihub also fork this loss function please check)


If you have any questions, please send email to me (xingshuli600@gmail.com).




