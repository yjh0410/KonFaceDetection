# KonFaceDetection
I love HTT！

# Dataset
I build a very small train dataset with 549 pictures and a test dataset with 10 pictures.

You can download it from BaiDuYunDisk:

Link：https://pan.baidu.com/s/1jdh9pcZKZ_ifs435IB3VhQ 
Password：nhrw

You will find them in ```dataset/``` and get two zip files ```dataset/train.zip``` and ```dataset/test.zip``. Just unzip them.

Someday, I will upload them to GoogleCloud. 

# Model
It is just a simple CenterNet. I don't wanna say too much. Just check ```models/centerface.py```

# Weight
I love HTT！

You can download the trained model from BaiDuYunDisk：

Link：https://pan.baidu.com/s/1jdh9pcZKZ_ifs435IB3VhQ 
Password：nhrw

You will find them in ```weights/``` and get two files ```CenterFace/CenterFace.pth``` and ```pretrained/CenterFace.pth```.

Attention, the ```pretrained/CenterFace.pth``` is trained on widerface which means it is a pretrained model as I have only 549 images.
It is too hard to train from scratch.

# Detection
I visualize the detection results on test images:

![Image](https://github.com/yjh0410/KonFaceDetection/blob/main/img_files/0.jpg)
![Image](https://github.com/yjh0410/KonFaceDetection/blob/main/img_files/1.jpg)
![Image](https://github.com/yjh0410/KonFaceDetection/blob/main/img_files/2.jpg)
![Image](https://github.com/yjh0410/KonFaceDetection/blob/main/img_files/3.jpg)
![Image](https://github.com/yjh0410/KonFaceDetection/blob/main/img_files/4.jpg)
![Image](https://github.com/yjh0410/KonFaceDetection/blob/main/img_files/5.jpg)
![Image](https://github.com/yjh0410/KonFaceDetection/blob/main/img_files/6.jpg)
![Image](https://github.com/yjh0410/KonFaceDetection/blob/main/img_files/7.jpg)
![Image](https://github.com/yjh0410/KonFaceDetection/blob/main/img_files/8.jpg)
![Image](https://github.com/yjh0410/KonFaceDetection/blob/main/img_files/9.jpg)

The five girls are all cute and pretty, right?

So, enjoy it.

# Train
```Shell
python train.py --cuda -ms --mosaic -r
```

```--cuda``` means you use gpu to train it.

```-ms``` means you use multi scale trick.

```--mosaic``` means you use mosaic augmentation trick.

# Test
```Shell
python test.py --cuda --trained_model [select a model file]
```

# Eval
```Shell
python eval.py --cuda --trained_model [select a model file]
```
I don't have a test labels, so it is useless.