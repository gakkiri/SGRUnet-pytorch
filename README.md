# Anime Sketch Coloring with Swish-Gated Residual U-Net
Pytorch unofficial port of SGRUnet(the official: [here](https://github.com/pradeeplam/Anime-Sketch-Coloring-with-Swish-Gated-Residual-UNet))

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200325151220770.png?x-oss-Process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)

## Performance
This is the performance of training 13 epochs, config is consistent with this ```config.py```. [google  drive](https://drive.google.com/file/d/1hvm3ycr3uhaEEeLSQqsBxemsQWoo5XL7/view?usp=sharing)  
Training this model takes a lot of time, so I only trained 13 epochs, which does not represent the best performance.
![在这里插入图片描述](https://raw.githubusercontent.com/gakkiri/SGRUnet-pytorch/master/test/result.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://raw.githubusercontent.com/gakkiri/SGRUnet-pytorch/master/test/result2.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://raw.githubusercontent.com/gakkiri/SGRUnet-pytorch/master/log/log.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
## Feature
 - **LayerNorm** requires a lot of memory, so **BatchNorm** was implemented, **which greatly speeds up the training, but may have an impact on performance**. You can choose which to use in ```config.py```.
 - For save your memory, you can choose **bilinear** or **transpose convolution**(paper) to upsample.
 - Two datasets are supported. [Anime Sketch Colorization Pair](https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair) and another one that was used in the paper. Also optional in ```config.py```.
 - The network that calculates the loss is different(**ResNet family** vs **VGG family**). Also optional in ```config.py```.
 - Support mini-batch training.
 
## Data folder
colorgram  
```
anime_colorization
└── data/
    ├── train/
    |   
    └── val/
```

safebooru (paper)
```
anime_colorization
└── data/
    ├── train/
    |    ├── img/
    |    └── label/
    └── val/
         ├── img/
         └── label/
```

## Setup
*pytorch >= 1.1.0*

Use the **requirements.txt** file to install the necessary depedencies for this project.
```
$ pip install -r requirements.txt
```

## Config
Modify ```config.py```as needed.

## Train
```
python  main.py
```

## Inference
Modify the 
- **model_path**   
- **file_name**
- **file_path**
- **output_path**  

in the ```inference.py```as needed.

and
```
python inference.py
```
