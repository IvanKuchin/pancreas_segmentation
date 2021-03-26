# Pancreas Segmentation

Tensorflow implementation of a 3D-CNN U-net with Grid Attention and DSV for pancreas segmentation from CT.

## Network architecture

- [Classic U-net](https://arxiv.org/pdf/1505.04597.pdf) with residual connections
- [Grid Attention](https://arxiv.org/pdf/1804.03999.pdf) gave a biggest boost in the performance
- [DSV](http://proceedings.mlr.press/v38/lee15a.pdf) forces intermediate feature-maps to be semantically discriminative

![image](https://user-images.githubusercontent.com/26530162/112643787-202ccc00-8e1b-11eb-9d4f-16fcd6376a3e.png)

## Training

Network has been trained on publicly accessible dataset CT-82 from [TCIA](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT#0c26eab54502412cbbd0e1c0fddd917b) (64/16 split between train/validation)

Weighted DSC (Dice Similarity Coefficient) used as a loss function. Best weight hyperparameter served to my purposes selected as 7.
- recall ~95%
- precision ~57%
- DCS/F1 ~72% (though it is not really important for my experiments)

![image](https://user-images.githubusercontent.com/26530162/112645949-65ea9400-8e1d-11eb-9668-b0917acbea87.png)

[Here](https://tensorboard.dev/experiment/jdJUxCWrQiWk4ezy0i9TvA/) is the Tensorboard.dev comparison between weight hyperparameter with values: 1, 7, 10. I recommend to enable only `validation` runs and apply filter tag as follow `f1|recall|prec`

### Training process

Whole network has been trained end-to-end, w/o any tiling. Reasoning is to avoid artifacts where pancreas segmentation cut to a tile edge.

Every CT downscaled to dimensionality 160x160x160, this is the maximum size that fits into TeslaK40m (12GB RAM). Pooling implemented over WxD dimensions, D (depth) keeps constant (ie 160 over whole network), this helps a little with segmentation recovery.

Optimization algoritm Adam with start learning rate 0.002 then reduce on plateau by 0.1 over 30 epochs. Total number of epochs restricted to 1000.

Training took ~60 hours on a single server with a single GPU NVIDIA TeslaK40m. Most of the progress achieved in first 3-5 hours. 

tensorflow container used as a runtime environment:

```
docker pull tensorflow/tensorflow:2.3.2-gpu
github clone https://github.com/IvanKuchin/pancreas_segmentation.git
python train_segmentation.py
```

