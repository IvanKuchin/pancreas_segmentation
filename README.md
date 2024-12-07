# Pancreas segmentation and cancer classification

### Segmentation:   
Tensorflow implementation of a 3D-CNN U-net with Grid Attention and DSV for pancreas segmentation from CT.

### Classification:   
Encoder part of the network kept intact, generation part has been removed. At the bottom of the network binary-classification part has been added.

(**Roadmap:** add another classification-output to the segmentation network, which helps enrich learned features with additional cancer/non-cancer information)

## Network architecture

- [Classic U-net](https://arxiv.org/pdf/1505.04597.pdf) with residual connections
- [Grid Attention](https://arxiv.org/pdf/1804.03999.pdf) gave a biggest boost in the performance
- [DSV](http://proceedings.mlr.press/v38/lee15a.pdf) forces intermediate feature-maps to be semantically discriminative

![image](https://user-images.githubusercontent.com/26530162/112643787-202ccc00-8e1b-11eb-9d4f-16fcd6376a3e.png)

## Dataset and metrics

Network has been trained on publicly accessible dataset CT-82 from [TCIA](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT#0c26eab54502412cbbd0e1c0fddd917b) (64/16 split between training/validation)

Weighted DSC (Dice Similarity Coefficient) used as a loss function. Best weight hyperparameter served to my purposes selected as 7.
- recall ~95%
- precision ~57%
- DCS/F1 ~72% (though it is not really important for my experiments)

![image](https://user-images.githubusercontent.com/26530162/112645949-65ea9400-8e1d-11eb-9668-b0917acbea87.png)

[Here](https://tensorboard.dev/experiment/jdJUxCWrQiWk4ezy0i9TvA/) is the Tensorboard.dev comparison between weight hyperparameter with values: 1, 7, 10. I recommend to enable only `validation` runs and apply filter tag as follow `f1|recall|prec`

### Segmentation training process

Whole network has been trained end-to-end, w/o any tiling. Reasoning is to avoid artifacts where pancreas segmentation cut to a tile edge.

Every CT downscaled to dimensionality 160x160x160, this is the maximum size that fits into TeslaK40m (12GB RAM). Pooling implemented over WxH dimensions only, D (depth) keeps constant (ie 160 over the whole network), this helps a little with segmentation recovery. Single CT in a training batch, therefore BatchNormalization was not in use.

Optimization algoritm Adam with start learning rate 0.002 then reduce on plateau by 0.1 over 30 epochs. Total number of epochs restricted to 1000.

Training took ~60 hours on a single server with a single GPU NVIDIA TeslaK40m. Most of the progress achieved in first 3-5 hours. 

tensorflow container used as a runtime environment:

```
docker run --rm -it tensorflow/tensorflow:2.3.2-gpu /bin/bash
github clone https://github.com/IvanKuchin/pancreas_segmentation.git
cd panreas_segmentation
python train_segmentation.py
```

Segmentation learned weights are available [here](http://fun.conn-me.ru/pancreas_segmentation/weights.hdf5) due to GitHub limitation on big files.

Classification weights available on [HuggingFace](https://huggingface.co/IvanKuchin/pancreas_cancer_classification)

### Segmentation inference

*Pre-requisite*: tensorflow 2.3 (you could try latest version, but no guarantee that it will work)

Inference can be done on a regular laptop without any GPU installed. Time required for inference ~10-15 seconds.

To test segmentation on your data
1. Clone this repository `github clone https://github.com/IvanKuchin/pancreas_segmentation.git`
2. Create `predict` folder in cloned folder and put there single pass CT. If it will contain multiple passes result is unpredictable.
3. Download `weights.hdf5` from the link above and put it in the root of cloned folder
4. `python src/pancreas_ai/bin/predict_segmentation.py`

Output will be `prediction.nii` which [Neuroimaging Informatics Technology Initiative](https://nifti.nimh.nih.gov/)

All magic happening in last three lines 
```
if __name__ == "__main__":
    pred = Predict()
    pred.main("predict", "prediction.nii")
```

I used [3DSlicer](https://download.slicer.org/) to check the results visually.

## An importance of probability distribution in source data

Network has been trained on CT-82 with every scan is contrast-free. The network should recognize similar scans to CT-82 probability distribution.

I've tried to test input CT **with contrast**, result was unsatisfied. 

## Segmentation results

Video recording of segmentation results posted on [connme.ru](https://www.connme.ru) in a group *Pancreas cancer detection*

Example of prediction in 3DSlicer (prediction: green, ground truth: red)

![image](https://user-images.githubusercontent.com/26530162/113589582-8970c400-95ff-11eb-8bb7-aa85f1d312dd.png)

---

# Classification part

## Training

All information about training/metrics/results as well as trained weights are on the [model card](https://huggingface.co/IvanKuchin/pancreas_cancer_classification)

## Inference

### Option 1. Docker container (preferred)

### Option 2. Python package

1. Install python >= 3.12
2. Create virtual environment: `python -m venv .venv`
3. Install pancreas_ai: `pip install git+https://github.com/IvanKuchin/pancreas_segmentation`
4. Create folder *predict* `mkdir predict`
5. Run the inference: `predict`

