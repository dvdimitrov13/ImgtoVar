<!-- PROJECT SHIELDS -->
<div align="center">
	<a href =https://github.com/dvdimitrov13/Predict-recipe-cuisine-using-NLPLogo_Detection/blob/master/LICENSE><img src =https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat>
	</a>
	<a href =https://www.linkedin.com/in/dimitarvalentindimitrov/><img src =https://img.shields.io/badge/-LinkedIn-black.svg?style=flat&logo=linkedin&colorB=555>
	</a>
	<a href =https://www.linkedin.com/in/dimitarvalentindimitrov/><img src =https://static.pepy.tech/personalized-badge/imgtovar?period=total&units=international_system&left_color=grey&right_color=blue&left_text=pypi%20downloads>
	</a>
	<!-- More to be added - stars, doi -->
</div>

<h1 align="center">imgtovar</h1>

  <p align="left">
   Imgtovar is a Python module, developed in collaboration with researchers, which allows for variable extraction from image data. The pipeline consists of three steps - image extraction, cleaning and prediction. This module was bootstrapped using the popular face analysis framework DeepFace.

Currently supporting Natural vs Man-made background analysis, Chart recognition and identification, Age, Gender, Race and Emotion prediction, and finally Object Detection of a total of 100+ objects.
  </p>
</div>


<!-- TABLE OF CONTENTS -->
## Table of Contents
<details>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#variable-extraction-pipeline">Variable extraction pipeline</a>
      <ul>
        <li><a href="#image-extraction">Image extraction</a></li>
   </ul>
      <ul>
        <li><a href="#cleaning">Cleaning</a></li>
        <li><a href="#feature-extraction">Feature extraction</a></li>
        <ul>
         <li><a href="#facial-attributes-analysis">Facial attributes analysis</a></li>
        <li><a href="#background-analysis">Background analysis</a></li>
        <li><a href="#object-detection">Object-detection</a></li>
      </ul>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


## Installation [![PyPI](https://img.shields.io/pypi/v/imgtovar.svg)](https://pypi.org/project/imgtovar/) 

The easiest way to install imgtovar is to download it from [`PyPI`](https://pypi.org/project/imgtovar/). It's going to install the library itself and its prerequisites as well.

```shell
$ pip install imgtovar
```

## Variable Extraction Pipeline

Here is an example of a full feature extraction pipeline:

```
commands - add later
```

Now we go over a quick explanation of all the methods.

:heavy_exclamation_mark: **All code has detailed comments explaining the different parameters and their functions**

### Image extraction
Imgtovar allows the user to extract images from documents in case there is no image database. 

```python
ImgtoVar.extract(data, mode="PDF")
```

This function extracts all images from a document and stores them in a directory named: ``./extract_output``

The `data` parameter can be either a single file or a directory.

:warning: **Currently supported modes**: PDF only!

<p align="right">(<a href="#top">back to top</a>)</p>

### Cleaning
During initial development, cleaning was found to be a crucial step, especially when the images were acquired through automatic image extraction from documents. 

There are three methods used in the cleaning process, I have found that its best to filter out corrupted images and infographics before running the color analysis in order to limit false positives in the last step, here are the methods in the intended order:

**1. detect_infographics**

This method detects if an image is an infographic and if it predicts it's type. The detection stage has an  f1-score of 97% showing strong performance. On the other hand the overall classifier accuracy is 87% which is an indication of the performance of the identification stage.

```python
charts_df = ImgtoVar.detect_infographics(data)
```
<!-- Change the extract parameters to true by default -->

The functions returns a DataFrame with the image file names and the predicted chart_type. Additionally, provided that the user agrees, Infographics will be moved to a new directory.

This method should be used before the `color_analysis` in order to reduce the false positives in that step.

**2. detect_invertedImg**

Through trial and error we have found that image extraction from PDF files results in a small percentage of images being corrupted. Those images have inverted channel values and additional problems with contrast and lightness. To identify those images ImgtoVar provides a method that has 93% accuracy in detecting the corrupted images.

```python
hl_pairs_df = ImgtoVar.detect_invertedImg(data)
```

The method returns a DataFrame with the predicted status of the analysed images. Additionally, provided user approval, the images identified as inverted are moved to a new directory.

**3. color_analysis**

To filter out the undesirable images, ImgtoVar provides a method for analysing the distribution on hue/lightness pairs across all pixels.

```python
hl_pairs_df = ImgtoVar.color_analysis(data)
```

A pandas DataFrame is returned containing the image file name, the total H/L pairs found the and proportion represented by the top 200 pairs (this is adjustable). Additionally, provided that the user agrees, images identified as "Artificial" will be moved to a new directory.

The filtering is based on that proportion. Where real images will have low proportion and drawings, logos or single color images will have a high proportion. By varying the `threshold` parameter the user can make the filtering more or less aggressive. 

 |     Color analysis  |	200 H/L Width	|
:-------------------------:|:-------------------------:|  
| ![](https://github.com/dvdimitrov13/ImgtoVar/blob/master/images/nature_9.jpg)| ![](https://github.com/dvdimitrov13/ImgtoVar/blob/master/images/other_10.jpg) |
|Dominant pairs proportion: 3% | Dominant pairs proportion: 64%

This method is very effective at identifying real photographs, but can mistakenly label simpler images like infographics as undesirable.

<p align="right">(<a href="#top">back to top</a>)</p>

### Feature extraction
Once we have clean data, we can begin structuring our image data by using the methods included in ImgtoVar.

This section outlines the three main methods behind feature extraction. Several pre-trained models are included with the library, but the methods also allow for integration with custom models.

#### Facial attributes analysis
The facial attribute analysis was made based on the popular Python module [DeepFace](https://github.com/serengil/deepface).  ImgtoVar adds two important features.

First, the face detection function was reworked in order to return all detected faces in an image, and therefore run the analysis on each. 

Second, the apparent age classifier was changed for a new custom model, as the original model included with DeepFace lacked training examples below 18 years old, and had limited examples in the higher age groups as well. This leads to poor performance on the test data. The new model classifies age in one of the following groups: child, young adult, adult, middle age, old with 72% accuracy. 

![](https://github.com/dvdimitrov13/ImgtoVar/blob/master/images/Age_CM.png)

Here is a potential workflow:

```python
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

#facial analysis
demography_df = ImgtoVar.face_analysis(data, actions=("emotion", "age", "gender", "race"), detector_backend = backends[4])
```
The method returns a DataFrame with the image file names and the predicted label for each action specified. By default, retinaface is used as a backend and all actions are predicted. To see a comparison of the different backends you can refer to [this demo](https://youtu.be/GZ2p2hj2H5k) created by the author of DeepFace.

As he writes in the DeepFace documentation: "[RetinaFace](https://sefiks.com/2021/04/27/deep-face-detection-with-retinaface-in-python/) and [MTCNN](https://sefiks.com/2020/09/09/deep-face-detection-with-mtcnn-in-python/) seem to overperform in detection and alignment stages but they are much slower. If the speed of your pipeline is more important, then you should use opencv or ssd. On the other hand, if you consider the accuracy, then you should use retinaface or mtcnn."

#### Background analysis
The background analysis detects the context of an image, e.g. if an image depicts an urban skyline or a forrest. The classifier has 93% accuracy in identifying natural vs man-made image backgrounds.

```python
background_df = ImgtoVar.background_analysis(data)
```

The method returns a DataFrame with the image file names and the predicted background. To train this classifier, a custom dataset was created.

Due to limitations of the training data some nature examples have limited man-made structures, therefore this classifier cannot be used on its own to filter out images in which no man-made objects exist.

For example, if an image shows a natural landscape with a small house in the middle, that will be classified as natural.

 |     Label: |	Natural	| Natural | Man-made|
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|  
| Image:| <img src="https://github.com/dvdimitrov13/ImgtoVar/blob/master/images/nature_8.jpg" height="150" width="200"/>| <img src="https://github.com/dvdimitrov13/ImgtoVar/blob/master/images/nature_4.jpg" height="150" width="200"/>| <img src="https://github.com/dvdimitrov13/ImgtoVar/blob/master/images/not_nature_5.jpg" height="150" width="200"/>| 

If the researcher wants to detect images with nothing man-made in them, the `background_analysis` method can be used in combination with the `object_detection` method to identify false positive cases.

#### Object Detection

For object detection, ImgtoVar uses the [YoloV5](https://github.com/ultralytics/yolov5) family of models. 

There are 2 custom pre-trained models included with the module, as well as all the pre-trained models on the COCO dataset included with YoloV5 itself. 

The COCO dataset covers 80 classes, to which ImgtoVar adds 24 classes extracted from the OpenImages dataset and additional 6 classes trained on a custom dataset. Finally, the module allows users to specify their own custom pre-trained weights and model architecture. 

To use the COCO dataset:

```python
coco_od_df = ImgtoVar.detect_objects(data, model="custom", weights="yolov5l.pt",labels=None)
```

The method returns a DataFrame with the the image file name, object detected, the position and the confidence of prediction. 

Since for most researchers the existence of an object is more important than its exact coordinates, we report mAP at 0.5 IoU, which for the `yolov5l.pt` model is 67.3%. 

<details>
  <summary>Here is a list of the labels included in the COCO dataset:
</summary>

  ```
  [

"person",

"bicycle",

"car",

"motorcycle",

"airplane",

"bus",

"train",

"truck",

"boat",

"traffic light",

"fire hydrant",

"stop sign",

"parking meter",

"bench",

"bird",

"cat",

"dog",

"horse",

"sheep",

"cow",

"elephant",

"bear",

"zebra",

"giraffe",

"backpack",

"umbrella",

"handbag",

"tie",

"suitcase",

"frisbee",

"skis",

"snowboard",

"sports ball",

"kite",

"baseball bat",

"baseball glove",

"skateboard",

"surfboard",

"tennis racket",

"bottle",

"wine glass",

"cup",

"fork",

"knife",

"spoon",

"bowl",

"banana",

"apple",

"sandwich",

"orange",

"broccoli",

"carrot",

"hot dog",

"pizza",

"donut",

"cake",

"chair",

"couch",

"potted plant",

"bed",

"dining table",

"toilet",

"tv",

"laptop",

"mouse",

"remote",

"keyboard",

"cell phone",

"microwave",

"oven",

"toaster",

"sink",

"refrigerator",

"book",

"clock",

"vase",

"scissors",

"teddy bear",

"hair drier",

"toothbrush",

]
  ```

</details>

To use the subset of OpenImages dataset:

```python
coco_od_df = ImgtoVar.detect_objects(data, model="sub_open_images")
```

The mAP at 0.5 IoU is 59%. This value is comparable with scores from the OpenImages data challenge where a much more complicated model achieved an overall score of 65% mAP at 0.5. The advantage of YoloV5 is its cutting edge speed, which allows researchers to extract variables from larger datasets.

<details>
  <summary>Here is a list of the labels included in the custom OpenImages dataset:
</summary>

  ```
  [

"Animal",

"Tree",

"Plant",

"Flower",

"Fruit",

"Suitcase",

"Motorcycle",

"Helicopter",

"Sports_equipment",

"Office_building",

"Tool",

"Medical_equipment",

"Mug",

"Sunglasses",

"Headphones",

"Swimwear",

"Suit",

"Dress",

"Shirt",

"Desk",

"Whiteboard",

"Jeans",

"Helmet",

"Building",

]
  ```

</details>

The final dataset is a custom dataset, made in connection with the research that this module is being applied to. It includes 6 labels connected to sustainability, such as wind mills, solar panels, oil pumps etc.

```python
coco_od_df = ImgtoVar.detect_objects(data, model="c_energy")
```

The mAP at 0.5 IoU is 91%. 

<details>
  <summary>Finally, here is a list of the 6 labels included in this dataset:
</summary>

  ```
  [

"Crane",

"Wind turbine",

"farm equipment",

"oil pumps",

"plant chimney",

"solar panels",

]
  ```

</details>

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the GNU License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Dimitar Dimitrov 
	<br>
Email - dvdimitrov13@gmail.com
	<br>
LinkedIn - https://www.linkedin.com/in/dimitarvalentindimitrov/

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Aknowledgements -->
## Acknowledgements

Special thanks goes to my thesis advisor, Francesco Grossetti, who has helped me develop and verify my work.

<p align="right">(<a href="#top">back to top</a>)</p>
