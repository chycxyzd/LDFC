# A Lung Nodule Dataset with Histopathology-based Cancer Type Annotation
This repository contains Python code that may be used when using datasets

## Description of the Document
```
- classification       # Contains model/training files on classification tasks
  - ConvNext.py
  - EfficientNet.py
  - Res2Net.py
  - ResNext.py
  - resnet50.py
  - train.py
  - train_utils.py
- detection            # Contains model/training files on detection tasks
  - build_utils        # Tools used in building the training network
    - _init_.py
    - datasets.py
    - img_utils.py
    - layers.py
    - parse_config.py
    - torch_utils.py
    - utils.py
  - cfg                # Configuration file directory
    - hyp.yaml         # Network structure configuration
    - yolov3-spp.cfg   # Network structure configuration
  - data               # Stores a cache of information about the dataset during training
    - my_data_label.names
    - pascal_voc_classes.json
  - train_utils        # Tools used in training the validation network
    - _init_.py
    - coco_eval.py
    - coco_utils.py
    - distributed_utils.py
    - group_by_aspect_ratio.py
    - train_eval_utils.py
  - calculate_dataset.py
  - draw_box_utils.py
  - models.py
  - train.py
- augment.py           # For data enhancement
- augment_2D.py        # For 2D data enhancement
- draw_bbox.py         # drawing bbox
- read_mhd.py          # Read mhd data
```
## Environment Configuration
* Python 3.8
* Pytorch >= 1.10.0
* pycocotools(Linux: `pip install pycocotools`;   
  Windows: `pip install pycocotools-windows`
* For more information on environment configuration, see the `requirements.txt` file
