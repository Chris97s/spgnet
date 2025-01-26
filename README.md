# SPGNet
This is the official code for the paper "SPGNet: A Shape-Prior Guided Network for Medical Image Segmentation" (https://www.ijcai.org/proceedings/2024/0140.pdf).

Our paper has been accepted to IJCAI 2024.

## 1. Prepare Dataset
You should download the datasets mentioned in the paper from the following links:

(1) BUSI: [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data)

(2) miniJSRT(Segmentation02): [Japanese Society of Radiological Technology Database](http://imgcom.jsrt.or.jp/minijsrtdb/)

If necessary, run the code in ./dataset/split_masks.py to pre-separate the connected components.

Next, run the code in ./dataset/LMK_dataset.py to generate the landmarks data. You can also adjust the number of shape points in this file.
```text
cd dataset

python LMK_dataset.py
```

## 2. Environment
```text
pip install -r requirements.txt
```

## 3. Training
You can set the hyperparameters in train.py according to your preferences.

```text
python train.py
```

## 4. Citation
```text
@inproceedings{song2024spgnet,
  title={SPGNet: a shape-prior guided network for medical image segmentation},
  author={Song, Zhengxuan and Liu, Xun and Zhang, Wenhao and Gong, Yongyi and Hao, Tianyong and Zeng, Kun},
  booktitle={Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence},
  pages={1263--1271},
  year={2024}
}
```
