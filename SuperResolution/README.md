# Super Resolution

## Overview
![](./data/concept.png)

- Enhance image resolution of low-resolution image

## Usage
```bash
$ python train.py --data_path <DATA_PATH>
```

- `DATA_PATH`: Path name that contains low-resolution images

## Results

| Examples (bilinear, bicubic, mine) |
| ---- |
| ![](./data/result0.png) |
| ![](./data/result1.png) |
| ![](./data/result2.png) |
| ![](./data/result3.png) |
| ![](./data/result4.png) |
| ![](./data/result5.png) |
| ![](./data/result6.png) |
| ![](./data/result7.png) |
| ![](./data/result8.png) |
| ![](./data/result9.png) |

## Application

- If you would like to train GANs for high-resolution images but you have only low-resolution images, you can apply super resolution model to low-resolution images and obtain high-resolution images.
- Result: dataset is obtained by this super resolution model

![](./data/ada.gif)

## References
- [MAMNet: Multi-path Adaptive Modulation Network for Image Super-Resolution](https://arxiv.org/pdf/1811.12043.pdf)