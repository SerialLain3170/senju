# KawaiiGenerator

## Overview
KawaiiGenerator enables users to try various deep learning tasks related to anime.
- ImageManupilation
- SuperResolution

## Quick Results

| Task | Concept |
| ---- | ---- |
| ImageManupilation | ![](./ImageManupilation/data/image.png)![](./ImageManupilation/data/image2.png) |
| SuperResolution | ![](./SuperResolution/data/concept.png) |

## GUI Application
![](./data/gui_im.png)

- KawaiiGenerator also offers a GUI application that enables users to try applications like the figure above.

### Getting Started
#### 0. Download Pre-trained file
You can download pre-trained files for GUI application from [this link](https://github.com/SerialLain3170/KawaiiGenerator/releases/tag/v0.1.0-alpha).  
After that, you need to move `decoder.pt` and `encoder.pt` to `gui/server/ckpts/`.

#### 1. Docker image preparation
Build docker images via

```
$ bash build.sh
```

#### 2. Start GUI
Start application via the command below and access 0.0.0.0:5000.

```
$ docker-compose up -d
```

### System Configuration
![](./data/system.png)
