
# Natural Color Fool: Towards Boosting Black-box Unrestricted Attacks

This repository is the official implementation of [Natural Color Fool: Towards Boosting Black-box Unrestricted Attacks](). 


The simplified pipeline of NCF (optimizing one image variant without initialization reset):
![The simplified pipeline of NCF](./attack_pipe.png)

## Requirements

* PyTorch 1.10+
* Python 3.8+


* Datasets: 
  Download ImageNet-compatible Dataset from [Google Drive](https://drive.google.com/drive/folders/1EZSFzDqUnccXrNkSCPM3PPC2PU87AFEX?usp=sharing) and put it in `./dataset/`.

* Color Distribution Library:
  Download from [Google Drive](https://drive.google.com/file/d/14XTkPc-2MfDWEfpCmEix5lcsKE1RCFV6/view?usp=sharing) and put it in `./segm/pretrained/`.


## Segmentation

To reproduce this paper, you need to obtain masks of all images using the semantic segmentation model **Swin-T**.

1. Downloading [pre-trained weights](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth) for semantic segmentation models **Swin-T** and put it in `segm/pretrained/`.

2. Configuring the semantic segmentation environment. Please refer to [REMEADNE](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation#usage).
   
3. To perform semantic segmentation of images, run:
```bash
cd segm
python segm/get_segMasks.py
```

## Natural Color Fool (NCF)

1. Store the color distribution space of each image in advance:

```bash
python dataset/get_lib.py
```

2. To generate adversarial examples, run:

```bash
python main.py 
```
The results are stored in `./adv/`.
>The parameters of NCF are shown in : [config_NCF.yaml](). Test different models by modifying parameters **white_models_name** and **black_models_name** in `config_NCF.yaml`.

## Results

### Transferability comparison on normally trained CNNs and ViTs.

| Model        | Attacks    | res18 | vgg19 | mobileNet_v2 | Inc_v3 | densenet121 | res50 | Vit-small | xcit-nano | Deit-small |
|--------------|------------|-------|-------|--------------|--------|-------------|-------|-----------|-----------|------------|
| Res-18       | Clean      | 16.1  | 11.4  | 12.8         | 19.2   | 7.9         | 7.5   | 13.3      | 13.7      | 5.8        |
|              | SAE        | 93.4  | 46.9  | 45.5         | 31.3   | 36.5        | 37    | 44.5      | 37.4      | 22.2       |
|              | ReColorAdv | 98.6  | 39.9  | 47.3         | 38.2   | 37.2        | 38.1  | 21.4      | 36.7      | 17.3       |
|              | cAdv       | 100   | 40.8  | 48.2         | 41.6   | 43          | 41.2  | 34.4      | 44.9      | 30.4       |
|              | ColorFool  | 93    | 27.8  | 30.5         | 28.1   | 19.8        | 22.9  | 35.5      | 22.3      | 9.2        |
|              | ACE        | 99.4  | 26    | 27.2         | 27.6   | 19.9        | 18.3  | 21.6      | 22.4      | 9.1        |
|              | NCF(Ours)  | 92.9  | 72.1  | 72.7         | 48.3   | 55.3        | 66.7  | 53        | 55.3      | 32.8       |
| vgg19        | Clean      | 16.1  | 11.4  | 12.8         | 19.2   | 7.9         | 7.5   | 13.3      | 13.7      | 5.8        |
|              | SAE        | 52.2  | 91.4  | 48.8         | 32.3   | 39.3        | 39    | 48.3      | 37.6      | 24.3       |
|              | ReColorAdv | 42.5  | 96    | 41.9         | 33.2   | 33.8        | 31.7  | 20.4      | 33.4      | 16.6       |
|              | cAdv       | 54    | 100   | 48           | 43.7   | 43.4        | 40.7  | 38.8      | 43.9      | 32.9       |
|              | ColorFool  | 44    | 90.9  | 36.5         | 29.2   | 23.5        | 26.6  | 42.2      | 25.6      | 9.6        |
|              | ACE        | 33.4  | 99.7  | 27.8         | 28.3   | 21.6        | 18    | 20.7      | 21.6      | 9.5        |
|              | NCF(Ours)  | 73.7  | 93.3  | 70.3         | 49.4   | 53.6        | 64.3  | 56.5      | 53.5      | 30.7       |
| Mobilenet_v2 | Clean      | 16.1  | 11.4  | 12.8         | 19.2   | 7.9         | 7.5   | 13.3      | 13.7      | 5.8        |
|              | SAE        | 53.5  | 49.6  | 92.2         | 34.5   | 38.1        | 39.3  | 46.6      | 37.7      | 23.3       |
|              | ReColorAdv | 46.3  | 36.5  | 97.8         | 36.4   | 32.4        | 34.4  | 20.7      | 36.7      | 20         |
|              | cAdv       | 54.7  | 39.9  | 100          | 42.8   | 44.3        | 39.1  | 36        | 44.1      | 30.8       |
|              | ColorFool  | 41.5  | 30.6  | 93.2         | 28.1   | 23.3        | 24.5  | 39.7      | 22.8      | 9.4        |
|              | ACE        | 31.8  | 25.8  | 99.1         | 26.7   | 20          | 19    | 20.3      | 22.6      | 9.3        |
|              | NCF(Ours)  | 72.8  | 72.2  | 92.7         | 50     | 54.4        | 66.2  | 55.4      | 56.4      | 32.6       |
| Inception_v3 | Clean      | 16.1  | 11.4  | 12.8         | 19.2   | 7.9         | 7.5   | 13.3      | 13.7      | 5.8        |
|              | SAE        | 49.5  | 45.8  | 45.4         | 78.2   | 36.1        | 36.5  | 46.6      | 34.7      | 23.4       |
|              | ReColorAdv | 26.5  | 19.9  | 21.9         | 96.2   | 17.2        | 15.9  | 16.3      | 22.5      | 10.5       |
|              | cAdv       | 32.7  | 23.4  | 27.6         | 99.8   | 23.9        | 20.8  | 26        | 28.2      | 18.4       |
|              | ColorFool  | 40.4  | 31.8  | 35.4         | 84.1   | 23.9        | 25.3  | 42.6      | 26.5      | 12.6       |
|              | ACE        | 28.6  | 24.1  | 23.9         | 96.9   | 18.6        | 15.5  | 19.4      | 21.8      | 9.2        |
|              | NCF(Ours)  | 57.7  | 57.7  | 56.8         | 83.8   | 40.1        | 47.7  | 45.3      | 45.2      | 23.8       |


