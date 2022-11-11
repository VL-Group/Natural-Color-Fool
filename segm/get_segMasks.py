import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

from mmseg.apis import inference_segmentor, init_segmentor

def segm(segm_model, images):
    result = inference_segmentor(segm_model, images)
    
    result = result[0] + 1  

    return result

if __name__ == "__main__":
    # config = 'segm/configs/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py'
    # checkpoint = 'segm/pretrained/upernet_swin_tiny_patch4_window7_512x512.pth'
    # img_dir = 'dataset/images'


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='segm/configs/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py', help='config')
    parser.add_argument('--checkpoint', type=str, default='segm/pretrained/upernet_swin_tiny_patch4_window7_512x512.pth', help='pre-trained model')
    parser.add_argument('--img_dir', type=str, default='dataset/images', help='input images directory')

    opt = parser.parse_args()

    segm_model = init_segmentor(opt.config, opt.checkpoint, device='cuda:0')
    masks = []
    for idx in tqdm(range(1, 1001)):
        img_path = os.path.join(opt.img_dir, '{}.png'.format(idx))
        pil_image = Image.open(img_path).convert('RGB')

        images = np.array(pil_image)  # (H, W, 3)
        filename = str(idx) + '.png'        

        mask = segm(segm_model, images)
        masks.append(mask)

    np.save("./segm/masks.npy", masks)
    print("masks saved in './segm/masks.npy'")