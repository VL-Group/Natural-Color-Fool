import os
import torch
import numpy as np
import torchvision
import argparse
from PIL import Image

from multiprocessing import Pool
from tqdm import tqdm

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def pick_candidate(mask):
    """Extract the "others" object class, and get all object classes in the input image. We removed objects with area ratios less than 0.0005.

    Args:
        mask (torch.Tensor): mask, (244,224)

    Returns:
        objects_list (lsit): all object class in the mask
        scores_ (int): area
    """
    objects = torch.unique(mask)

    scores = torch.zeros(len(objects), 1)
    mask_area = mask.shape[-1]*mask.shape[-2]
    counter = 0
    for obj in objects:   
        temp_BW = (mask==obj)
        area_ratio = temp_BW.sum()/mask_area
        scores[counter] = area_ratio  
        counter = counter + 1
    scores_, ind = scores.sort(0)  # Ascending order
    idx = (scores_>0.0005)  # 
    ind = ind[idx]
    
    ind = ind.flip(0)
    objects_list = objects[ind]
    scores_ = scores[ind]

    return objects_list.long(), scores_

def async_fn(obj_area, target, tmp_pixels, bin):

    # Convert to pixel values with no spatial information.
    inds_target = np.nonzero(target)

    start_t = 0
    len_color = len(inds_target[0])
    for i in range(len_color):
        (R, G, B) = inds_target[0][i], inds_target[1][i], inds_target[2][i] 

        tmp_pixels[start_t: (start_t + np.floor((obj_area*target[R, G, B]))).astype(int), 0] = R / bin
        tmp_pixels[start_t: (start_t + np.floor((obj_area*target[R, G, B]))).astype(int), 1] = G / bin
        tmp_pixels[start_t: (start_t + np.floor((obj_area*target[R, G, B]))).astype(int), 2] = B / bin
        start_t = start_t + np.floor((obj_area*target[R, G, B])).astype(int)

    return tmp_pixels

def hist2img_multi(image, mask, all_objects, target_hist, area_objs, bin=32):

    num_objs = len(all_objects)
    target_imgs = np.zeros_like(image)[None,:,:,:].repeat(20,axis=0)
    # target_imgs = image[None,:,:,:].repeat(20,axis=0)

    for obj_idx in range(num_objs):

        obj = all_objects[obj_idx]
        BW = (mask==obj)
        obj_area = BW.sum()
        pool = Pool()
        results = []
        for img_idx in range(20):
            assert target_hist[obj_idx][img_idx].sum() - 1 <0.0001

            # target_imgs[img_idx][BW] = async_fn(obj_area, target_hist[obj_idx][img_idx], target_imgs[img_idx][BW], bin)
            results.append(pool.apply_async(async_fn, args=(obj_area, target_hist[obj_idx][img_idx], target_imgs[img_idx][BW],bin,)))

        pool.close()
        pool.join()
        for img_idx in range(20):
            target_imgs[img_idx][BW] = results[img_idx].get()

    return target_imgs  # (20, H, W, 3)

if __name__ == "__main__":

    re_size = 299

    parser = argparse.ArgumentParser()
    parser.add_argument('--lib_path', type=str, default='dataset/150_20_hist.npy', help='Path of color distribution library.')
    parser.add_argument('--img_dir', type=str, default='dataset/images', help='input images directory')
    parser.add_argument('--masks_path', type=str, default='segm/masks.npy', help='mask path')
    parser.add_argument('--out_lib_dir', type=str, default='dataset/lib_299', help='mask path')
    opt = parser.parse_args()

    device = torch.device('cpu')  # cuda:0

    mkdir(opt.out_lib_dir)
    all_hist = torch.tensor(np.load(opt.lib_path))  #(150,20,32,32,32)
    masks = np.load(opt.masks_path)  # (1000,299,299)

    imgs_list = os.listdir(opt.img_dir)
    (imgs_list).sort()

    for name in tqdm(imgs_list):
        img_idx = int(name.split(".")[0])

        # load mask
        mask = torch.tensor(masks[img_idx-1:img_idx], dtype=torch.int, device=device)
        mask = torchvision.transforms.functional.resize(mask, [re_size, re_size], torchvision.transforms.InterpolationMode.NEAREST)
        all_objects, area_objs = pick_candidate(mask)

        # load image
        img_path = os.path.join(opt.img_dir, name)
        pil_image = Image.open(img_path).convert('RGB').resize((re_size, re_size))
        image = torch.tensor(np.array(pil_image), device=device) / 255. # (H, W, 3)

        # lib
        target_hist = all_hist[all_objects-1]

        hist_imgs = hist2img_multi(image.numpy(), mask[0].numpy(), all_objects.numpy(), target_hist.numpy(), area_objs.numpy(), bin=32)  # (20, H, W, 3)

        # Save
        hist_imgs = torch.tensor(hist_imgs).permute(0,3,1,2)
        torchvision.utils.save_image(hist_imgs, os.path.join(opt.out_lib_dir, name))