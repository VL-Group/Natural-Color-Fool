import torch
import torchvision

import os
import yaml
import wandb

import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

from torch_attack.attacks.NCF import NCF

from utils.models import get_models
from utils.general import save_img, seed_torch

def attack_pipeline(hyperparameters, opt):
    """pipeline

    Args:
        hyperparameters (dict): There are only super parameters and will be uploaded to wandb.
        opt ([type]): other parameters
    """
    if hyperparameters['local']: 
        run_mode="disabled" 
    else: 
        run_mode=None  
    with wandb.init(project="NCF", config=hyperparameters, name=opt.run_name, mode=run_mode, anonymous="allow"):
        config = wandb.config

        # Initialization
        white_models = get_models(config.white_models_name, config.model_mode, config.device)
        black_models = get_models(config.black_models_name, config.model_mode, config.device)

        attacker = NCF(white_models, config)

        # Start attack
        attack(attacker, white_models, black_models, config, output_dir=config.output_dir)

def attack(attacker, white_models:dict, black_models:dict, config, output_dir):

    # ground truth
    with open(config.label_path) as f:
        ground_truth=f.read().split('\n')[:-1]

    device = torch.device(config.device)

    test_models = white_models.copy()
    test_models.update(black_models)
    models_names = list(test_models.keys())

    input_num = config.data_range[1] - config.data_range[0]
    logits, correct_num, correct_class_num = {}, {}, {}
    for model_name in models_names:
        correct_num[model_name] = 0
        correct_class_num[model_name] = []

    # load images masks
    masks = np.load(config.masks_path)  # (1000,299,299)

    batch_idx = 0
    for idx in tqdm(range(config.data_range[0], config.data_range[1], config.batch_size)):
        # load data
        if (config.data_range[1]-idx) < config.batch_size:
            end = config.data_range[1]
        else:
            end = idx + config.batch_size
        images, labels, filenames, color_20 = [], [], [], []

        re_size = config.images_size

        for i in range(idx, end):
            img_path = os.path.join(config.img_dir, '{}.png'.format(i))
            pil_image = Image.open(img_path).convert('RGB').resize((re_size, re_size))
            image = torch.tensor(np.array(pil_image), device=device)  # (H, W, 3)
            label = int(ground_truth[i-1])
            filename = str(i) + '.png'
            # load color distribution library
            color_path = os.path.join(config.color_dir, filename)
            pil_color = Image.open(color_path).convert('RGB').resize((re_size*8+18, re_size*3+8))  # resize(width,height)
            color = torch.tensor(np.array(pil_color), device=device) / 255.  # (H*3+8,W*8+18,3)
   
            color_20.append(color)
            images.append(image)
            labels.append(label)
            filenames.append(filename)
        images = torch.stack(images, dim=0)
        images = (images/255.).permute(0, 3, 1, 2)
        labels = torch.tensor(labels, device=device)
        mask = torch.tensor(masks[idx-1:end-1], dtype=torch.int, device=device)
        mask = torchvision.transforms.functional.resize(mask, [re_size, re_size], torchvision.transforms.InterpolationMode.NEAREST)
        color_20 = torch.stack(color_20, dim=0)  # (n,H*3+8,W*8+18,3)
        

        if config.model_mode == 'torch':
            labels = labels - 1

        # start attack
        adv_images = attacker(images, labels, filenames, mask, color_20, batch_idx)

        # save adversarial images
        save_img(adv_images, filenames, output_dir)

        # Test attack success rate
        current_batch_size = images.shape[0]
        with torch.no_grad():
            for model_name in models_names:
                # Calculate logits
                if config.model_mode == 'torch':
                    logits[model_name] = test_models[model_name](adv_images.clone())
                elif config.model_mode == 'tf':
                    logits[model_name] = test_models[model_name](adv_images.clone())[0]

                # Calculate the number of successful attacks.
                if current_batch_size == 1:
                    correct_num[model_name] += (torch.argmax(logits[model_name]) != labels).detach().sum().cpu()
                else:
                    max_index = torch.argmax(logits[model_name], axis=1) != labels
                    correct_num[model_name] += max_index.detach().sum().cpu()
                    # Test the success rate of different types of attacks
                    correct_class_num[model_name] = correct_class_num[model_name] + list(max_index.cpu().numpy()*1)

        batch_idx += 1


    # Print attack result.
    for i, net in enumerate(models_names):
        wandb.log({net: correct_num[net]/input_num})
        print('{} attack success rate: {:.2%}'.format(net, correct_num[net]/input_num))


if __name__ == "__main__":
    seed_torch(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='')
    parser.add_argument('--run_name', type=str, default='', help='')

    parser.add_argument('--batch_size', type=int, help='batch_size.')
    parser.add_argument('--num_reset', type=int, help='The number of initialization reset.')
    parser.add_argument('--eta', type=int, help='The number of random searches.')
    parser.add_argument('--num_iter', type=int, help='The iteration of neighborhood search.')
    parser.add_argument('--T_step', type=float, help='The iterative step size of T.')
    parser.add_argument('--momentum', type=float, help='momentum.')

    opt = parser.parse_args()

    # set gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu    

    # load config
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(cur_dir, "config_NCF.yaml")
    f = open(yaml_path, 'r', encoding='utf-8')
    attack_config = yaml.load(f.read(), Loader=yaml.FullLoader) 
    f.close()

    # hyperparameters
    config = attack_config.copy()
    config['batch_size'] = config['batch_size'] if opt.batch_size is None else opt.batch_size
    config['num_reset'] = config['num_reset'] if opt.batch_size is None else opt.num_reset
    config['eta'] = config['eta'] if opt.eta is None else opt.eta
    config['num_iter'] = config['num_iter'] if opt.num_iter is None else opt.num_iter
    config['T_step'] = config['T_step'] if opt.T_step is None else opt.T_step

    # start
    print("config:", config)
    attack_pipeline(config, opt)