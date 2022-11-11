import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_attack.attacks.attack_utils.color_space import rgb2lab, lab2rgb, lab_type_convert
from torch_attack.attacks.attack_utils.loss_fn import cw_loss6

from torch_attack.base_attack import BaseAttack

from torch_attack.attacks.attack_utils.object_name import getObjectName
from torch_attack.attacks.attack_utils.transfer_colors import MKL, colour_transfer, get_imgs_T, torch_cov




class NCF(BaseAttack):
    def __init__(self, models: dict, config):
        super().__init__('NCF', models, config)

        self._supported_mode = ['untargeted']

        self.color_mode = config.color_mode
        self.batch_size = config.batch_size

        self.num_reset = config.num_reset  # the number of initialization reset.
        self.eta = config.eta  # the number of random searches
        self.num_iter_T = config.num_iter  # the iteration of neighborhood search

        self.T_step = config.T_step  # the step size
        self.momentum = config.momentum
    
    def forward(self, images, labels, filenames, masks, color_20,img_idx):

        # images  # (N,3,H,W)
        
        # library
        tar_hist_imgs = color_20  # (N,H*3+8,W*8+18,3)

        n_img = images.shape[0]
        H = images.shape[2]
        W = images.shape[3]

        # Record the semantic classes and number of classes for all images.
        all_objs = []
        len_obj = []
        for i in range(n_img):
            all_objs_, _ = self.pick_candidate(masks[i])
            len_obj.append(len(all_objs_))
            all_objs.append(all_objs_)

        # Convert color space.
        if self.color_mode == 'Lab':
            images_int_lab = rgb2lab(images)
            images_float_lab = lab_type_convert(images_int_lab, mode="int2float")
            ori_images_float = images_float_lab

            tar_hist_imgs = tar_hist_imgs.permute(0,3,1,2) #(n,3,H*3+8,W*8+18)

            tar_hist_imgs_int_lab = rgb2lab(tar_hist_imgs)
            tar_hist_imgs_float_lab = lab_type_convert(tar_hist_imgs_int_lab, 'int2float')
            tar_hist_imgs_float_ = tar_hist_imgs_float_lab.permute(0,2,3,1)  # (N,H*3+8,W*8+18,3)
            ori_mean_channel = torch.mean(ori_images_float, dim=[2,3])

        # library
        tar_hist_imgs_float = torch.zeros((n_img,20,H,W,3),device=self.device)  # (N,20,H,W,3)

        for i in range(n_img):
            tar_hist_imgs_float[i] = spilt_color(tar_hist_imgs_float_[i].clone())


        best_loss_per_example = {i: None for i in range(n_img)}
        best_attack_succ = [None for i in range(n_img)]
        best_adv_loss_per_example = torch.zeros((n_img), device=self.device)
        best_T_loss_per_example = torch.zeros((n_img), device=self.device)  
        best_adv_images = ori_images_float.clone()

        # III. Initialization Reset(IR)
        for iter_no in range(self.num_reset):
            # I. Select a good initial point.
            best_mean_channel = torch.zeros((n_img,3), device=self.device)
            best_T_matrix = torch.zeros((n_img, 3, 3), device=self.device)
            best_target_images_float = torch.zeros_like(ori_images_float)

            for i in range(self.eta):
                with torch.no_grad():

                    # random index
                    rand_p = torch.randint(0,20,(n_img, 151),device=self.device)
                    hist_p = F.one_hot(rand_p, num_classes=20).float()

                    # Get target color images x_H according to index.
                    target_images_float = self.get_target_image(ori_images_float, hist_p, tar_hist_imgs_float, masks.clone(), all_objs)  # (N,3,H,W)

                    # Get transfer matrix and mean.
                    T_true = get_imgs_T(ori_images_float, target_images_float)
                    mean_channel = torch.mean(target_images_float, dim=[2,3]) 

                    # Obtaining the adversarial loss of adversarial examples based on the transfer matrix and the mean.
                    loss, adv_loss, adv_images = self.get_loss(ori_images_float.clone(), labels, T_true.clone(), mean_channel)

                    if i==0:
                        best_tar_loss_per_example = adv_loss
                        best_msk = torch.ones_like(adv_loss) == 1
                    else:
                        best_msk = (best_tar_loss_per_example-adv_loss < 0)

                    best_tar_loss_per_example[best_msk] = adv_loss[best_msk]

                    best_T_matrix[best_msk] = T_true[best_msk].clone().detach()
                    best_mean_channel[best_msk] = mean_channel[best_msk].clone().detach()
                    best_target_images_float[best_msk] = target_images_float[best_msk].clone().detach()

            mean_channel = best_mean_channel
            mean_channel[:,0] = ori_mean_channel[:,0] 
            T_matrix = best_T_matrix.clone().detach()

            # II. Neighborhood Search (NS)
            old_grad = 0.
            for j in range(self.num_iter_T):

                T_matrix.requires_grad = True 

                # Obtaining the adversarial loss.
                loss, adv_loss, adv_images = self.get_loss(ori_images_float.clone(), labels, T_matrix.clone(), mean_channel)
    
                loss.backward()
                T_grad = T_matrix.grad.data

                # MI
                T_grad = T_grad / torch.abs(T_grad).mean([1,2], keepdim=True)
                T_grad = self.momentum * old_grad + T_grad
                old_grad = T_grad

                # Update the adversarial transfer matrix. 
                T_matrix = T_matrix.detach() - self.T_step * T_grad.sign()

            # Preserve the best adversarial examples in IR. 
            for i, el in enumerate(adv_loss):
                this_best_loss = best_loss_per_example[i]
                if (this_best_loss is None) or (this_best_loss[1] < float(el)):
                    best_loss_per_example[i] = (j, float(el))
                    # best_adv_loss_per_example[i] = adv_loss[i].item()
                    best_adv_images[i] = adv_images[i].clone().detach()

        return best_adv_images                    

    def get_loss(self, ori_images, labels, T_matrix, mean_channel):
        # Generating adversarial examples.
        adv_images_float = colour_transfer(ori_images, mean_channel, T_matrix)

        if self.color_mode == 'Lab':
            adv_images_int_lab = lab_type_convert(adv_images_float, 'float2int')
            adv_images = lab2rgb(adv_images_int_lab)

        adv_images = torch.clamp(adv_images, 0, 1)
        outputs = self._get_output(adv_images)

        # Calculate accuracy
        predict_y = torch.max(outputs, dim=1)[1]
        attack_succ = torch.ne(predict_y, labels)

        # Calculate adv loss
        adv_loss = torch.zeros(adv_images.shape[0], device=self.device)  # (N)
        for k in range(adv_images.shape[0]):
            adv_loss[k] = -cw_loss6(outputs[k,None], labels[k,None], kappa=float('inf'))
        loss_per_example = -adv_loss 
        loss = loss_per_example.sum()

        return loss, adv_loss, adv_images.clone().detach()

    def pick_candidate(self, mask):
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

    def get_target_image(self, ori_images, hist_p, tar_hist_imgs, mask, all_objs):
        """Merge the color distribution histogram according to p.
        """
        assert len(ori_images.shape) == 4
        assert len(hist_p.shape) == 3
        assert len(tar_hist_imgs.shape) == 5

        target_image = ori_images.clone().detach().permute(0,2,3,1)  # shape,(N,H,W,3)
        for i in range(ori_images.shape[0]):
            for j in range(len(all_objs[i])):
                obj = all_objs[i][j]
                BW = (mask[i]==obj)
                target_image[i][BW] = (hist_p[i][j][:,None,None] * tar_hist_imgs[i][:,BW]).sum(0)
        target_image = target_image.permute(0,3,1,2)
        return target_image

def spilt_color(color):
    """spilt color,

    Args:
        color (torch.tensor): shape(H*3+8,W*8+18,3)

    Returns:
        (torch.tensor): shape(20,img_size,img_size,3)
    """
    h = color.shape[0]
    w = color.shape[1]
    img_size = int(((h-8)/3))
    color_20 = torch.zeros((20, img_size, img_size, 3), device=color.device)
    h_start = 2
    w_start = 2
    for i in range(20):
        color_20[i] = color[h_start:h_start+img_size,w_start:w_start+img_size]
        w_start = w_start + img_size + 2
        if w_start >= w-1:
            w_start = 2
            h_start = h_start + img_size + 2
    
    return color_20