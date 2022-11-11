import torch

##########################
# Pytorch: RGB <---> Lab #
##########################

def rgb2lab(imgs_rgb):
    """The pytorch version of RGB2Lab, which makes it easier to backward.

    Args:
        imgs_rgb (torch.float): A tensor image in RGB format with shape (N,3,H,W). [0, 1] 

    Returns:
        lab (torch.int): A tensor image in Lab format with shape (N,3,H,W). L[0,100], ab[-128,127]
    """
    assert len(imgs_rgb.shape) == 4

    # https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=XYZ#the-image-class
    Mps = torch.tensor(
        [[0.412453, 0.357580, 0.180423],
         [0.212671, 0.715160, 0.072169],
         [0.019334, 0.119193, 0.950227]])

    xn = 0.950456
    zn = 1.088754
    delta = 0

    def f(t):
        res = 7.787*t + 16/116
        index = t>0.008856
        res[index] = torch.pow(t[index], 1./3.)
        return res
    
    def gamma(t):
        res = t / 12.92
        index = t > 0.04045
        res[index] = torch.pow((t[index] + 0.055) / 1.055, 2.4)
        return res

    # convert RGB to CIEXYZ
    rgb = gamma(imgs_rgb)
    xyz = torch.zeros_like(rgb, dtype=torch.float32)
    xyz[:,0,:,:] = rgb[:,0,:,:]*Mps[0][0] + rgb[:,1,:,:]*Mps[0][1] + rgb[:,2,:,:]*Mps[0][2]  # X
    xyz[:,1,:,:] = rgb[:,0,:,:]*Mps[1][0] + rgb[:,1,:,:]*Mps[1][1] + rgb[:,2,:,:]*Mps[1][2]  # Y
    xyz[:,2,:,:] = rgb[:,0,:,:]*Mps[2][0] + rgb[:,1,:,:]*Mps[2][1] + rgb[:,2,:,:]*Mps[2][2]  # Z

    xyz[:,0,:,:] = xyz[:,0,:,:] / xn
    xyz[:,2,:,:] = xyz[:,2,:,:] / zn
    
    # convert CIEXYZ to CIELab
    lab = torch.zeros_like(rgb)
    lab[:,0,:,:] = 903.3 * xyz[:,1,:,:]
    y_index = xyz[:,1,:,:] > 0.008856
    lab[:,0,:,:][y_index] = 116 * torch.pow(xyz[:,1,:,:][y_index], 1./3.) - 16.  # L
    # lab[:,0,:,:] = 116 * f(xyz[:,1,:,:]) - 16.  # L
    lab[:,1,:,:] = 500*(f(xyz[:,0,:,:])-f(xyz[:,1,:,:])) + delta  # a
    lab[:,2,:,:] = 200*(f(xyz[:,1,:,:])-f(xyz[:,2,:,:])) + delta  # b

    return lab

def lab2rgb(imgs_lab):
    """The pytorch version of Lab2RGB, which makes it easier to backward.
    
    Args:
        imgs_lab (torch.int): A tensor image in Lab format with shape (N,3,H,W). L[0,100], ab[-128,127]
    
    Returns:
        imgs_rgb (torch.float): A tensor image in RGB format with shape (N,3,H,W). [0, 1] 
    """
    assert len(imgs_lab.shape) == 4
    # https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=XYZ#the-image-class
    Mps = torch.tensor(
        [[ 3.240479, -1.537150, -0.498535],
         [-0.969256,  1.875992,  0.041556],
         [ 0.055648, -0.204043,  1.057311]])

    xn = 0.950456
    zn = 1.088754
    delta = 0

    def anti_f(t):
        res = (t-16/116) / 7.787
        index = t > 0.206893
        res[index] = torch.pow(t[index], 3)
        return res

    def anti_gamma(t):
        res = t * 12.92
        index = t > 0.0031308
        res[index] = torch.pow(t[index], 1.0/2.4)*1.055 - 0.055
        return res
        
    lab = imgs_lab
    # convert CIELab to CIEXYZ
    xyz = torch.zeros_like(lab)
    xyz[:,1,:,:] = (lab[:,0,:,:] + 16) / 116.  # Y
    xyz[:,0,:,:] = lab[:,1,:,:] / 500 + xyz[:,1,:,:]  # X
    xyz[:,2,:,:] = xyz[:,1,:,:] - lab[:,2,:,:] / 200  # Z

    xyz[:,0,:,:] = anti_f(xyz[:,0,:,:])
    xyz[:,1,:,:] = anti_f(xyz[:,1,:,:])
    xyz[:,2,:,:] = anti_f(xyz[:,2,:,:])

    xyz[:,0,:,:] = xyz[:,0,:,:] * xn
    xyz[:,2,:,:] = xyz[:,2,:,:] * zn

    # convert CIEXYZ to RGB 
    rgb = torch.zeros_like(xyz)
    rgb[:,0,:,:] = xyz[:,0,:,:]*Mps[0][0] + xyz[:,1,:,:]*Mps[0][1] + xyz[:,2,:,:]*Mps[0][2]  # R
    rgb[:,1,:,:] = xyz[:,0,:,:]*Mps[1][0] + xyz[:,1,:,:]*Mps[1][1] + xyz[:,2,:,:]*Mps[1][2]  # G
    rgb[:,2,:,:] = xyz[:,0,:,:]*Mps[2][0] + xyz[:,1,:,:]*Mps[2][1] + xyz[:,2,:,:]*Mps[2][2]  # B
    rgb = anti_gamma(rgb)

    return torch.clamp(rgb, 0, 1)

def lab_type_convert(lab, mode="float2int"):
    if mode == "float2int":
        res = lab * torch.tensor([100, 255, 255], device=lab.device)[None,:,None,None] + torch.tensor([0, -128, -128], device=lab.device)[None,:,None,None]
    elif mode == "int2float":
        res = (lab + torch.tensor([0, 128, 128], device=lab.device)[None,:,None,None]) / torch.tensor([100, 255, 255], device=lab.device)[None,:,None,None]
    return res