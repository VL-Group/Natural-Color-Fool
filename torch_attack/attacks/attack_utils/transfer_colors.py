import torch

def torch_cov(X):
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    return 1 / (D - 1) * X @ X.transpose(-1, -2)

def MKL(A, B):
    """only torch>=1.10

    Args:
        A (torch.tensor): inpute shape:(3,3)
        B (torch.tensor): target shape:(3,3)
    """
    eps = 1e-10
    N = A.shape[0]
    Da2, Ua = torch.linalg.eig(A)
    Da2 = Da2.real.to(A.device)
    Ua = Ua.real.to(A.device)

    Da2[Da2<0] = 0
    Da = (Da2 + eps).sqrt().diag()  # (N,N)

    # U = Ua.T.contiguous()
    C = Da.mm(Ua.T).mm(B).mm(Ua).mm(Da)  # (N,N)

    Dc2_, Uc_ = torch.linalg.eig(C.clone())
    Dc2 = Dc2_.real.to(A.device)
    Uc = Uc_.real.to(A.device)
    Dc2[Dc2<0] = 0

    Dc = (Dc2 + eps).sqrt().diag()  # (N,N)

    Da_inv = (1/(Da.diag())).diag()  # (N,N)

    T = Ua.mm(Da_inv).mm(Uc).mm(Dc).mm(Uc.T).mm(Da_inv).mm(Ua.T)

    return T

def get_imgs_T(ori_images, target_images):
    """Calculate the transfer matrix based on the original and target images.

    Args:
        ori_images (torch.Tensor): original images. shape(N,3,H,W)
        target_images (torch.Tensor): shape(N,3,H,W)
    """
    n_imgs = ori_images.shape[0]
    ori_images_ = ori_images.reshape(n_imgs,3,-1)
    target_images_ = target_images.reshape(n_imgs,3,-1)

    cov_ori = torch_cov(ori_images_)
    cov_tar = torch_cov(target_images_)

    T_true = torch.zeros((n_imgs, 3, 3), device=target_images.device)
    for idx in range(n_imgs):
        T_true[idx] = MKL(cov_ori[idx], cov_tar[idx])
    
    return T_true

def colour_transfer(I0, mean_target, T):
    """colour transfer algorithm based on T

    Args:
        I0 (torch.FloatTensor): input images
        mean_target (torch.FloatTensor): mean of target images
        T (torch.FloatTensor): transfer matrix

    """
    assert len(I0.shape) == 4  # (N, 3, H,W)
    assert len(T.shape) == 3  # (N,3,3)
    
    X0 = I0.reshape(I0.shape[0], I0.shape[1], -1)  # (N, 3, HW)

    mX0 = X0.mean(dim=-1, keepdim=True)
    mX1 = mean_target.unsqueeze(-1)

    XR = []
    for i in range(I0.shape[0]):
        tmp = ((X0[i]-mX0[i]).T.mm(T[i])).T + mX1[i]
        XR.append(tmp)

    XR_ = torch.stack(XR, dim=0)
    IR = XR_.reshape(tuple(I0.shape))

    return IR

