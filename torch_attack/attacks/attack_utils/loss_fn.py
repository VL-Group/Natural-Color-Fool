import torch

def cw_loss6(logits, labels, kappa=0.0, targeted=False):

    # get target logits
    target_logits = torch.gather(logits, 1, labels.view(-1,1))

    # get largest non-target logits
    max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1)
    top_max, second_max = max_2_logits.chunk(2, dim=1)
    top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
    targets_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
    targets_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
    max_other = targets_eq_max * second_max + targets_ne_max * top_max

    if targeted:
        # in targeted case, want to make target most likely
        f6 = torch.clamp(max_other - target_logits, min=-1 * kappa)
    else:
        # in NONtargeted case, want to make NONtarget most likely
        f6 = torch.clamp(target_logits - max_other, min=-1 * kappa)
    
    return f6.squeeze()