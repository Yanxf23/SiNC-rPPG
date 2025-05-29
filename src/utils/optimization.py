import torch
import sys
from utils.losses import torch_power_spectral_density, normalize_psd


def select_optimization_step(arg_obj):
    optim_step = arg_obj.optimization_step
    if optim_step == 'unsupervised':
        return unsupervised_train_step
    elif optim_step == 'supervised':
        return supervised_train_step
    else:
        print('Unknown optimization_step: {optim_step}, exiting.')
        sys.exit(-1)


def select_validation_step(arg_obj):
    validation_step = arg_obj.validation_step
    if validation_step == 'unsupervised':
        return unsupervised_validation_step
    elif validation_step == 'supervised':
        return supervised_validation_step
    else:
        print('Unknown validation_step: {validation_step}, exiting.')
        sys.exit(-1)


def optimization_loop(model, train_loader, optimizer, optimization_step, criterions, logger, global_i, epoch, device, arg_obj):
    model.train()
    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        losses_dict = optimization_step(model, data, criterions, device, arg_obj)
        losses_dict['total'].backward()
        optimizer.step()
        global_i += 1
        logger.log(epoch, global_i, i, losses_dict)
    return model, optimizer, logger, global_i


# def unsupervised_validation_step(model, data, criterions, device, fps, arg_obj, return_pred=False):
#     frames = data[0].to(device)
#     # print("[DEBUG] frames dtype:", frames.dtype)
#     outputs = model(frames)
#     freqs, psd = torch_power_spectral_density(outputs, fps=fps, normalize=False, bandpass=False)
#     losses_dict = accumulate_validation_losses(freqs, psd, criterions, device, arg_obj)
#     if return_pred:
#         psd = normalize_psd(psd)
#         return losses_dict, outputs, freqs, psd
#     return losses_dict

# def unsupervised_validation_step(model, data, criterions, device, fps, arg_obj, return_pred=False):
#     frames = data[0].to(device)  # shape: (B, C, T, H, W)
#     outputs = model(frames)      # shape: (B, T)
#     freqs, psd = torch_power_spectral_density(outputs, fps=fps, normalize=False, bandpass=False)

#     # --- If the model has a mask predictor, get masks ---
#     if hasattr(model, 'mask_predictor'):
#         masks = model.mask_predictor(frames)  # shape: (B, 1, T, H, W)
#     else:
#         masks = None

#     losses_dict = accumulate_validation_losses(freqs, psd, masks, criterions, device, arg_obj)

#     if return_pred:
#         psd = normalize_psd(psd)
#         return losses_dict, outputs, freqs, psd

#     return losses_dict

def unsupervised_validation_step(model, data, criterions, device, fps, arg_obj, return_pred=False):
    frames = data[0].to(device)  # shape: (B, C, T, H, W)

    outputs, early_mask, mid_mask = model(frames)

    freqs, psd = torch_power_spectral_density(outputs, fps=fps, normalize=False, bandpass=False)

    losses_dict = accumulate_validation_losses(freqs, psd, criterions, device, arg_obj, early_mask, mid_mask, frames)

    if return_pred:
        psd = normalize_psd(psd)
        return losses_dict, outputs, freqs, psd

    return losses_dict


def supervised_train_step(model, data, criterions, device, arg_obj):
    frames, wave = (data[0].to(device), data[1].to(device))
    outputs = model(frames)
    loss = criterions['supervised'](outputs, wave)
    losses_dict = {'total': loss}
    return losses_dict


def supervised_validation_step(model, data, criterions, device, fps, arg_obj, return_pred=False):
    frames, wave = (data[0].to(device), data[1].to(device))
    outputs = model(frames)
    loss = criterions['supervised'](outputs, wave)
    losses_dict = {'total': loss}
    if return_pred:
        freqs, psd = torch_power_spectral_density(outputs, fps=fps, normalize=False, bandpass=False)
        psd = normalize_psd(psd)
        return losses_dict, outputs, freqs, psd
    return losses_dict


# def unsupervised_train_step(model, data, criterions, device, arg_obj):
#     fps = float(arg_obj.fps)
#     low_hz = float(arg_obj.low_hz)
#     high_hz = float(arg_obj.high_hz)
#     frames, speed = (data[0].to(device), data[3])
#     # print("[DEBUG] frames dtype:", frames.dtype)
#     predictions = model(frames)
#     predictions = add_noise_to_constants(predictions)
#     freqs, psd = torch_power_spectral_density(predictions, fps=fps, low_hz=low_hz, high_hz=high_hz, normalize=False, bandpass=False)
#     losses_dict = accumulate_unsupervised_losses(freqs, psd, speed, criterions, device, arg_obj)
#     return losses_dict

# def unsupervised_train_step(model, data, criterions, device, arg_obj):
#     fps = float(arg_obj.fps)
#     low_hz = float(arg_obj.low_hz)
#     high_hz = float(arg_obj.high_hz)
#     frames, speed = (data[0].to(device), data[3])  # frames: (B, C, T, H, W)

#     predictions = model(frames)  # (B, T)
#     predictions = add_noise_to_constants(predictions)

#     freqs, psd = torch_power_spectral_density(
#         predictions, fps=fps, low_hz=low_hz, high_hz=high_hz,
#         normalize=False, bandpass=False
#     )

#     # --- Optional: retrieve mask from model ---
#     masks = model.mask_predictor(frames) if hasattr(model, 'mask_predictor') else None

#     # --- Accumulate main losses + mask-based regularizers ---
#     losses_dict = accumulate_unsupervised_losses(freqs, psd, speed, criterions, device, arg_obj, masks=masks)

#     return losses_dict

def unsupervised_train_step(model, data, criterions, device, arg_obj):
    fps = float(arg_obj.fps)
    low_hz = float(arg_obj.low_hz)
    high_hz = float(arg_obj.high_hz)
    frames, speed = data[0].to(device), data[3]  # (B, C, T, H, W)

    # --- Forward pass and retrieve both masks ---
    predictions, early_mask, mid_mask = model(frames)

    predictions = add_noise_to_constants(predictions)

    freqs, psd = torch_power_spectral_density(
        predictions, fps=fps, low_hz=low_hz, high_hz=high_hz,
        normalize=False, bandpass=False
    )

    losses_dict = accumulate_unsupervised_losses(
        freqs, psd, speed, criterions, device, arg_obj,
        early_mask, mid_mask, frames
    )

    return losses_dict

# def accumulate_unsupervised_losses(freqs, psd, speed, criterions, device, arg_obj, masks=None):
#     total_loss = 0.0
#     losses_dict = {}

#     if 'b' in arg_obj.losses:
#         bandwidth_loss = criterions['bandwidth'](freqs, psd, low_hz=arg_obj.low_hz, high_hz=arg_obj.high_hz, device=device)
#         total_loss += arg_obj.bandwidth_scalar * bandwidth_loss
#         losses_dict['bandwidth'] = bandwidth_loss

#     if 's' in arg_obj.losses:
#         sparsity_loss = criterions['sparsity'](freqs, psd, low_hz=arg_obj.low_hz, high_hz=arg_obj.high_hz, device=device)
#         total_loss += arg_obj.sparsity_scalar * sparsity_loss
#         losses_dict['sparsity'] = sparsity_loss

#     # Optional: Mask entropy and L1 loss
#     if masks is not None:
#         entropy = -(masks * torch.log(masks + 1e-6) + (1 - masks) * torch.log(1 - masks + 1e-6))
#         mask_entropy_loss = entropy.mean()
#         mask_l1_loss = masks.mean()

#         total_loss += arg_obj.mask_entropy_scalar * mask_entropy_loss
#         total_loss += arg_obj.mask_l1_scalar * mask_l1_loss

#         losses_dict['mask_entropy'] = mask_entropy_loss
#         losses_dict['mask_l1'] = mask_l1_loss

#     losses_dict['total'] = total_loss
#     return losses_dict

def loss(x, k, lower_bound, upper_bound):
    return torch.log1p(torch.exp(k * (lower_bound - x))) + torch.log1p(torch.exp(k * (x - upper_bound)))

def mask_losses(mask, tag, arg_obj):
    # --- Entropy ---
    entropy = -(mask * torch.log(mask + 1e-6) + (1 - mask) * torch.log(1 - mask + 1e-6)).mean()

    B, _, T, H, W = mask.shape
    mask_reshaped = mask.view(B, T, -1)
    mu_t = mask_reshaped.mean(dim=2)    # (B, T) per-frame per-sample
    mu_t = mu_t.mean(dim=0)             # (T,) per-frame average across batch

    lower_bound = 0.05
    upper_bound = 0.20
    k = 10  # sharpness parameter (increase for sharper transitions)
    l1 = loss(mu_t, k, lower_bound, upper_bound).mean()

    return {
        f'{tag}_entropy': entropy,
        f'{tag}_l1': l1,
        f'{tag}_loss': arg_obj.__dict__[f'{tag}_entropy_scalar'] * entropy +
                       arg_obj.__dict__[f'{tag}_l1_scalar'] * l1
    }

def accumulate_unsupervised_losses(freqs, psd, speed, criterions, device, arg_obj, early_mask, mid_mask, input):
    total_loss = 0.0
    losses_dict = {}

    if 'b' in arg_obj.losses:
        bandwidth_loss = criterions['bandwidth'](freqs, psd, low_hz=arg_obj.low_hz, high_hz=arg_obj.high_hz, device=device)
        total_loss += arg_obj.bandwidth_scalar * bandwidth_loss
        losses_dict['bandwidth'] = bandwidth_loss

    if 's' in arg_obj.losses:
        sparsity_loss = criterions['sparsity'](freqs, psd, low_hz=arg_obj.low_hz, high_hz=arg_obj.high_hz, device=device)
        total_loss += arg_obj.sparsity_scalar * sparsity_loss
        losses_dict['sparsity'] = sparsity_loss

    if 'm' in arg_obj.losses:
        mask_contrast_loss = criterions['maskcontrast'](
            x=input,              # (B, C, T, H, W)
            mask=early_mask,          # (B, 1, T, H, W)
            fps=arg_obj.fps,
            device=device
        )
        # print(f"[DEBUG] mask_contrast_loss: {mask_contrast_loss.item()}")
        total_loss += arg_obj.mask_contrast_scalar * mask_contrast_loss
        losses_dict['mask_contrast'] = mask_contrast_loss

    # Early mask regularization
    if arg_obj.early_mask:
        losses = mask_losses(early_mask, tag='mask', arg_obj=arg_obj)
        total_loss += losses['mask_loss']
        losses_dict.update(losses)

    # Mid-layer mask regularization
    if arg_obj.mid_mask:
        losses = mask_losses(mid_mask, tag='mid_mask', arg_obj=arg_obj)
        total_loss += losses['mid_mask_loss']
        losses_dict.update(losses)

    losses_dict['total'] = total_loss
    return losses_dict
    
# def accumulate_unsupervised_losses(freqs, psd, speed, criterions, device, arg_obj):
#     criterions_str = arg_obj.losses
#     low_hz = float(arg_obj.low_hz)
#     high_hz = float(arg_obj.high_hz)
#     total_loss = 0.0
#     losses_dict = {}
#     if 'b' in criterions_str:
#         bandwidth_loss = criterions['bandwidth'](freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device=device)
#         total_loss += (arg_obj.bandwidth_scalar*bandwidth_loss)
#         losses_dict['bandwidth'] = bandwidth_loss
#     if 's' in criterions_str:
#         sparsity_loss = criterions['sparsity'](freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device=device)
#         total_loss += (arg_obj.sparsity_scalar*sparsity_loss)
#         losses_dict['sparsity'] = sparsity_loss
#     if 'v' in criterions_str:
#         variance_loss = criterions['variance'](freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device=device)
#         total_loss += (arg_obj.variance_scalar*variance_loss)
#         losses_dict['variance'] = variance_loss
#     losses_dict['total'] = total_loss
#     return losses_dict


# def accumulate_validation_losses(freqs, psd, criterions, device, arg_obj):
#     criterions_str = arg_obj.validation_loss
#     low_hz = float(arg_obj.low_hz)
#     high_hz = float(arg_obj.high_hz)
#     total_loss = 0.0
#     losses_dict = {}
#     if 'b' in criterions_str:
#         bandwidth_loss = criterions['bandwidth'](freqs, psd, low_hz=low_hz, high_hz=high_hz, device=device)
#         total_loss = total_loss + (arg_obj.bandwidth_scalar*bandwidth_loss)
#         losses_dict['bandwidth'] = bandwidth_loss
#     if 's' in criterions_str:
#         sparsity_loss = criterions['sparsity'](freqs, psd, low_hz=low_hz, high_hz=high_hz, device=device)
#         total_loss = total_loss + (arg_obj.sparsity_scalar*sparsity_loss)
#         losses_dict['sparsity'] = sparsity_loss
#     losses_dict['total'] = total_loss
#     return losses_dict

# def accumulate_validation_losses(freqs, psd, masks, criterions, device, arg_obj):
#     criterions_str = arg_obj.validation_loss
#     low_hz = float(arg_obj.low_hz)
#     high_hz = float(arg_obj.high_hz)
#     total_loss = 0.0
#     losses_dict = {}

#     if 'b' in criterions_str:
#         bandwidth_loss = criterions['bandwidth'](freqs, psd, low_hz=low_hz, high_hz=high_hz, device=device)
#         total_loss += arg_obj.bandwidth_scalar * bandwidth_loss
#         losses_dict['bandwidth'] = bandwidth_loss

#     if 's' in criterions_str:
#         sparsity_loss = criterions['sparsity'](freqs, psd, low_hz=low_hz, high_hz=high_hz, device=device)
#         total_loss += arg_obj.sparsity_scalar * sparsity_loss
#         losses_dict['sparsity'] = sparsity_loss

#     # Optional mask-based losses
#     if masks is not None:
#         # Entropy: encourage confident attention
#         entropy = -(masks * torch.log(masks + 1e-6) + (1 - masks) * torch.log(1 - masks + 1e-6))
#         mask_entropy_loss = entropy.mean()
#         total_loss += arg_obj.mask_entropy_scalar * mask_entropy_loss
#         losses_dict['mask_entropy'] = mask_entropy_loss

#         # L1: encourage sparse mask
#         mask_l1_loss = masks.mean()
#         total_loss += arg_obj.mask_l1_scalar * mask_l1_loss
#         losses_dict['mask_l1'] = mask_l1_loss

#     losses_dict['total'] = total_loss
#     return losses_dict

def accumulate_validation_losses(freqs, psd, criterions, device, arg_obj, early_mask, mid_mask, input):
    criterions_str = arg_obj.validation_loss
    low_hz = float(arg_obj.low_hz)
    high_hz = float(arg_obj.high_hz)
    total_loss = 0.0
    losses_dict = {}

    if 'b' in criterions_str:
        bandwidth_loss = criterions['bandwidth'](freqs, psd, low_hz=low_hz, high_hz=high_hz, device=device)
        total_loss += arg_obj.bandwidth_scalar * bandwidth_loss
        losses_dict['bandwidth'] = bandwidth_loss

    if 's' in criterions_str:
        sparsity_loss = criterions['sparsity'](freqs, psd, low_hz=low_hz, high_hz=high_hz, device=device)
        total_loss += arg_obj.sparsity_scalar * sparsity_loss
        losses_dict['sparsity'] = sparsity_loss

    # ✅ Add contrast loss during validation
    if 'm' in criterions_str:
        mask_contrast_loss = criterions['maskcontrast'](
            x=input,              # (B, C, T, H, W)
            mask=early_mask,          # (B, 1, T, H, W)
            fps=arg_obj.fps,
            device=device
        )
        total_loss += arg_obj.mask_contrast_scalar * mask_contrast_loss
        losses_dict['mask_contrast'] = mask_contrast_loss

    if arg_obj.early_mask:
        losses = mask_losses(early_mask, tag='mask', arg_obj=arg_obj)
        total_loss += losses['mask_loss']
        losses_dict.update(losses)

    if arg_obj.mid_mask:
        losses = mask_losses(mid_mask, tag='mid_mask', arg_obj=arg_obj)
        total_loss += losses['mid_mask_loss']
        losses_dict.update(losses)

    losses_dict['total'] = total_loss
    return losses_dict


def add_noise_to_constants(predictions):
    B,T = predictions.shape
    for b in range(B):
        if torch.allclose(predictions[b][0], predictions[b]): # constant volume
            predictions[b] = torch.rand(T) - 0.5
    return predictions
