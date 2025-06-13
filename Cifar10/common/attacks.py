import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon, device, clamp_min=0, clamp_max=1):
    """
    Generates adversarial examples using the Fast Gradient Sign Method (FGSM).

    Args:
        model (nn.Module): The model to attack.
        images (torch.Tensor): Original images (batch_size, C, H, W).
        labels (torch.Tensor): True labels for the images.
        epsilon (float): Perturbation magnitude.
        device (torch.device): Device to perform computations on.
        clamp_min (float): Minimum value for image clipping.
        clamp_max (float): Maximum value for image clipping.

    Returns:
        torch.Tensor: Adversarial images.
    """
    images_for_attack = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    with torch.enable_grad(): # Ensure gradients are enabled
        images_for_attack.requires_grad = True
        outputs = model(images_for_attack)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        # It's good practice to zero_grad the model you are getting gradients from,
        # if it's a nn.Module. The ModelAttackWrapper handles this.
        model.zero_grad() 
        loss.backward()
        data_grad = images_for_attack.grad.data

    perturbed_image = images_for_attack + epsilon * data_grad.sign()
    # Clip perturbed image. Note: CIFAR100 script uses wide clamps for PGD.
    # This FGSM clamp might need adjustment if data isn't [0,1].
    perturbed_image = torch.clamp(perturbed_image, 0, 1) 
    return perturbed_image.detach()


def pgd_attack(model, images, labels, epsilon, alpha, num_iter, device, clamp_min=0, clamp_max=1):
    """
    Generates adversarial examples using Projected Gradient Descent (PGD).

    Args:
        model (nn.Module): The model to attack.
        images (torch.Tensor): Original images (batch_size, C, H, W).
        labels (torch.Tensor): True labels for the images.
        epsilon (float): Maximum perturbation magnitude (L-infinity norm).
        alpha (float): Step size for each iteration.
        num_iter (int): Number of PGD iterations.
        device (torch.device): Device to perform computations on.
        clamp_min (float): Minimum value for image clipping.
        clamp_max (float): Maximum value for image clipping.

    Returns:
        torch.Tensor: Adversarial images.
    """
    original_images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    # Initialize adversarial images from original images
    adv_images = original_images.clone().detach()
    
    # Start with a random perturbation within the epsilon ball
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    # Project to make sure initial perturbation is within bounds and respects epsilon constraint relative to original
    delta_init = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
    adv_images = torch.clamp(original_images + delta_init, min=clamp_min, max=clamp_max).detach()


    for _ in range(num_iter):
        with torch.enable_grad(): # Ensure gradients are enabled for each iteration
            adv_images.requires_grad = True
            outputs = model(adv_images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            model.zero_grad()
            loss.backward()
            grad = adv_images.grad.data
        
        # Perform PGD step
        adv_images = adv_images.detach() + alpha * grad.sign()
        
        # Project perturbation back to L-infinity ball around original images
        delta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        # Clip to valid image range
        adv_images = torch.clamp(original_images + delta, min=clamp_min, max=clamp_max).detach()
        
    return adv_images 