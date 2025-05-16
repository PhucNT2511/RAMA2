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
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad_(True)

    model.eval() # Ensure model is in evaluation mode
    
    with torch.enable_grad():  # Temporarily re-enable gradient calculation
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad() # It's good practice to zero gradients before backward pass
        loss.backward()

    # Collect the gradient of the loss w.r.t. the input image
    data_grad = images.grad.data

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = images + epsilon * data_grad.sign()
    # Clip to the specified range (defaults to [0,1] for backward compatibility)
    perturbed_image = torch.clamp(perturbed_image, min=clamp_min, max=clamp_max)
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
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    # The PGD attack starts with a random perturbation
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, min=clamp_min, max=clamp_max).detach()

    model.eval() # Ensure model is in evaluation mode

    for _ in range(num_iter):
        adv_images.requires_grad_(True)
        with torch.enable_grad():  # Temporarily re-enable gradient calculation for each iteration
            outputs = model(adv_images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            model.zero_grad() # Zero gradients before backward pass
            loss.backward()

        # Collect the gradient
        grad = adv_images.grad.data

        # Perform PGD step
        adv_images = adv_images.detach() + alpha * grad.sign()
        
        # Project perturbation back to L-infinity ball
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + delta, min=clamp_min, max=clamp_max).detach()
        
    return adv_images 