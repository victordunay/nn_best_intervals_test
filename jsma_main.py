import math
import torch
from torch.autograd.gradcheck import zero_gradients
import numpy as np


def compute_jacobian(inputs, output):
    """
	:param inputs: Batch X Size (e.g. Depth X Width X Height)
	:param output: Batch X Classes
	:return: jacobian: Batch X Classes X Size
	"""
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)


def fgsm(inputs, targets, model, criterion, eps):
    """
	:param inputs: Clean samples (Batch X Size)
	:param targets: True labels
	:param model: Model
	:param criterion: Loss function
	:param gamma:
	:return:
	"""

    crafting_input = torch.autograd.Variable(inputs.clone(), requires_grad=True)
    crafting_target = torch.autograd.Variable(targets.clone())
    output = model(crafting_input)
    loss = criterion(output, crafting_target)
    if crafting_input.grad is not None:
        crafting_input.grad.data.zero_()
    loss.backward()
    crafting_output = crafting_input.data + eps * torch.sign(crafting_input.grad.data)

    return crafting_output


def saliency_map(jacobian, search_space, target_index, increasing=True):
    all_sum = torch.sum(jacobian, 1).squeeze()

    alpha = jacobian[0, target_index, :].squeeze()
    beta = all_sum - alpha

    if increasing:
        mask1 = torch.ge(alpha, 0.0)
        mask2 = torch.le(beta, 0.0)
    else:
        mask1 = torch.le(alpha, 0.0)
        mask2 = torch.ge(beta, 0.0)

    mask = torch.mul(torch.mul(mask1, mask2), search_space)

    if increasing:
        saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
    else:
        saliency_map = torch.mul(torch.mul(torch.abs(alpha), beta), mask.float())

    max_value, max_idx = torch.max(saliency_map, dim=0)

    return max_value, max_idx


def jsma(model, input_tensor, target_class, max_distortion, max_iter, lr):
    # Make a clone since we will alter the values
    input_features = torch.autograd.Variable(input_tensor.clone(), requires_grad=True)
    num_features = input_features.size(1)
    count = 0
    modified_pixels = []
    max_num_of_modified_pixels = 5
    # a mask whose values are one for feature dimensions in search space
    search_space = torch.ones(num_features).byte()
    if input_features.is_cuda:
        search_space = search_space.cuda()

    output = model(input_features)
    _, source_class = torch.max(output.data, 1)
    min_pixel_dist_val = 0
    max_pixel_dist_val = 1

    while (count < max_iter) and (source_class[0] != target_class) and (search_space.sum() != 0):
        # Calculate Jacobian
        jacobian = compute_jacobian(input_features, output)

        increasing_saliency_value, increasing_feature_index = saliency_map(jacobian, search_space, target_class,
                                                                           increasing=True)

        mask_zero = torch.gt(input_features.data.squeeze(), 0.0)
        search_space_decreasing = torch.mul(mask_zero, search_space)
        decreasing_saliency_value, decreasing_feature_index = saliency_map(jacobian, search_space_decreasing,
                                                                           target_class, increasing=False)

        # if increasing_saliency_value[0] == 0.0 and decreasing_saliency_value[0] == 0.0:
        if increasing_saliency_value.item() == 0.00000000 and decreasing_saliency_value.item() == 0.0000000:
            break

        # if increasing_saliency_value[0] > decreasing_saliency_value[0]:
        if increasing_saliency_value.item() > decreasing_saliency_value.item():
            if increasing_feature_index not in modified_pixels:
                modified_pixels.append(increasing_feature_index)
            num_of_modified_pixels = len(modified_pixels)
            if num_of_modified_pixels == max_num_of_modified_pixels + 1:
                modified_pixels.remove(increasing_feature_index)
                print("modified pixels fo target", target_class, " is ", modified_pixels)
                break
            input_features.data[0][increasing_feature_index] += lr
            #print("curr changed pix=", increasing_feature_index, " its val is ",
                  #input_features.data[0][increasing_feature_index])

            diff = abs(input_features.data[0][increasing_feature_index] - input_tensor[0][
                increasing_feature_index]) > max_distortion
            if input_features.data[0][increasing_feature_index] > 1 - lr or diff:
                search_space[increasing_feature_index] = 0

        else:
            input_features.data[0][decreasing_feature_index] -= lr
            #print("curr changed pix=", decreasing_feature_index, " its val is ",
                  #input_features.data[0][decreasing_feature_index])
            diff = abs(input_features.data[0][decreasing_feature_index] - input_tensor[0][
                decreasing_feature_index]) > max_distortion

            if input_features.data[0][decreasing_feature_index] < lr or diff:
                search_space[decreasing_feature_index] = 0

        output = model(input_features)
        _, source_class = torch.max(output.data, 1)

        count += 1

    return input_features
