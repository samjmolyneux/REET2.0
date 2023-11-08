import random
import numpy as np
from reetoolbox.classification_optimisers import targeted_loss, untargeted_loss
import reetoolbox.semantic_segmentation_optimisers
import reetoolbox.classification_optimisers
import reetoolbox.constants
import reetoolbox.transforms
import os
import reetoolbox.hover_optimisers
from torch.utils.data import DataLoader
import reetoolbox.hover_loader
import torch
import time
from reetoolbox.maps import transform_dictionary, default_classification_optimiser, default_semantic_segmentation_optimiser, default_hover_optimiser, default_optimiser_params, default_transform_params, classification_loss, semantic_segmentation_loss, hover_net_loss


class ss_insert_trainer():

    def __init__(self, model, transform_list, model_input_shape, criterion, num_classes, device="cuda"):
        self.attack_list = []
        self.num_classes = num_classes
        criterion = semantic_segmentation_loss[criterion]
        for transform_name in transform_list:
            transform = transform_dictionary[transform_name]
            optimiser = default_semantic_segmentation_optimiser[transform_name]
            optimiser_params = default_optimiser_params[transform_name]
            trans_params = default_transform_params[transform_name]

            attack = optimiser(model, transform, optimiser_params, trans_params, criterion=criterion, device=device, model_input_shape=model_input_shape, num_classes=num_classes)
            self.attack_list.append(attack)

    #GET IT BACK TO DOING ATTACKS ON THE WHOLE BATCH
    def attack_images(self, imgs, masks, subset_size):
        attack = random.choice(self.attack_list)

        return attack.get_batch(imgs, targets=masks, reset_weights=True, return_masks=True)

    def attack_images_individually(self, imgs, masks, subset_size):
        one_hot_masks = torch.empty(imgs.shape[0], self.num_classes, imgs.shape[2], imgs.shape[3], dtype=torch.float32)
        for i in range(imgs.shape[0]):
          
            #NEED TO MAKE IT SO THAT WE CAN SAMPLE MORE THAN ONE IMAGE.
            sample_attacks = random.sample(self.attack_list, subset_size)
            for attack in sample_attacks:

                results_dict = attack.get_adversarial_images(imgs[i].numpy(), targets=masks[i].numpy(), reset_weights=True, return_masks=True)
                imgs[i] = results_dict["adv_inputs"]
                one_hot_masks[i] = results_dict["type_masks"]

        return imgs, masks


class hover_insert_trainer():

    def __init__(self, model, transform_list, model_input_shape, criterion, num_classes, device="cuda"):
        self.attack_list = []
        self.num_classes = num_classes
        criterion = hover_net_loss[criterion]
        for transform_name in transform_list:
            transform = transform_dictionary[transform_name]
            optimiser = default_hover_optimiser[transform_name]
            optimiser_params = default_optimiser_params[transform_name]
            trans_params = default_transform_params[transform_name]

            attack = optimiser(model, transform, optimiser_params, trans_params, criterion=criterion, device=device, model_input_shape=model_input_shape)
            self.attack_list.append(attack)

    #GET IT BACK TO DOING ATTACKS ON THE WHOLE BATCH
    def attack_images(self, imgs, masks, subset_size):
        attack = random.choice(self.attack_list)

        return attack.get_batch(imgs, masks_dict=masks, reset_weights=True)

    def attack_images_individually(self, imgs, masks, subset_size):
        one_hot_masks = torch.empty(imgs.shape[0], self.num_classes, imgs.shape[2], imgs.shape[3], dtype=torch.float32)
        for i in range(imgs.shape[0]):
          
            #NEED TO MAKE IT SO THAT WE CAN SAMPLE MORE THAN ONE IMAGE.
            sample_attacks = random.sample(self.attack_list, subset_size)
            for attack in sample_attacks:

                results_dict = attack.get_adversarial_images(imgs[i].numpy(), targets=masks[i].numpy(), reset_weights=True, return_masks=True)
                imgs[i] = results_dict["adv_inputs"]
                one_hot_masks[i] = results_dict["type_masks"]

        return imgs, masks



