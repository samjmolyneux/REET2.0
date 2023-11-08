from abc import ABC, abstractmethod
import torch
from reetoolbox.constraints import Constraints
import torch.nn.functional as F

from reetoolbox.utils import cropping_center, gen_targets, msge_loss, mse_loss, dice_loss, xentropy_loss, crop_to_shape, crop_op, gen_truth_inst_info, fix_mirror_padding, crop_center_2_3
import copy
import numpy as np
from collections import OrderedDict

from reetoolbox.hover_metrics import remap_label

loss_func_dict = {
    "bce": xentropy_loss,
    "dice": dice_loss,
    "mse": mse_loss,
    "msge": msge_loss,
}

#COULD DO ATTACKS ON SPECIFIC BRANCHES
def hover_untargeted_loss(pred_dict, true_dict):
    loss = 0
    loss_opts = {  
        "np": {"bce": 1, "dice": 1},
        "hv": {"mse": 1, "msge": 1},
        "tp": {"bce": 1, "dice": 1},
    }
    

    for branch_name in pred_dict.keys():
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
            loss_args = [true_dict[branch_name], pred_dict[branch_name]]
            #What is happening here?
            if loss_name == "msge":
                loss_args.append(true_dict["np"][..., 1])
            term_loss = loss_func(*loss_args)
            loss -= loss_weight * term_loss
    return loss

def hover_untargeted_loss_by_batch(pred_dict, true_dict, device):
    loss = 0
    loss_opts = {  
        "np": {"bce": 1, "dice": 1},
        "hv": {"mse": 1, "msge": 1},
        "tp": {"bce": 1, "dice": 1},
    }
    
    #I THINK THIS SECTION IS RIGHT BUT DEFINITELY NEEDS DOUBLE CHECKING
    batch_size = true_dict["np"].shape[0]
    loss = torch.zeros(batch_size).to(device)

    for branch_name in pred_dict.keys():
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
        
            for img_idx in range(batch_size):
                loss_args = [true_dict[branch_name][img_idx].unsqueeze(0), pred_dict[branch_name][img_idx].unsqueeze(0)]

                if loss_name == "msge":
                    loss_args.append(true_dict["np"][img_idx, ..., 1].unsqueeze(0))
                
                term_loss = loss_func(*loss_args)
                loss[img_idx] -= loss_weight * term_loss
    
    return loss


def untargeted_hv_loss(pred_dict, true_dict):
    loss_args = [true_dict["hv"], pred_dict["hv"]]
    loss = 0
    loss -= mse_loss(*loss_args)
    loss_args.append(true_dict["np"][..., 1])
    loss -= msge_loss(*loss_args)  


    return loss

def radian_loss(pred_dict, true_dict):

    #NEED TO INCLUDE ONLY ONE OF THEM
    loss_args = [true_dict["hv"], pred_dict["hv"]]
    loss = 0
    loss -= mse_loss(*loss_args)
    loss_args.append(true_dict["np"][..., 1])
    loss -= msge_loss(*loss_args)  

    return loss
    



#ONLY TAKE 1 IMAGE AT A TIME FOR TIME BEING
class HoVer_Optimiser(ABC):
    def __init__(self, model, Transform, hyperparameters, transform_hyperparameters, criterion=hover_untargeted_loss,
                 device="cuda:0", model_input_shape=(270,270), model_output_shape=(80,80), background_index=0):
        self.model = model
        self.Transform = Transform
        self.hyperparameters = hyperparameters
        self.transform_hyperparameters = transform_hyperparameters
        self.device = device
        self.transform = None
        self.criterion = criterion
        #MAKE THE FOLLOWING CHANGEABLE
        self.post_transform_shape = model_input_shape
        self.eval_shape = model_output_shape
        self.background_index=background_index


    @abstractmethod
    def get_adversarial_images(self):
        pass
    

#CREATE AN OPTION FOR A PARALLEL MODEL?
#COULD MOVE THE CLONING OUTSIDE OF THE TRANSFORMS SO THAT THEY ARE NEVER ON THE GPU, THAT WAY WE JUST ACCESS THEM FROM THE EVALUATION, 
#THAT PROBABLY MAKES WAY MORE SENSE
class HoVer_PGD(HoVer_Optimiser):

    def get_adversarial_images(self, inputs, targets=None, reset_weights=True, return_masks=False, return_transform_weights = False):
        
        self.best_loss = None

        inst_maps = np.expand_dims(targets[...,0], axis=0)
        type_maps = np.expand_dims(targets[...,1], axis=0)

        #We can use the fact that no pgd transforms add black space
        inputs = cropping_center(inputs, self.post_transform_shape, batch=True)
        inputs = torch.from_numpy(inputs) #THIS IS ONLY TEMPORARY

        inst_maps = cropping_center(inst_maps, self.post_transform_shape, batch=True)        
        type_maps = cropping_center(type_maps, self.eval_shape, batch=True)
        
        inputs = inputs.to(self.device).type(torch.float32).contiguous()
        
        true_dict = gen_hv_np_maps(inst_maps, self.eval_shape)        
        true_dict["tp"] = gen_tp_onehot_map(type_maps, self.model.nr_types).unsqueeze(0)

        
        #Set PGD hyperparamters
        epsilon = self.hyperparameters["epsilon"]
        steps = self.hyperparameters["steps"]
        constraint = self.hyperparameters["constraint"]
        C = self.hyperparameters["C"]
        input_range = self.hyperparameters["input_range"]

        #Get constraint function and instatiate a Contraints object
        if constraint is not None:
            constraint_func = getattr(Constraints, constraint)
            constraints = Constraints


        #Set the transform weights if they haven't been set.
        #If they are set but reset_weights is true, reset them to default.
        if self.transform is None or reset_weights:
            self.transform = self.Transform(input_shape=inputs.shape, device=self.device,
                                            **self.transform_hyperparameters)

        in_train_mode = self.model.training
        self.model.eval()
        
        #Remember whether each paramter of model requires_grad, then set them all to false for optimizing transform
        grads = []
        for param in self.model.parameters():
            grads.append(param.requires_grad)
            param.requires_grad = False

        opt = torch.optim.RMSprop([self.transform.weights], lr=epsilon)
        
        #Loop to optimize the adversarial attack
        for i in range(steps):


            if targets is None:
                ##MAKE IT SELECT CURRENT PREDICTION MASKS AS LABELS.
                pass
            
            #Transform inputs and get model results of transformed inputs
            adv_inputs = self.transform.forward(inputs)
            adv_pred_dict = self.model(adv_inputs)  
            adv_pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in adv_pred_dict.items()]
            )
            adv_pred_dict["np"] = F.softmax(adv_pred_dict["np"], dim=-1)
            adv_pred_dict["tp"] = F.softmax(adv_pred_dict["tp"], dim=-1)

            loss = self.criterion(adv_pred_dict, true_dict)

            #Only takes one at a time
            loss.backward()
            opt.step()

            #Project data back into the valid range
            if constraint is not None:
                self.transform.weights = constraint_func(constraints, self.transform.weights,
                                                         self.transform.base_weights, C)
            
            opt.zero_grad()

            if self.best_loss is None or self.best_adv_inputs is None:
                self.best_loss = loss.clone().detach()
                self.best_adv_inputs = adv_inputs.clone().detach()
                #THESE ARE NOT THE CORRECT TRANSFORM WEIGHTS
                self.best_transform_weights = self.transform.weights.clone().detach()
            else:
                if loss < self.best_loss:
                    self.best_loss = loss.clone().detach()
                    self.best_adv_inputs = adv_inputs.clone().detach()
                    #THESE ARE NOT THE CORRECT TRANSFORM WEIGHTS
                    self.best_transform_weights = self.transform.weights.clone().detach()

        #PRETTY SURE THIS LINE IS NOT NEEDED
        adv_inputs = self.transform.forward(inputs).detach()
        
        #Restore requires_grad for each paramter of the model now that transform is optimized.
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = grads[i]

        #Restore models initial mode
        if in_train_mode:
            self.model.train()

        #SHOULD CHANGE THESE SO THAT THEY ACTUALLY RETURN BEST AND DO RANDOM REPEATS.
        return_dict = {"adv_inputs" : self.best_adv_inputs}
        if return_masks:
            #PGD doesnt change the mask, so just return the original
            return_dict["truth_masks_dict"] = true_dict
            inst_maps = cropping_center(inst_maps, self.eval_shape, batch=True).squeeze(0)
            inst_maps = fix_mirror_padding(inst_maps)
            inst_maps = remap_label(inst_maps)

            type_maps = cropping_center(type_maps, self.eval_shape, batch=True).squeeze(0)
            return_dict["truth_inst_map"] = inst_maps
            return_dict["truth_inst_info"] = gen_truth_inst_info(inst_maps, type_maps)

        if return_transform_weights:
            return_dict["transform_weights"] = self.best_transform_weights

        return return_dict



    def get_batch(self, inputs, masks_dict, reset_weights=True):
        

        self.best_loss = None

        inputs = cropping_center(inputs, self.post_transform_shape, batch=True)

        masks_dict["np"] = crop_center_2_3(masks_dict["np"], self.eval_shape)
        masks_dict["hv"] = crop_center_2_3(masks_dict["hv"], self.eval_shape)
        masks_dict["tp"] = crop_center_2_3(masks_dict["tp"], self.eval_shape)

        
        #Set PGD hyperparamters
        epsilon = self.hyperparameters["epsilon"]
        steps = self.hyperparameters["steps"]
        constraint = self.hyperparameters["constraint"]
        C = self.hyperparameters["C"]
        input_range = self.hyperparameters["input_range"]

        #Get constraint function and instatiate a Contraints object
        if constraint is not None:
            constraint_func = getattr(Constraints, constraint)
            constraints = Constraints


        #Set the transform weights if they haven't been set.
        #If they are set but reset_weights is true, reset them to default.
        if self.transform is None or reset_weights:
            self.transform = self.Transform(input_shape=inputs.shape, device=self.device,
                                            **self.transform_hyperparameters)

        in_train_mode = self.model.training
        self.model.eval()
        
        #Remember whether each paramter of model requires_grad, then set them all to false for optimizing transform
        grads = []
        for param in self.model.parameters():
            grads.append(param.requires_grad)
            param.requires_grad = False

        opt = torch.optim.RMSprop([self.transform.weights], lr=epsilon)
        #opt = torch.optim.Adam([self.transform.weights], lr=epsilon)
        
        #Loop to optimize the adversarial attack
        for i in range(steps):


            
            #Transform inputs and get model results of transformed inputs
            adv_inputs = self.transform.forward(inputs)
            adv_pred_dict = self.model(adv_inputs)  
            adv_pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in adv_pred_dict.items()]
            )
            adv_pred_dict["np"] = F.softmax(adv_pred_dict["np"], dim=-1)
            adv_pred_dict["tp"] = F.softmax(adv_pred_dict["tp"], dim=-1)

            loss = self.criterion(adv_pred_dict, masks_dict, self.device)


            #Only takes one at a time
            loss.backward(torch.ones_like(loss))
            opt.step()

            #Project data back into the valid range
            if constraint is not None:
                self.transform.weights = constraint_func(constraints, self.transform.weights,
                                                         self.transform.base_weights, C)
            
            opt.zero_grad()
                                        
            if self.best_loss is None or self.best_adv_inputs is None:
                self.best_loss = loss.clone()
                self.best_adv_inputs = adv_inputs.clone().detach()
            else:
                # Go over all losses in current batch
                for j, input_loss in enumerate(loss):
                    #change transformed image and loss if they are the current best.
                    if input_loss < self.best_loss[j]:
                        self.best_adv_inputs[j] = adv_inputs[j].clone().detach()
                        self.best_loss[j] = input_loss.clone().detach()
        
        #Restore requires_grad for each paramter of the model now that transform is optimized.
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = grads[i]

        #Restore models initial mode
        if in_train_mode:
            self.model.train()



        return self.best_adv_inputs, masks_dict









class HoVer_StochasticSearch(HoVer_Optimiser):


    #Background index specifies which index of the mask is used to specify background.
    def get_adversarial_images(self, inputs, targets=None, reset_weights=True, return_masks = False, return_transform_weights = False):
        self.best_loss = None
        

        #SHOULD CHANGE MAPS FORWARD SO THAT WE DONT HAVE TO DO THIS
        inst_maps = np.expand_dims(targets[...,0], axis=0)
        type_maps = np.expand_dims(targets[...,1], axis=0)

        #We can use the fact that no pgd transforms add black space
        # inputs = cropping_center(inputs, self.post_transform_shape, batch=True)

        inputs = torch.from_numpy(inputs) #THIS IS ONLY TEMPORARY     
        inputs = inputs.to(self.device).contiguous().type(torch.float32)

        #Set up the base tp map for gradient descent.
        true_base_tp_onehot = gen_tp_onehot_map(type_maps, self.model.nr_types).permute(2,0,1).unsqueeze(0)

        #Get the optimiser hyperparameters
        samples = self.hyperparameters["samples"]
        weight_ranges = self.hyperparameters["weight_ranges"]
        input_range = self.hyperparameters["input_range"]
        

        #Define transform
        if self.transform is None or reset_weights:
            self.transform = self.Transform(input_shape=inputs.shape, device=self.device,
                                            **self.transform_hyperparameters)
            self.best_loss = None
            self.best_adv_inputs = inputs

        #Store the mode of the model 
        in_train_mode = self.model.training
        self.model.eval()

        #Memorize the gradient for each parameter
        grads = []
        for param in self.model.parameters():
            grads.append(param.requires_grad)
            param.requires_grad = False


        count = 0
        with torch.no_grad():
            #iterate over the number of steps given (samples)
            for i in range(samples):
                #Initialize the transform weights
                for j, weight_name in enumerate(weight_ranges):
                    self.transform.weights[weight_name] = torch.FloatTensor(
                        *self.transform.weights[weight_name].shape).uniform_(*weight_ranges[weight_name]).to(self.device)
                    
                if targets is None:
                    #Make it select the current prediction as label
                    pass
                
                #Transform inputs and get model results of transformed inputs
                adv_inputs = self.transform.forward(inputs)
                adv_inputs = cropping_center(adv_inputs, self.post_transform_shape, batch=True)
                

                adv_pred_dict = self.model(adv_inputs)
                adv_pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in adv_pred_dict.items()]
                )
                adv_pred_dict["np"] = F.softmax(adv_pred_dict["np"], dim=-1)
                adv_pred_dict["tp"] = F.softmax(adv_pred_dict["tp"], dim=-1)
                
                adv_true_inst = self.transform.maps_forward(inst_maps)
                adv_true_inst = cropping_center(adv_true_inst, self.post_transform_shape, batch=True)

                # title_str = "count: ", count
                # import matplotlib.pyplot as plt
                # fig, axs = plt.subplots(1, 2, figsize=(16, 8))
                # axs[0].imshow(adv_inputs.detach().cpu().squeeze().permute(1,2,0).numpy().astype("uint8"))
                # axs[1].imshow(adv_true_inst.squeeze().astype("uint8"))
                # axs[0].set_title(title_str)
                # plt.savefig("zoom_test.png")
                # breakpoint()
                # count+=1



                adv_true_dict = gen_hv_np_maps(adv_true_inst, self.eval_shape)

                
                #CHECK THAT TP IS TRANSFORMED PROPERLY TOO
                adv_true_dict["tp"] = self.transform.mask_forward(true_base_tp_onehot)
                adv_true_dict["tp"] = cropping_center(adv_true_dict["tp"], self.eval_shape, batch=True).permute(0,2,3,1)

                
                loss = self.criterion(adv_pred_dict, adv_true_dict)

                #WE ARE CLONING BECAUSE THOSE VALUES ARE ABOUT TO CHANGE, IS THERE ANYWAY THAT WE COULD SAVE THE TRANSFORM INSTEAD OF THE IMGAES
                if self.best_loss is None or self.best_adv_inputs is None:
                    self.best_loss = loss.clone().detach()
                    self.best_adv_inputs = adv_inputs.clone().detach()
                    self.best_adv_true_dict = copy.deepcopy(adv_true_dict)
                    self.best_transform = copy.deepcopy(self.transform)
                else:
                    if loss < self.best_loss:
                        self.best_loss = loss.clone().detach()
                        self.best_adv_inputs = adv_inputs.clone().detach()
                        self.best_adv_true_dict = copy.deepcopy(adv_true_dict)
                        self.best_transform = copy.deepcopy(self.transform)


        #Turn gradients back on
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = grads[i]

        #Return model to original mode.
        if in_train_mode:
            self.model.train()


        return_dict = {"adv_inputs" : self.best_adv_inputs}
        if return_masks:
            return_dict["truth_masks_dict"] = self.best_adv_true_dict


            #CHECK IF INST_MAPS AND TYPE MAPS ARE EVER CHANGED BY THE THINGS WE PUT THEM THROUGH, 
            #TYPE MAPS COULD BE CHANGED BY THE MASK_FORWARD, SO DOUBLE CHECK

            inst_maps = self.best_transform.maps_forward(inst_maps)
            inst_maps = cropping_center(inst_maps, self.eval_shape, batch=True)
            inst_maps = inst_maps.squeeze(0)
            inst_maps = fix_mirror_padding(inst_maps)
            inst_maps = remap_label(inst_maps)
            
            type_maps = self.best_transform.maps_forward(type_maps).squeeze(0)
            type_maps = cropping_center(type_maps, self.eval_shape, batch=True)

            return_dict["truth_inst_map"] = inst_maps
            return_dict["truth_inst_info"] = gen_truth_inst_info(inst_maps, type_maps)


                        
        if return_transform_weights:
            return_dict["transform_weights"] = self.best_transform.weights

        return return_dict











    def get_batch(self, inputs, masks_dict, reset_weights=True):
        self.best_loss = None
        
        inst_masks = masks_dict["inst"].to(self.device)


        true_base_tp_onehot = masks_dict["tp"]

        #Get the optimiser hyperparameters
        samples = self.hyperparameters["samples"]
        weight_ranges = self.hyperparameters["weight_ranges"]
        input_range = self.hyperparameters["input_range"]
        

        #Define transform
        if self.transform is None or reset_weights:
            self.transform = self.Transform(input_shape=inputs.shape, device=self.device,
                                            **self.transform_hyperparameters)
            self.best_loss = None
            self.best_adv_inputs = inputs

        #Store the mode of the model 
        in_train_mode = self.model.training
        self.model.eval()

        #Memorize the gradient for each parameter
        grads = []
        for param in self.model.parameters():
            grads.append(param.requires_grad)
            param.requires_grad = False


        count = 0
        with torch.no_grad():
            #iterate over the number of steps given (samples)
            for i in range(samples):
                #Initialize the transform weights
                for j, weight_name in enumerate(weight_ranges):
                    self.transform.weights[weight_name] = torch.FloatTensor(
                        *self.transform.weights[weight_name].shape).uniform_(*weight_ranges[weight_name]).to(self.device)
                    

                #Transform inputs and get model results of transformed inputs
                adv_inputs = self.transform.forward(inputs)
                adv_inputs = cropping_center(adv_inputs, self.post_transform_shape, batch=True)
                

                adv_pred_dict = self.model(adv_inputs)
                adv_pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in adv_pred_dict.items()]
                )
                adv_pred_dict["np"] = F.softmax(adv_pred_dict["np"], dim=-1)
                adv_pred_dict["tp"] = F.softmax(adv_pred_dict["tp"], dim=-1)


                adv_inst_masks = self.transform.mask_forward(inst_masks.unsqueeze(1), assign_back=False).squeeze(1)
                adv_inst_masks = cropping_center(adv_inst_masks, self.post_transform_shape, batch=True)

                adv_true_dict = gen_batch_hv_np_maps(adv_inst_masks, self.eval_shape)


                adv_true_dict["tp"] = self.transform.mask_forward(true_base_tp_onehot.permute(0,3,1,2))
                adv_true_dict["tp"] = cropping_center(adv_true_dict["tp"], self.eval_shape, batch=True).permute(0,2,3,1)


                
                loss = self.criterion(adv_pred_dict, adv_true_dict, self.device)


                if self.best_loss is None or self.best_adv_inputs is None:
                    self.best_loss = loss.clone()
                    self.best_adv_inputs = adv_inputs.clone().detach()
                    self.best_adv_true_dict = copy.deepcopy(adv_true_dict)
                else:
                    # Go over all losses in current batch
                    for j, input_loss in enumerate(loss):
                        #change transformed image and loss if they are the current best.
                        if input_loss < self.best_loss[j]:
                            self.best_adv_inputs[j] = adv_inputs[j].clone().detach()
                            self.best_loss[j] = input_loss.clone().detach()
                            self.best_adv_true_dict["np"][j] = adv_true_dict["np"][j]
                            self.best_adv_true_dict["hv"][j] = adv_true_dict["hv"][j]
                            self.best_adv_true_dict["tp"][j] = adv_true_dict["tp"][j]


        #Turn gradients back on
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = grads[i]

        #Return model to original mode.
        if in_train_mode:
            self.model.train()




        return self.best_adv_inputs, self.best_adv_true_dict




def gen_tp_onehot_map(type_maps, nr_types, to_device=True):
    tp = torch.from_numpy(type_maps) #CHANGED THIS ONE TOO
    if to_device:
        tp = torch.squeeze(tp).to("cuda").type(torch.int64)
    tp = F.one_hot(tp, num_classes=nr_types)
    return tp.type(torch.float32)

def gen_hv_np_maps(inst_maps, eval_shape, to_device=True):
    target_dict = gen_targets(inst_maps.squeeze(), eval_shape, batch=False)

    true_np = torch.from_numpy(target_dict["np_map"]).unsqueeze(0)#CHANGED THIS
    true_hv = torch.from_numpy(target_dict["hv_map"]).unsqueeze(0)#CHANGED THIS

    if to_device:
        true_np = true_np.to("cuda").type(torch.int64)
        true_hv = true_hv.to("cuda").type(torch.float32)

    true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
    true_dict = {
    "np": true_np_onehot,
    "hv": true_hv,
    }
    return true_dict

from reetoolbox.utils import gen_targets_batch

def gen_batch_hv_np_maps(inst_maps, eval_shape, to_device=True):
    inst_maps = inst_maps.detach().cpu().numpy()

    target_dict = gen_targets_batch(inst_maps, eval_shape)


    true_np = torch.from_numpy(target_dict["np_map"])
    true_hv = torch.from_numpy(target_dict["hv_map"])

    if to_device:
        true_np = true_np.to("cuda").type(torch.int64)
        true_hv = true_hv.to("cuda").type(torch.float32)

    true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
    true_dict = {
    "np": true_np_onehot,
    "hv": true_hv,
    }
    return true_dict
