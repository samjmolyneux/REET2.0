from abc import ABC, abstractmethod
import torch
from reetoolbox.constraints import Constraints
from torch.nn import NLLLoss
from torch.nn import CrossEntropyLoss
from reetoolbox.utils import cropping_center
import numpy as np
import copy
import torch.nn.functional as F
import time


def multiple_untargeted_segmentation_CE_loss(outputs, labels):
    loss_func = CrossEntropyLoss(reduction='none')    
    loss = -loss_func(outputs, labels)
    loss = loss.view(outputs.shape[0], -1).mean(1)
    return loss

def untargeted_segmentation_CE_loss(outputs, labels):
    loss_func = CrossEntropyLoss()    
    loss = -loss_func(outputs, labels)
    return loss



class Semantic_Segmentation_Optimiser(ABC):

    def __init__(self, model, Transform, hyperparameters, transform_hyperparameters, num_classes, criterion=untargeted_segmentation_CE_loss,
                 device="cuda:0", model_input_shape=(256,256), background_index=0):
        self.model = model
        self.Transform = Transform
        self.hyperparameters = hyperparameters
        self.transform_hyperparameters = transform_hyperparameters
        self.device = device
        self.transform = None
        self.criterion = criterion
        self.num_classes = num_classes
        #MAKE THE FOLLOWING CHANGEABLE
        self.post_transform_shape = model_input_shape


    @abstractmethod
    def get_adversarial_images(self):
        pass
    


class Semantic_Segmentation_PGD(Semantic_Segmentation_Optimiser):

    def get_adversarial_images(self, inputs, targets, reset_weights=True, return_masks=False, return_transform_weights = False):
        self.best_loss = None

        inputs = np.expand_dims(inputs, axis=0)
        inputs = cropping_center(inputs, self.post_transform_shape, batch=True)
        inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        
        targets = torch.tensor(np.expand_dims(targets, axis=0), dtype=torch.int64)
        targets = cropping_center(targets, self.post_transform_shape, batch=True)
        one_hot_targets = F.one_hot(targets, self.num_classes).type(torch.float32).to(self.device).permute(0,3,1,2)

        

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
            opt.zero_grad()


            if targets is None:
                ##MAKE IT SELECT CURRENT PREDICTION MASKS AS LABELS.
                pass
            
            #Transform inputs and get model results of transformed inputs
            adv_inputs = self.transform.forward(inputs)
            adv_outputs = self.model(adv_inputs)


            #MAYBE A STUPID QUESTION, DOES CLONING TARGETS AND INPUTS BEFORE CALCULATING LOSS NOT AFFECT THE LOSS FUNCTION?
            loss = self.criterion(adv_outputs, one_hot_targets)
            

            #Why do we need to retain our graph here? I don't think we do so i have removed it.
            # loss.backward(torch.ones_like(loss), retain_graph=True)
            # loss.backward(torch.ones_like(loss))
            loss.backward()
            opt.step()

            #Project data back into the valid range
            if constraint is not None:
                self.transform.weights = constraint_func(constraints, self.transform.weights,
                                                         self.transform.base_weights, C)
            
            if self.best_loss is None or self.best_adv_inputs is None:
                self.best_loss = loss.clone().detach()
                self.best_adv_inputs = adv_inputs.clone().detach().squeeze(0)
                self.best_transform_weights = self.transform.weights
            else:
                if loss < self.best_loss:
                    self.best_loss = loss.clone().detach()
                    self.best_adv_inputs = adv_inputs.clone().detach().squeeze(0)
                    #THESE ARE NOT THE CORRECT TRANSFORM WEIGHTS
                    self.best_transform_weights = self.transform.weights.clone().detach()

        
        #Restore requires_grad for each paramter of the model now that transform is optimized.
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = grads[i]

        #Restore models initial mode
        if in_train_mode:
            self.model.train()
        
        #SHOULD CHANGE THESE SO THAT THEY DO RANDOM REPEATS.
        return_dict = {"adv_inputs" : self.best_adv_inputs.clone()}
        if return_masks:
            #PGD doesnt change the mask, so just return the original
            return_dict["type_masks"] = one_hot_targets.clone().squeeze(0)

        if return_transform_weights:
            return_dict["transform_weights"] = self.best_transform_weights.clone()
        
        return return_dict


    def get_batch(self, inputs, targets, reset_weights=True, return_masks=False, return_transform_weights = False):
        self.best_loss = None

        inputs = cropping_center(inputs, self.post_transform_shape, batch=True)
  
        targets = cropping_center(targets, self.post_transform_shape, batch=True)

        

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
            opt.zero_grad()


            if targets is None:
                ##MAKE IT SELECT CURRENT PREDICTION MASKS AS LABELS.
                pass
            
            #Transform inputs and get model results of transformed inputs
            adv_inputs = self.transform.forward(inputs)
            adv_outputs = self.model(adv_inputs)


            #MAYBE A STUPID QUESTION, DOES CLONING TARGETS AND INPUTS BEFORE CALCULATING LOSS NOT AFFECT THE LOSS FUNCTION?
            loss = self.criterion(adv_outputs, targets)
            

            #Why do we need to retain our graph here? I don't think we do so i have removed it.
            # loss.backward(torch.ones_like(loss), retain_graph=True)
            # loss.backward(torch.ones_like(loss))
            loss.backward(torch.ones_like(loss))
            opt.step()


            #Project data back into the valid range
            if constraint is not None:
                self.transform.weights = constraint_func(constraints, self.transform.weights,
                                                         self.transform.base_weights, C)
            

            if self.best_loss is None or self.best_adv_inputs is None:
                self.best_loss = loss.clone().detach()
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
        
        #SHOULD CHANGE THESE SO THAT THEY DO RANDOM REPEATS.

        return self.best_adv_inputs, targets















class Semantic_Segmentation_StochasticSearch(Semantic_Segmentation_Optimiser):


    #Background index specifies which index of the mask is used to specify background.
    def get_adversarial_images(self, inputs, targets, reset_weights=True, return_masks = False, return_transform_weights = False):

        self.best_loss = None

        targets = np.expand_dims(targets, axis=0)
        
        inputs = np.expand_dims(inputs, axis=0)
        inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)

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

        
        with torch.no_grad():
            #iterate over the number of steps given (samples)
            # print(f"samples {samples}")
            # t1 = time.time()
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
                adv_outputs = self.model(adv_inputs)  

                adv_masks = self.transform.maps_forward(targets)
                adv_masks = torch.tensor(cropping_center(adv_masks, self.post_transform_shape, batch=True), dtype=torch.int64)
                one_hot_targets = F.one_hot(adv_masks, self.num_classes).type(torch.float32).to(self.device).permute(0,3,1,2)


                loss = self.criterion(adv_outputs, one_hot_targets)

                #WE ARE CLONING BECAUSE THOSE VALUES ARE ABOUT TO CHANGE, IS THERE ANYWAY THAT WE COULD SAVE THE TRANSFORM INSTEAD OF THE IMGAES
                # if self.best_loss is None or self.best_adv_inputs is None:
                #     self.best_loss = loss.clone()
                #     self.best_adv_inputs = adv_inputs.clone().detach()
                #     self.best_adv_masks = adv_masks.clone().detach()
                # else:
                    #Go over all losses in current batch
                    # for j, input_loss in enumerate(loss):
                        #change transformed image and loss if they are the current best.
                        # if input_loss < self.best_loss[j]:
                        #     self.best_adv_inputs[j] = adv_inputs[j].detach()
                        #     self.best_adv_masks[j] = adv_masks[j].detach()
                        #     self.best_loss[j] = input_loss

                if self.best_loss is None or self.best_adv_inputs is None:
                    self.best_loss = loss.clone().detach()
                    self.best_adv_inputs = adv_inputs.clone().detach().squeeze(0)
                    self.best_transform_weights = copy.deepcopy(self.transform.weights)
                    self.best_masks = one_hot_targets.clone().detach().squeeze(0)
                else:
                    if loss < self.best_loss:
                        self.best_loss = loss.clone().detach()
                        self.best_adv_inputs = adv_inputs.clone().detach().squeeze(0)
                        self.best_transform_weights = copy.deepcopy(self.transform.weights)
                        self.best_masks = one_hot_targets.clone().detach().squeeze(0)
        t2 = time.time()
        # print(f"full attack time: {t2-t1}")
        #Turn gradients back on
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = grads[i]

        #Return model to original mode.
        if in_train_mode:
            self.model.train()

        return_dict = {"adv_inputs" : self.best_adv_inputs.clone()}
        if return_masks:
            #PGD doesnt change the mask, so just return the original
            return_dict["type_masks"] = self.best_masks.clone()

        if return_transform_weights:
            return_dict["transform_weights"] = copy.deepcopy(self.best_transform_weights)
        
        return return_dict











    def get_batch(self, inputs, targets, reset_weights=True, return_masks = False, return_transform_weights = False):

        self.best_loss = None


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

        
        with torch.no_grad():
            #iterate over the number of steps given (samples)
            # print(f"samples {samples}")
            # t1 = time.time()
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
                adv_outputs = self.model(adv_inputs)  

                # t1 = time.time()
                # adv_masks = targets
                adv_masks = self.transform.mask_forward(targets)
                # t2 = time.time()
                # print(f"time: {t2-t1}")
                adv_masks = cropping_center(adv_masks, self.post_transform_shape, batch=True)


                loss = self.criterion(adv_outputs, adv_masks)

                #WE ARE CLONING BECAUSE THOSE VALUES ARE ABOUT TO CHANGE, IS THERE ANYWAY THAT WE COULD SAVE THE TRANSFORM INSTEAD OF THE IMGAES
                if self.best_loss is None or self.best_adv_inputs is None:
                    self.best_loss = loss.clone()
                    self.best_adv_inputs = adv_inputs.clone().detach()
                    self.best_adv_masks = adv_masks.clone().detach()
                else:
                    # Go over all losses in current batch
                    for j, input_loss in enumerate(loss):
                        #change transformed image and loss if they are the current best.
                        if input_loss < self.best_loss[j]:
                            self.best_adv_inputs[j] = adv_inputs[j].detach()
                            self.best_adv_masks[j] = adv_masks[j].detach()
                            self.best_loss[j] = input_loss


        # print(f"full attack time: {t2-t1}")
        #Turn gradients back on
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = grads[i]

        #Return model to original mode.
        if in_train_mode:
            self.model.train()

        return self.best_adv_inputs, self.best_adv_masks






