from abc import ABC, abstractmethod
import torch
from reetoolbox.constraints import Constraints
from torch.nn import NLLLoss
from torch.nn import CrossEntropyLoss

#This method uses untargeted loss with nllloss
def new_untargeted_loss(outputs, labels):
    loss_fn = NLLLoss()
    loss = -loss_fn(outputs, labels)
    return loss

#Currently only works for pgd (maybe not.)
def new_targeted_loss(outputs, targets):
    loss_fn = NLLLoss()
    loss = loss_fn(outputs, targets)
    return loss



#For models that have a log softmax as their output.
def untargeted_loss(outputs, labels):
    loss = outputs.gather(1, labels.unsqueeze(1))[:, 0]
    return loss

#I think that our losses might just be the same.
def targeted_loss(outputs, targets):
    num_classes = list(outputs.shape)[1]
    all_out = torch.sum(outputs, dim=1)
    target_out = outputs.gather(1, targets.unsqueeze(1))[:, 0]
    loss = (all_out / num_classes) - target_out
    # loss = -target_out
    return loss


class Classification_Optimiser(ABC):
    def __init__(self, model, Transform, hyperparameters, transform_hyperparameters, criterion=untargeted_loss,
                 device="cuda:0", scale_factor=1):
        self.model = model
        self.Transform = Transform
        self.hyperparameters = hyperparameters
        self.transform_hyperparameters = transform_hyperparameters
        self.device = device
        self.transform = None
        self.criterion = criterion
        self.scale_factor = scale_factor

    @abstractmethod
    def get_adversarial_images(self):
        pass




class Classification_PGD(Classification_Optimiser):

    # reset_weights: true if the transform weights are reset to default after optimizing for an image, false otherwise
    def get_adversarial_images(self, inputs, targets=None, reset_weights=True):
        

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

        inputs = inputs.to(self.device)  
        targets = targets.to(self.device)

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

            #Not needed when doing predict, but maybe in other methods
            #Sets the criterion targets as results of the model. 
            if targets is None:
                outputs = self.model(inputs)
                targets = torch.argmax(outputs, dim=1)

            #Transform inputs and get model results of transformed inputs
            adv_inputs = self.transform.forward(inputs*self.scale_factor)/self.scale_factor
            adv_outputs = self.model(adv_inputs)

            loss = self.criterion(adv_outputs, targets)
            

            #Why do we need to retain our graph here? I don't think we do so i have removed it.
            #MAYBE WE RETAIN GRAPH SO THAT WE CAN DO ADV FOR FREE?
            loss.backward(torch.ones_like(loss), retain_graph=True)
            #loss.backward(torch.ones_like(loss))
            opt.step()

            #Project data back into the valid range
            if constraint is not None:
                self.transform.weights = constraint_func(constraints, self.transform.weights,
                                                         self.transform.base_weights, C)

        adv_inputs = (self.transform.forward(inputs*self.scale_factor)/self.scale_factor).detach()

        #Restore requires_grad for each paramter of the model now that transform is optimized.
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = grads[i]

        #Restore models initial mode
        if in_train_mode:
            self.model.train()

        return adv_inputs





class Classification_StochasticSearch(Classification_Optimiser):

    #run stochastic search on a batch
    def get_adversarial_images(self, inputs, targets=None, reset_weights=True):
        #Get the optimiser hyperparameters
        samples = self.hyperparameters["samples"]
        weight_ranges = self.hyperparameters["weight_ranges"]
        input_range = self.hyperparameters["input_range"]

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

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
            for i in range(samples):
                #Initialize the transform weights
                for j, weight_name in enumerate(weight_ranges):
                    self.transform.weights[weight_name] = torch.FloatTensor(
                        *self.transform.weights[weight_name].shape).uniform_(*weight_ranges[weight_name]).to(self.device)
                    
                #Set targets as current predictions
                if targets is None:
                    outputs = self.model(inputs)
                    targets = torch.argmax(outputs, dim=1)

                #Apply transform
                adv_inputs = self.transform.forward(inputs*self.scale_factor)/self.scale_factor
                adv_outputs = self.model(adv_inputs)

                loss = self.criterion(adv_outputs, targets)

                
                if self.best_loss is None or self.best_adv_inputs is None:
                    self.best_loss = loss.clone()
                    self.best_adv_inputs = adv_inputs.clone().detach()
                else:
                    #Go over all losses in current batch
                    for j, input_loss in enumerate(loss):
                        #change transformed image and loss if they are the current best.
                        if input_loss < self.best_loss[j]:
                            self.best_adv_inputs[j] = adv_inputs[j].detach()
                            self.best_loss[j] = input_loss


        #Turn gradients back on
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = grads[i]

        #Return model to original mode.
        if in_train_mode:
            self.model.train()

        return self.best_adv_inputs

