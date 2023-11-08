from IPython.core.interactiveshell import default
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
import copy
from sklearn.metrics import accuracy_score, jaccard_score
import os
from torch.utils.data import DataLoader
import math
from reetoolbox.maps import transform_dictionary, default_classification_optimiser, default_semantic_segmentation_optimiser, default_hover_optimiser, default_optimiser_params, default_transform_params, classification_loss, semantic_segmentation_loss, hover_net_loss
from reetoolbox.file_loader import FileLoader

class Evaluator(ABC):
    def __init__(self, model, dataset, transform, criterion="Default", dataloader = None, optimiser_params=None, trans_params=None, device="cuda:0"):  
        self.model = model 
        self.dataset = dataset
        self.device = device
        self.model.eval()
        self.model.to(device)
        self.set_transform_and_optimiser(transform)

        if dataloader == None :
            self.dataloader = self.get_default_dataloader()
        else:
            self.dataloader = dataloader
        
        if trans_params == None :
            self.trans_params = default_transform_params[transform]
        else:
            self.trans_params = trans_params
        
        self.criterion = self.set_criterion(criterion)
        self.results_dictionary = {}


        if dataset.__getitem__(0)[0].max() > 1:
            self.scale_factor = 1
        else:
            self.scale_factor = 255
        
        test_img = dataset.__getitem__(0)[1]
        if not dataset.__getitem__(0)[1].shape:
            self.num_classes = 1
        else:
            self.num_classes = dataset.__getitem__(0)[1].shape[0]


        if optimiser_params == None:
            self.attack = self.TransformOptimiser(self.model, self.Transform, default_optimiser_params[transform], self.trans_params, criterion=self.criterion, device=device, scale_factor=self.scale_factor)
        else:
            self.attack = self.TransformOptimiser(self.model, self.Transform, optimiser_params, self.trans_params, criterion=self.criterion, device=device, scale_factor=self.scale_factor)


        #ADD A WAY OF GETTING THE CORRECT get_adversarial_images HERE.
    
    #SHOULD IMPROVE THIS SO THAT IT WORKS WITHOUT __INIT__
    @abstractmethod
    def set_transform_and_optimiser(self, transform):
        self.Transform = transform_dictionary[transform]
        """self.TransformOptimiser = taks_default_optimiser[transform]"""
        pass
    
    #SHOULD IMPROVE THIS SO THAT IT WORKS WITHOUT __INIT__
    @abstractmethod
    def set_criterion(self, criterion):
        """self.criterion = task_loss[criterion]"""
        pass
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def display_results_by_index(self, indices, targets, criterion):
        #Should be easy for both, just get a images and adv for classification, show labels too
        #For semantic display predicted masks and masks and adv masks
        pass

    def get_default_dataloader(self):
        return DataLoader(self.dataset, shuffle=True,
                                  batch_size=1, pin_memory=True,
                                  num_workers=os.cpu_count())
        
    

    def set_attack_hyperparameters(self, hyperparameters):
        self.attack.hyperparameters = hyperparameters
      
    


#Dataloader: should contain an input image and the label that it has or should be targetted too, based on criterion.
class Classification_Evaluator(Evaluator):
    

    def set_transform_and_optimiser(self, transform):
        self.Transform = transform_dictionary[transform]
        self.TransformOptimiser = default_classification_optimiser[transform]
        return 

    # Set the criterion 
    def set_criterion(self, criterion):
        return classification_loss[criterion]
      

    #Will have to do some changing around for this function so that it will work for as much data as segmentation.
    def predict(self, adversarial, perturbation_measure=None, weight_measure=None):
        outputs = []
        adv_outputs = []
        all_labels = []
        pert_measures = []
        weight_measures = []

        count = 0

        for inputs, labels in self.dataloader:
            count += len(inputs)
            print(count)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            if adversarial:
                #Create the adversarial images
                adv_inputs = self.attack.get_adversarial_images(inputs, targets=labels, reset_weights=True)


                #Get the results for the adversarial images on model
                batch_output = self.model(adv_inputs).cpu().detach()
                adv_outputs.extend(batch_output)

                #Measure how large perturbation is
                if perturbation_measure is not None:
                    input_pert = (adv_inputs - inputs).cpu().detach()
                    pert_measures.append(perturbation_measure(input_pert))

                #Still not found a use for this weight_measure
                if weight_measure is not None:
                    weight_pert = (self.attack.transform.weights - self.attack.transform.base_weights).cpu().detach()
                    weight_measures.append(weight_measure(weight_pert))

            #Get model outputs for actual images
            batch_output = self.model(inputs).cpu().detach()
            outputs.extend(batch_output)
            all_labels.extend(labels.cpu().detach())

        #Add outputs, their labels, the advesarial image's results and other to the results dict. 
        self.results_dictionary = {}
        self.results_dictionary["outputs"] = torch.stack(outputs)
        
        self.results_dictionary["labels"] = torch.stack(all_labels)
        if adversarial:
            self.results_dictionary["adversarial_outputs"] = torch.stack(adv_outputs)
        if pert_measures:
            self.results_dictionary["perturbation_measures"] = torch.stack(pert_measures)
        if weight_measures:
            self.results_dictionary["weight_measures"] = torch.stack(weight_measures)
        return self.results_dictionary

    def display_results_by_index(self, indices, targets=None, criterion=None, scale_perturbation=False):
        inputs, adv_inputs = self.attack_inputs_by_index(indices, targets, criterion)

        if inputs.max() > 1 :
            inputs = inputs.detach().permute(0,2,3,1).cpu().numpy().astype(np.uint8)
            adv_inputs = adv_inputs.detach().permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        else:
            inputs = inputs.detach().permute(0,2,3,1).cpu().numpy()
            adv_inputs = adv_inputs.detach().permute(0,2,3,1).cpu().numpy()

        
        for i in range(inputs.shape[0]):

              image_index = indices[i]
              fig, ax = plt.subplots(1,3, figsize=(10,10))

              perturbation = (inputs[i].astype(np.float32)-adv_inputs[i].astype(np.float32))

              rmse = np.sqrt(np.mean(np.power(perturbation, 2)))
              l_inf = np.max(np.abs(perturbation))
              abs_pert = np.abs(perturbation).astype(np.uint8)
              if scale_perturbation:
                  abs_pert = math.floor((255/np.max(abs_pert)))*abs_pert

              ax[0].imshow(inputs[i])
              ax[1].imshow(abs_pert)
              ax[2].imshow(adv_inputs[i])
              
              title_0 = f"Image idx {image_index}"
              title_1 = f"Adv Image idx {image_index}"
              ax[0].set_title(title_0)
              ax[1].set_title("|Perturbation|")
              ax[2].set_title(title_1)

              ax[1].spines['right'].set_visible(False)
              ax[1].spines['top'].set_visible(False)
              ax[1].spines['bottom'].set_visible(False)
              ax[1].spines['left'].set_visible(False)
              ax[1].get_xaxis().set_ticks([])
              ax[1].get_yaxis().set_ticks([])


              ax[0].axis("off")
              # ax[1].axis('off')
              ax[2].axis("off")

              x_lab = f"L2: {rmse:.3f}, L infinity: {l_inf:.3f}"
              ax[1].set_xlabel(x_lab)
              
              # ax[1].text(-0.5, 0.5, x_lab, fontdict={'size':12},
              #     verticalalignment='center', transform=ax[1].transAxes)
              # fig.text(0.5, 0.04, x_lab, ha="center")

              plt.savefig("catch.png")

              prediction = torch.argmax(self.results_dictionary["outputs"][image_index])
              prediction_adv = torch.argmax(self.results_dictionary["adversarial_outputs"][image_index])
              truth_label = self.results_dictionary["labels"][image_index]
              print(f"Classification before: {prediction}. Classifcation after: {prediction_adv}. Truth Label: {truth_label}")
              print(f"")
              print("")



    #FIX OPTIMISE FOR THIS ONE, I THINK WE CHANGE THE OPTIMISE FUNCTION TO SOMETHING LIKE GET ADVERSARIAL IMAGES, THEN WE HAVE ANOTHER 
    #FUNCTION THAT WE USE TO GET ADVERSARIAL IMAGES AND MASKS
    #Returns them on device and uses input indices.
    def attack_inputs_by_index(self, input_indices, target_classes=None, criterion=None):
        inputs = []
        labels = []

        for i in input_indices:
            inputs.append(self.dataset[i][0])
            labels.append(self.dataset[i][1])
            

        #Dont need to send to device, they will be sent in optimise
        inputs = torch.stack(inputs).to(self.device)
        labels = torch.stack(labels).to(self.device)

        #The logic here is a little bad. Maybe?
        if target_classes is not None and criterion is not None:
            self.attack.criterion = criterion
            adv_inputs = self.attack.get_adversarial_images(inputs, targets=target_classes, reset_weights=True)

            self.attack.criterion = self.criterion
        else:
            adv_inputs = self.attack.get_adversarial_images(inputs, targets=labels, reset_weights=True)
            

        return inputs, adv_inputs


    #Still needs fixing and testing
    def metric_vs_strength(self, param, param_range, step_size, metric, **metric_params):
        start = param_range[0]
        end = param_range[1] + step_size
        param_values = [f for f in np.arange(start, end, step_size)]

        all_scores = []

        printerval = np.round(len(param_values) / 4)

        original_parameters = copy.deepcopy(self.attack.hyperparameters)

        for i, value in enumerate(param_values):
            self.attack.hyperparameters[param] = value

            if i == 0:
                print("Starting hyperparameters:", self.attack.hyperparameters)
            if i % printerval == 0:
                print(f"{round(i * 100 / len(param_values))}% complete...")

            score = self.compute_metric(metric, **metric_params)
            all_scores.append(score)

        print("Done.")

        self.attack.hyperparameters = original_parameters

        return param_values, all_scores



    #Still needs fixing and testing
    def compute_metric(self, metric, **parameters):
        results = self.predict(**parameters)
        score = metric(results)
        return score
        

    def get_accuracy(self):
        labels = self.results_dictionary["labels"]
        outputs = self.results_dictionary["outputs"]
        _, predictions = torch.max(outputs, 1)
        return accuracy_score(labels, predictions)

    def get_adversarial_accuracy(self):
        labels = self.results_dictionary["labels"]
        outputs = self.results_dictionary["adversarial_outputs"]
        _, predictions = torch.max(outputs, 1)
        return accuracy_score(labels, predictions)


    def get_input_sensitivity(self):
        outputs = torch.exp(self.results_dictionary["outputs"])
        adv_outputs = torch.exp(self.results_dictionary["adversarial_outputs"])
        mean_out_diff = torch.mean(torch.abs(outputs - adv_outputs))
        return mean_out_diff.item()

    def get_normalised_input_sensitivity(self):
        in_sens = self.get_input_sensitivity()

        pert_measures = self.results_dictionary["perturbation_measures"]
        avg_pert = torch.mean(pert_measures)

        norm_in_sens = in_sens / avg_pert
        return norm_in_sens.item()

    #Needs to change, currently not accurate. 
    def get_fooling_ratio(self):

        labels = np.array(self.results_dictionary["labels"])
        outputs = np.array(self.results_dictionary["outputs"])
        adv_outputs = np.array(self.results_dictionary["adversarial_outputs"])

        #Want to only include labels that were correctly classified on original images
        equal_indices = np.where(labels == outputs)
        cc_labels = labels[equal_indices]
        cc_outputs = outputs[equal_indices]
        cc_adv_outputs = adv_outputs[equal_indices]
        
        fooled_images = np.where(cc_outputs != cc_adv_outputs)
        
        return len(fooled_images)/len(cc_outputs)
    
    def output_metrics(self):
        acc = self.get_accuracy()
        robust_acc = self.get_adversarial_accuracy()
        fool_ratio = self.get_fooling_ratio()
        print(f"Accuracy: {acc:.3f}, robust accuracy: {robust_acc:.3f}, fooling ratio: {fool_ratio:.3f}")




class Semantic_Segmentation_Evaluator(Evaluator):

    def __init__(self, 
                model, 
                data_dir_list, 
                transform, 
                criterion="Default", 
                optimiser_params=None, 
                trans_params=None, 
                device="cuda:0",
                num_classes = 5,
                background_index=0,
                model_input_shape = (256,256),
                evaluation_shape = (80,80)):

        self.model = model 
        self.dataset = FileLoader(data_dir_list)
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.set_transform_and_optimiser(transform)

        #background index is automatically 0 for hovernet
        self.background_index = background_index

        self.num_classes = num_classes

        #Determines input and output shape based on model type.
        self.model_input_shape = model_input_shape
        self.evaluation_shape = evaluation_shape


        if trans_params == None :
            self.trans_params = default_transform_params[transform]
        else:
            self.trans_params = trans_params
        
        self.criterion = self.set_criterion(criterion)
        self.results_dictionary = {}

        if optimiser_params == None:
            self.attack = self.TransformOptimiser(self.model, self.Transform, default_optimiser_params[transform], self.trans_params, criterion=self.criterion, device=device, model_input_shape=self.model_input_shape, num_classes=self.num_classes)
        else:
            self.attack = self.TransformOptimiser(self.model, self.Transform, optimiser_params, self.trans_params, criterion=self.criterion, device=device, model_input_shape=self.model_input_shape, num_classes=self.num_classes)


    #Set transform and optimiser
    def set_transform_and_optimiser(self, transform):
        self.Transform = transform_dictionary[transform]
        self.TransformOptimiser = default_semantic_segmentation_optimiser[transform]
        return 


      # Set the criterion 
    def set_criterion(self, criterion):
        return semantic_segmentation_loss[criterion]

    ##DO THIS PREDICT PROPERLY
    def predict(self, adversarial, perturbation_measure=None, weight_measure=None, display=False, scale_perturbation=False):

        return_weights_boolean = weight_measure is not None

        pert_measures = []
        weight_measures = []

        self.orig_results_dict = {}
        self.adv_results_dict = {}

        #Semantic segmentation metrics (pixelwise metrics)
        self.orig_results_dict["nucleus_pixel_tp"] = 0
        self.orig_results_dict["nucleus_pixel_fp"] = 0
        self.orig_results_dict["nucleus_pixel_fn"] = 0
        self.orig_results_dict["type_pixel_tp"] = {}
        self.orig_results_dict["type_pixel_fp"] = {}
        self.orig_results_dict["type_pixel_fn"] = {}
        for i in range(self.num_classes):
            self.orig_results_dict["type_pixel_tp"][i] = 0
            self.orig_results_dict["type_pixel_fp"][i] = 0
            self.orig_results_dict["type_pixel_fn"][i] = 0
        self.orig_results_dict["all_type_pixel_tp_tn"] = 0
        self.orig_results_dict["all_type_pixel_tp_tn_fp_fn"] = 0

        
        self.adv_results_dict["nucleus_pixel_tp"] = 0
        self.adv_results_dict["nucleus_pixel_fp"] = 0
        self.adv_results_dict["nucleus_pixel_fn"] = 0
        self.adv_results_dict["type_pixel_tp"] = {}
        self.adv_results_dict["type_pixel_fp"] = {}
        self.adv_results_dict["type_pixel_fn"] = {}
        for i in range(self.num_classes):
            self.adv_results_dict["type_pixel_tp"][i] = 0
            self.adv_results_dict["type_pixel_fp"][i] = 0
            self.adv_results_dict["type_pixel_fn"][i] = 0
        self.adv_results_dict["all_type_pixel_tp_tn"] = 0
        self.adv_results_dict["all_type_pixel_tp_tn_fp_fn"] = 0




        count = 0
        for imgs, masks in self.dataset:
            count+=1


            imgs = imgs.transpose(2,0,1)

            

            if adversarial:
                #Create the adversarial images
                attack_results_dict = self.attack.get_adversarial_images(imgs, targets=masks, reset_weights=True, return_masks=True, return_transform_weights=True)

                imgs = np.expand_dims(imgs, axis=0)
                inputs = torch.tensor(imgs, dtype=(torch.float32)).to(self.device)
                inputs = cropping_center(inputs, self.model_input_shape, batch=True)
                orig_pred_mask = self.model(inputs).detach().squeeze(0).permute(1,2,0).cpu().numpy()
                orig_pred_mask = np.argmax(orig_pred_mask, axis=-1)
                
                orig_truth_mask = cropping_center(masks, self.model_input_shape, batch=True)
                
                adv_pred_mask = self.model(attack_results_dict["adv_inputs"].unsqueeze(0)).detach().squeeze(0).permute(1,2,0).cpu().numpy()
                adv_pred_mask = np.argmax(adv_pred_mask, axis=-1)

                adv_truth_mask = attack_results_dict["type_masks"].permute(1,2,0).cpu().numpy()
                adv_truth_mask = np.argmax(adv_truth_mask, axis=-1)

                orig_img = cropping_center(imgs, self.model_input_shape, batch=True).squeeze(0).transpose((1,2,0))
                
                adv_img = attack_results_dict["adv_inputs"].permute(1,2,0).cpu().numpy().astype("uint8")


                #ADD AN IF STATEMENT FOR THIS HERE
                # orig_img = cropping_center(orig_img, (80,80))
                # adv_img = cropping_center(adv_img, (80,80))
                # orig_pred_mask = cropping_center(orig_pred_mask, (80,80))
                # adv_pred_mask = cropping_center(adv_pred_mask, (80,80))
                # orig_truth_mask = cropping_center(orig_truth_mask, (80,80))
                # adv_truth_mask = cropping_center(adv_truth_mask, (80,80))
                if display:
                    fig, ax = plt.subplots(3, 3, figsize=(16, 16))
                    ax[0,0].imshow(orig_img)
                    ax[0,1].imshow(orig_truth_mask, cmap = "rainbow", vmin=0, vmax=self.num_classes-1, interpolation="nearest")
                    im=ax[0,2].imshow(orig_pred_mask, cmap = "rainbow", vmin=0, vmax=self.num_classes-1, interpolation="nearest")
                    
                    perturbation = (orig_img.astype(np.float32)-adv_img.astype(np.float32))

                    rmse = np.sqrt(np.mean(np.power(perturbation, 2)))
                    l_inf = np.max(np.abs(perturbation))
                    abs_pert = np.abs(perturbation).astype(np.uint8)



                    if scale_perturbation:
                      abs_pert = math.floor((255/np.max(abs_pert)))*abs_pert

                    ax[2,0].imshow(abs_pert, vmin=0, vmax=255)


                    ax[1,0].imshow(adv_img)
                    ax[1,1].imshow(adv_truth_mask, cmap = "rainbow", vmin=0, vmax=self.num_classes-1, interpolation="nearest")
                    ax[1,2].imshow(adv_pred_mask, cmap = "rainbow", vmin=0, vmax=self.num_classes-1, interpolation="nearest")

                    title_0 = f"Images"
                    title_1 = f"Truth Masks"
                    title_2 = f"Prediction Masks"

                    y0_label = f"Original"
                    y1_label = f"Adversarial"
                    y2_label = f"|Perturbation|"

                    ax[0,0].set_title(title_0)
                    ax[0,1].set_title(title_1)
                    ax[0,2].set_title(title_2)


                    ax[0,0].set_ylabel(y0_label)
                    ax[1,0].set_ylabel(y1_label)
                    ax[2,0].set_ylabel(y2_label)

                    ax[0, 0].text(-0.5, 0.5, y0_label, fontdict={'size':12},
                      verticalalignment='center', transform=ax[0, 0].transAxes)
                    ax[1, 0].text(-0.5, 0.5, y1_label, fontdict={'size':12},
                      verticalalignment='center', transform=ax[1, 0].transAxes)
                    # ax[2, 0].text(-0.5, 0.5, y2_label, fontdict={'size':12},
                    #   verticalalignment='center', transform=ax[2, 0].transAxes)

                    

                    ax[2,0].spines['right'].set_visible(False)
                    ax[2,0].spines['top'].set_visible(False)
                    ax[2,0].spines['bottom'].set_visible(False)
                    ax[2,0].spines['left'].set_visible(False)
                    ax[2,0].get_xaxis().set_ticks([])
                    ax[2,0].get_yaxis().set_ticks([])




                    x_lab = f"L2: {rmse:.3f}, L infinity: {l_inf:.3f}"
                    ax[2,0].set_xlabel(x_lab)

                    ax[0,0].axis("off")
                    ax[0,1].axis("off")
                    ax[0,2].axis("off")
                    ax[1,0].axis("off")
                    ax[1,1].axis("off")
                    ax[1,2].axis("off")

                    ax[2,1].axis("off")
                    ax[2,2].axis("off")
                    
                    cax = fig.add_axes([1, 0.4, 0.05, 0.48])
                    cb = plt.colorbar(im, cax=cax, values=list(range(self.num_classes)))
                    plt.show()

                    

                if (np.array_equal(np.unique(orig_truth_mask), [0])) or (np.array_equal(np.unique(adv_truth_mask), [0])):
                    print("Didnt_make it")
                    continue
                  
                # #ADD AN IF STATEMENT FOR THIS HERE
                # orig_pred_mask = cropping_center(orig_pred_mask, (80,80))
                # adv_pred_mask = cropping_center(adv_pred_mask, (80,80))
                # orig_truth_mask = cropping_center(orig_truth_mask, (80,80))
                # adv_truth_mask = cropping_center(adv_truth_mask, (80,80))


                calculate_semantic_seg_metrics(orig_pred_mask, orig_truth_mask, self.orig_results_dict, self.num_classes, self.background_index)
                #Calculate the semantic segmentation metrics for adversarial image
                calculate_semantic_seg_metrics(adv_pred_mask, adv_truth_mask, self.adv_results_dict, self.num_classes, self.background_index)


                    # fig, axs = plt.subplots(1, 2, figsize=(8, 8))

                    # # Plot an image in the first subplot
                    # im1 = axs[0].imshow(truth_masks[i].detach().cpu().numpy(), cmap='cool', vmin=0, vmax=5)

                    # # Plot an image in the second subplot
                    # im2 = axs[1].imshow(pred_outputs[i].detach().cpu().numpy(), cmap='cool', vmin=0, vmax=5)

                    # fig.colorbar(im1, orientation='vertical')


                    # # Show the figure
                    # plt.show()
                

                #Measure how large perturbation is
                # if perturbation_measure is not None:
                #     # input_pert = (adv_inputs - inputs).cpu().detach()
                #     # pert_measures.append(perturbation_measure(input_pert))

                # #Does not work for Stochastic Search (stochastic search doesnt select the best transform)
                # if weight_measure is not None:
                #     weight_pert = (self.attack.transform.weights - self.attack.transform.base_weights).cpu().detach()
                #     weight_measures.append(weight_measure(weight_pert))


        #THE SEMANTIC SEGMENTATION METRICS
        #pixel_wise IoU
        self.orig_results_dict["pixel_wise_nucleus_IoU"] = self.orig_results_dict["nucleus_pixel_tp"]/(self.orig_results_dict["nucleus_pixel_tp"]+self.orig_results_dict["nucleus_pixel_fp"]+self.orig_results_dict["nucleus_pixel_fn"]+1.0e-6)
        self.adv_results_dict["pixel_wise_nucleus_IoU"] = self.adv_results_dict["nucleus_pixel_tp"]/(self.adv_results_dict["nucleus_pixel_tp"]+self.adv_results_dict["nucleus_pixel_fp"]+self.adv_results_dict["nucleus_pixel_fn"]+1.0e-6)

        #pixel_wise IoU for each type
        self.orig_results_dict["pixel_wise_type_IoU"] = {}
        self.adv_results_dict["pixel_wise_type_IoU"] = {}
        for i in range(self.num_classes):
            self.orig_results_dict["pixel_wise_type_IoU"][i] = self.orig_results_dict["type_pixel_tp"][i]/(self.orig_results_dict["type_pixel_tp"][i]+self.orig_results_dict["type_pixel_fp"][i]+self.orig_results_dict["type_pixel_fn"][i]+1.0e-6)
            self.adv_results_dict["pixel_wise_type_IoU"][i] = self.adv_results_dict["type_pixel_tp"][i]/(self.adv_results_dict["type_pixel_tp"][i]+self.adv_results_dict["type_pixel_fp"][i]+self.adv_results_dict["type_pixel_fn"][i]+1.0e-6)

        
        #pixel_wise Dice coefficient
        self.orig_results_dict["pixel_wise_nucleus_Dice"] = 2*self.orig_results_dict["nucleus_pixel_tp"]/(2*self.orig_results_dict["nucleus_pixel_tp"]+self.orig_results_dict["nucleus_pixel_fp"]+self.orig_results_dict["nucleus_pixel_fn"]+1.0e-6)
        self.adv_results_dict["pixel_wise_nucleus_Dice"] = 2*self.adv_results_dict["nucleus_pixel_tp"]/(2*self.adv_results_dict["nucleus_pixel_tp"]+self.adv_results_dict["nucleus_pixel_fp"]+self.adv_results_dict["nucleus_pixel_fn"]+1.0e-6)

        #pixel_wise Dice for each type
        self.orig_results_dict["pixel_wise_type_Dice"] = {}
        self.adv_results_dict["pixel_wise_type_Dice"] = {}
        for i in range(self.num_classes):
            self.orig_results_dict["pixel_wise_type_Dice"][i] = 2*self.orig_results_dict["type_pixel_tp"][i]/(2*self.orig_results_dict["type_pixel_tp"][i]+self.orig_results_dict["type_pixel_fp"][i]+self.orig_results_dict["type_pixel_fn"][i]+1.0e-6)
            self.adv_results_dict["pixel_wise_type_Dice"][i] = 2*self.adv_results_dict["type_pixel_tp"][i]/(2*self.adv_results_dict["type_pixel_tp"][i]+self.adv_results_dict["type_pixel_fp"][i]+self.adv_results_dict["type_pixel_fn"][i]+1.0e-6)

        #THINK THAT THERE MAY BE A PROBLEM WITH THE PIXEL ACCURACY
        #pixel_wise type accuracy 
        self.orig_results_dict["all_type_pixel_accuracy"] = self.orig_results_dict["all_type_pixel_tp_tn"]/self.orig_results_dict["all_type_pixel_tp_tn_fp_fn"]
        self.adv_results_dict["all_type_pixel_accuracy"] = self.adv_results_dict["all_type_pixel_tp_tn"]/self.adv_results_dict["all_type_pixel_tp_tn_fp_fn"]






        return self.orig_results_dict, self.adv_results_dict


    def display_results_by_index(self, indices, targets, criterion):
        return super().display_results_by_index(indices, targets, criterion)

    def get_mean_pixel_accuracy(self, include_background = True):
        if include_background:
            accuracys = self.results_dictionary["Pixel Accuracy"]  
        else:
            accuracys  = self.results_dictionary["Pixel Accuracy NB"]

        return sum(accuracys)/len(accuracys)

    def get_mean_adv_pixel_accuracy(self, include_background = True):
        if include_background:
            accuracys = self.results_dictionary["Adv Pixel Accuracy"]
        else:
            accuracys = self.results_dictionary["Adv Pixel Accuracy NB"]

        return sum(accuracys)/len(accuracys)
    

    def attack_inputs_by_index(self, input_indices, target_classes=None, criterion=None, return_masks=False):

        if return_masks == False:
            return super().attack_inputs_by_index(input_indices, target_classes=None, criterion=None)
        else:

            inputs = []
            masks = []

            for i in input_indices:
                inputs.append(self.dataset[i][0])
                masks.append(self.dataset[i][1])

            #Dont need to send to device, they will be sent in optimise
            inputs = torch.stack(inputs).to(self.device)
            masks = torch.stack(masks).to(self.device)

            #The logic here is a little bad. Maybe?
            if target_classes is not None and criterion is not None:
                self.attack.criterion = criterion
                adv_inputs, adv_masks = self.attack.get_adversarial_images(inputs, targets=target_classes, reset_weights=True, return_masks=True)
                self.attack.criterion = self.criterion
            else:
                adv_inputs, adv_masks = self.attack.get_adversarial_images(inputs, targets=masks, reset_weights=True, return_masks=True)
            return inputs, adv_inputs, masks, adv_masks
    
    #CURRENTLY DOENST WORK WITH BIG SETS OF INDICES BECAUSE WE ARE SENDING ALL TO THE GPU
    def display_results_by_index(self, indices, targets=None, criterion=None):

        inputs, adv_inputs, masks, adv_masks = self.attack_inputs_by_index(indices, targets, criterion, return_masks=True)

        pred = torch.argmax(self.model(inputs), 1).detach().cpu().numpy()
        pred_adv = torch.argmax(self.model(adv_inputs), 1).detach().cpu().numpy()

        

        if inputs.max() > 1 :
            inputs = inputs.detach().permute(0,2,3,1).cpu().numpy().astype(np.uint8)
            adv_inputs = adv_inputs.detach().permute(0,2,3,1).cpu().numpy().astype(np.uint8)

            masks = torch.argmax(masks, 1).detach().cpu().numpy().astype(np.uint8)
            adv_masks = torch.argmax(adv_masks, 1).detach().cpu().numpy().astype(np.uint8)
        else:
            inputs = inputs.detach().permute(0,2,3,1).cpu().numpy()
            adv_inputs = adv_inputs.detach().permute(0,2,3,1).cpu().numpy()

            masks = torch.argmax(masks, 1).detach().cpu().numpy()
            adv_masks = torch.argmax(adv_masks, 1).detach().cpu().numpy()

        
        for i in range(inputs.shape[0]):


              image_index = indices[i]
              fig, ax = plt.subplots(2,3, figsize=(12,8))

              ax[0,0].imshow(inputs[i])
              ax[0,1].imshow(masks[i], cmap = "rainbow", vmin=0, vmax=self.num_classes-1, interpolation="nearest")
              im=ax[0,2].imshow(pred[i], cmap = "rainbow", vmin=0, vmax=self.num_classes-1, interpolation="nearest")
              ax[1,0].imshow(adv_inputs[i])
              ax[1,1].imshow(adv_masks[i], cmap = "rainbow", vmin=0, vmax=self.num_classes-1, interpolation="nearest")
              ax[1,2].imshow(pred_adv[i], cmap = "rainbow", vmin=0, vmax=self.num_classes-1, interpolation="nearest")
              



              
              title_0 = f"Images"
              title_1 = f"Truth Masks"
              title_2 = f"Prediction Masks"

              y0_label = f"Image idx {image_index}"
              y1_label = f"Adv Image idx {image_index}"
              ax[0,0].set_title(title_0)
              ax[0,1].set_title(title_1)
              ax[0,2].set_title(title_2)

              ax[0,0].set_ylabel(y0_label)
              ax[1,0].set_ylabel(y1_label)

              ax[0, 0].text(-0.5, 0.5, y0_label, fontdict={'size':12},
                verticalalignment='center', transform=ax[0, 0].transAxes)
              ax[1, 0].text(-0.5, 0.5, y1_label, fontdict={'size':12},
                verticalalignment='center', transform=ax[1, 0].transAxes)


              ax[0,0].axis("off")
              ax[0,1].axis("off")
              ax[0,2].axis("off")
              ax[1,0].axis("off")
              ax[1,1].axis("off")
              ax[1,2].axis("off")

              cax = fig.add_axes([1, 0.10, 0.05, 0.8])
              cb = plt.colorbar(im, cax=cax, values=list(range(self.num_classes)))


              plt.show()

              accuracy = self.results_dictionary["Pixel Accuracy NB"][image_index]
              adv_accuracy = self.results_dictionary["Adv Pixel Accuracy NB"][image_index]

              print(f"Accuracy before: {accuracy:.4f}. Accuracy after: {adv_accuracy:.4f}")
              print("")
              print("")


from collections import OrderedDict
import torch.nn.functional as F
from reetoolbox.utils import gen_inference_pred_map, process, gen_hover_truth_map_from_dict, inst_to_semantic_segmentation_mask, cropping_center, gen_hover_truth_map_from_ann, gen_truth_inst_info, fix_mirror_padding
from reetoolbox.hover_optimisers import gen_tp_onehot_map, gen_hv_np_maps
from reetoolbox.hover_metrics import calculate_instance_segmentation_stats, calculate_type_classification_stats, remap_label
from reetoolbox.semantic_seg_metrics import calculate_semantic_seg_metrics


class HoVer_Evaluator(Evaluator):

    def __init__(self, 
                model, 
                data_dir_list, 
                transform, 
                criterion="Default", 
                optimiser_params=None, 
                trans_params=None, 
                device="cuda:0",
                num_classes = 5,
                background_index=0,
                model_type = "original"):

        self.model = model #Make model automatically go to device?
        self.dataset = FileLoader(data_dir_list, with_inst=True)
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.set_transform_and_optimiser(transform)

        #background index is automatically 0 for hovernet
        self.background_index = background_index

        self.num_classes = num_classes

        #Determines input and output shape based on model type.
        if model_type=="original":
            self.model_input_shape = (270,270)
            self.model_output_shape = (80,80)
        else:
            self.model_input_shape = (256,256)
            self.model_output_shape = (164,164)

        if trans_params == None :
            self.trans_params = default_transform_params[transform]
        else:
            self.trans_params = trans_params
        
        self.criterion = self.set_criterion(criterion)
        self.results_dictionary = {}

        if optimiser_params == None:
            self.attack = self.TransformOptimiser(self.model, self.Transform, default_optimiser_params[transform], self.trans_params, criterion=self.criterion, device=device, model_input_shape=self.model_input_shape, model_output_shape=self.model_output_shape)
        else:
            self.attack = self.TransformOptimiser(self.model, self.Transform, optimiser_params, self.trans_params, criterion=self.criterion, device=device, model_input_shape=self.model_input_shape, model_output_shape=self.model_output_shape)

    #Set transform and optimiser
    def set_transform_and_optimiser(self, transform):
        self.Transform = transform_dictionary[transform]
        self.TransformOptimiser = default_hover_optimiser[transform]
        return 


      # Set the criterion 
    def set_criterion(self, criterion):
        return hover_net_loss[criterion]

    ##DOES NOT WORK IF ADVERSARIAL IS SET TO FALSE, HAVE TO FIX THAT SURELY.
    def predict(self, adversarial, perturbation_measure=None, weight_measure=None, display=False, scale_perturbation=False):

        return_weights_boolean = weight_measure is not None

        pert_measures = []
        weight_measures = []

        self.orig_results_dict = {}
        self.adv_results_dict = {}

        #Semantic segmentation metrics (pixelwise metrics)
        self.orig_results_dict["nucleus_pixel_tp"] = 0
        self.orig_results_dict["nucleus_pixel_fp"] = 0
        self.orig_results_dict["nucleus_pixel_fn"] = 0
        self.orig_results_dict["type_pixel_tp"] = {}
        self.orig_results_dict["type_pixel_fp"] = {}
        self.orig_results_dict["type_pixel_fn"] = {}
        for i in range(self.num_classes):
            self.orig_results_dict["type_pixel_tp"][i] = 0
            self.orig_results_dict["type_pixel_fp"][i] = 0
            self.orig_results_dict["type_pixel_fn"][i] = 0
        self.orig_results_dict["all_type_pixel_tp_tn"] = 0
        self.orig_results_dict["all_type_pixel_tp_tn_fp_fn"] = 0

        
        self.adv_results_dict["nucleus_pixel_tp"] = 0
        self.adv_results_dict["nucleus_pixel_fp"] = 0
        self.adv_results_dict["nucleus_pixel_fn"] = 0
        self.adv_results_dict["type_pixel_tp"] = {}
        self.adv_results_dict["type_pixel_fp"] = {}
        self.adv_results_dict["type_pixel_fn"] = {}
        for i in range(self.num_classes):
            self.adv_results_dict["type_pixel_tp"][i] = 0
            self.adv_results_dict["type_pixel_fp"][i] = 0
            self.adv_results_dict["type_pixel_fn"][i] = 0
        self.adv_results_dict["all_type_pixel_tp_tn"] = 0
        self.adv_results_dict["all_type_pixel_tp_tn_fp_fn"] = 0


        #classifcation nucleus wise segmentation metrics
        self.orig_results_dict["nucleus_type_accuracies"] = []
        self.orig_results_dict["type_f1s"] = []
        self.orig_results_dict["per_type_f1s_dict"] = {}
        for i in range(self.num_classes):
            if i != self.background_index:
                self.orig_results_dict["per_type_f1s_dict"][i] = []
        self.orig_results_dict["nucleus_type_tp_tn"] = 0
        self.orig_results_dict["nucleus_type_fp_fn"] = 0

        
        self.adv_results_dict["nucleus_type_accuracies"] = []
        self.adv_results_dict["type_f1s"] = []
        self.adv_results_dict["per_type_f1s_dict"] = {}
        for i in range(self.num_classes):
            if i != self.background_index:
                self.adv_results_dict["per_type_f1s_dict"][i] = []
        self.adv_results_dict["nucleus_type_tp_tn"] = 0
        self.adv_results_dict["nucleus_type_fp_fn"] = 0


        #instance segmentation nucleus wise metrics
        self.orig_results_dict["dice_indices"] = []
        self.orig_results_dict["jaccard_indices"] = []
        self.orig_results_dict["instance_f1s"] = []
        self.orig_results_dict["segmentation_qualities"] = []
        self.orig_results_dict["panoptic_qualities"] = []
        self.orig_results_dict["jaccard_plus_indices"] = []
        self.orig_results_dict["nucleus_instance_tp"] = 0 
        self.orig_results_dict["nucleus_instance_fp"] = 0
        self.orig_results_dict["nucleus_instance_fn"] = 0
        self.orig_results_dict["paired_IoU_sum"] = 0.0

        self.adv_results_dict["dice_indices"] = []
        self.adv_results_dict["jaccard_indices"] = []
        self.adv_results_dict["instance_f1s"] = []
        self.adv_results_dict["segmentation_qualities"] = []
        self.adv_results_dict["panoptic_qualities"] = []
        self.adv_results_dict["jaccard_plus_indices"] = []
        self.adv_results_dict["nucleus_instance_tp"] = 0 
        self.adv_results_dict["nucleus_instance_fp"] = 0
        self.adv_results_dict["nucleus_instance_fn"] = 0
        self.adv_results_dict["paired_IoU_sum"] = 0.0
        
        count = 0

        #HOW ARE WE GOING TO DO THIS WITH THE FILELOADER THAT WE HAVE 
        for inputs, masks in self.dataset:


            count += 1
            
            inputs = np.expand_dims(inputs.transpose((2,0,1)), axis=0)

            if adversarial:
                

                attack_results_dict = self.attack.get_adversarial_images(inputs, targets=masks, reset_weights=True, return_masks=True, return_transform_weights=return_weights_boolean)

                #Generate orig_pred_instance map and map info dict
                cropped_inputs = cropping_center(inputs, self.model_input_shape, batch=True)
                orig_pred_map = gen_inference_pred_map(cropped_inputs, self.model, self.device)
                orig_pred_inst, orig_pred_inst_info_dict = process(orig_pred_map, nr_types=self.num_classes, return_centroids=True)

                #Generate orig_truth_instance map and map info dict
                inst_maps = masks[...,0]
                type_maps = masks[...,1]
                orig_truth_inst = cropping_center(inst_maps, self.model_output_shape, batch=True)
                orig_truth_inst = fix_mirror_padding(orig_truth_inst)
                orig_truth_inst = remap_label(orig_truth_inst) #Not sure about this

                orig_truth_type = cropping_center(type_maps, self.model_output_shape, batch=True)
                orig_truth_inst_info_dict = gen_truth_inst_info(orig_truth_inst, orig_truth_type)
                
                
                #Generate adv_pred_instance_map and the map info dict
                adv_pred_map = gen_inference_pred_map(attack_results_dict["adv_inputs"], self.model, self.device)
                adv_pred_inst, adv_pred_inst_info_dict = process(adv_pred_map, nr_types=self.num_classes, return_centroids=True)

                #Generate adv_truth_instance_map and the map info dict
                adv_truth_inst = attack_results_dict["truth_inst_map"]
                adv_truth_inst_info_dict = attack_results_dict["truth_inst_info"]
                

                # print(orig_truth_inst_info_dict.keys())
                # print(adv_truth_inst_info_dict.keys())
                # print(orig_pred_inst_info_dict.keys())
                # print(adv_pred_inst_info_dict.keys())
                # print("")
                # print("")


                #Generate the semantic segmentation maps.
                adv_truth_semantic_seg_mask = inst_to_semantic_segmentation_mask(adv_truth_inst, adv_truth_inst_info_dict)
                adv_pred_semantic_seg_mask = inst_to_semantic_segmentation_mask(adv_pred_inst, adv_pred_inst_info_dict)
                orig_truth_semantic_seg_mask = inst_to_semantic_segmentation_mask(orig_truth_inst, orig_truth_inst_info_dict)
                orig_pred_semantic_seg_mask = inst_to_semantic_segmentation_mask(orig_pred_inst, orig_pred_inst_info_dict)

##################  
                if display:
                    
                    fig, axs = plt.subplots(3, 5, figsize=(16, 8))


                    orig_img = cropping_center(inputs, self.model_output_shape, batch=True).squeeze().transpose(1,2,0).astype("uint8")
                    output_img = attack_results_dict["adv_inputs"].detach().cpu().squeeze().permute(1,2,0).numpy()
                    output_img = cropping_center(output_img, (80,80)).astype("uint8")

                    inst_map = cropping_center(masks[..., 0], (80,80), batch=True)
                    inst_map = fix_mirror_padding(inst_map)
                    inst_map = remap_label(inst_map)

                  

                    perturbation = (orig_img.astype(np.float32)-output_img.astype(np.float32))

                    rmse = np.sqrt(np.mean(np.power(perturbation, 2)))
                    l_inf = np.max(np.abs(perturbation))
                    abs_pert = np.abs(perturbation).astype(np.uint8)
                    if scale_perturbation:
                      abs_pert = math.floor((255/np.max(abs_pert)))*abs_pert

                    # HOW DO I GET THE ORIGINAL IMAGE HERE?
                    axs[0,0].imshow(orig_img)
                    axs[0,1].imshow(orig_truth_inst)
                    axs[0,2].imshow(orig_pred_inst)

                    axs[1,0].imshow(output_img)
                    axs[1,1].imshow(adv_truth_inst)
                    axs[1,2].imshow(adv_pred_inst)

                    im = axs[0,3].imshow(orig_truth_semantic_seg_mask, cmap="rainbow", vmin=0, vmax=self.num_classes-1)
                    axs[0,4].imshow(orig_pred_semantic_seg_mask, cmap="rainbow", vmin=0, vmax=self.num_classes-1)
                    axs[1,3].imshow(adv_truth_semantic_seg_mask, cmap="rainbow", vmin=0, vmax=self.num_classes-1)
                    axs[1,4].imshow(adv_pred_semantic_seg_mask, cmap="rainbow", vmin=0, vmax=self.num_classes-1)

                    axs[2, 0].imshow(abs_pert)

                    title_0 = f"Images"
                    title_1 = f"Truth Inst"
                    title_2 = f"Prediction Inst"
                    title_3 = f"Truth Type"
                    title_4 = f"Pred Type"

                    y0_label = f"Orig"
                    y1_label = f"Adv"
                    y2_label = f"|Perturbation|"
                    axs[0,0].set_title(title_0)
                    axs[0,1].set_title(title_1)
                    axs[0,2].set_title(title_2)
                    axs[0,3].set_title(title_3)
                    axs[0,4].set_title(title_4)
                  

                    axs[0,0].set_ylabel(y0_label)
                    axs[1,0].set_ylabel(y1_label)
                    axs[2,0].set_ylabel(y2_label)

                    axs[0, 0].text(-0.5, 0.5, y0_label, fontdict={'size':12},
                      verticalalignment='center', transform=axs[0, 0].transAxes)
                    axs[1, 0].text(-0.5, 0.5, y1_label, fontdict={'size':12},
                      verticalalignment='center', transform=axs[1, 0].transAxes)
                    # axs[2, 0].text(-0.5, 0.5, y2_label, fontdict={'size':12},
                    #   verticalalignment='center', transform=axs[2, 0].transAxes)

                    axs[2,0].spines['right'].set_visible(False)
                    axs[2,0].spines['top'].set_visible(False)
                    axs[2,0].spines['bottom'].set_visible(False)
                    axs[2,0].spines['left'].set_visible(False)
                    axs[2,0].get_xaxis().set_ticks([])
                    axs[2,0].get_yaxis().set_ticks([])


                    cax = fig.add_axes([0.9, 0.4, 0.05, 0.48])
                    cb = plt.colorbar(im, cax=cax, values=list(range(self.num_classes)))


                    x_lab = f"L2: {rmse:.3f}, L infinity: {l_inf:.3f}"
                    axs[2,0].set_xlabel(x_lab)

                    axs[0,0].axis("off")
                    axs[0,1].axis("off")
                    axs[0,2].axis("off")
                    axs[1,0].axis("off")
                    axs[1,1].axis("off")
                    axs[1,2].axis("off")

                    axs[0,3].axis("off")
                    axs[0,4].axis("off")

                    axs[1,3].axis("off")
                    axs[1,4].axis("off")

                    axs[2,3].axis("off")
                    axs[2,4].axis("off")

                    axs[2,1].axis("off")
                    axs[2,2].axis("off")

                    # plt.savefig("test.png")
                    plt.show()
###################
            
                # plt.imshow(orig_img)
                # plt.show()
                #REMEMBER, WE NEED TO BE ABLE TO CONSIDER THE CASE WHERE EVEN THE PREDICTED VALUES HAVE EMPTYNESS? OR MAYBE NOT?

                #HERE WE HAVE A PROBLEM WITH THE CALCULATIONS NOT WORKING WHEN THERE ARE NOT INSTANCES IN THE ORIGINAL OR THE PRED,
                #REALLY DO NEED A BETTER SOLUTION FOR THIS, COULD TRY MAKING THE ATTACK DO MULTIPLE WINDOWS.
                # 
                if (not orig_truth_inst_info_dict) or (not adv_truth_inst_info_dict):
                    print("Didn't make it")
                    continue

                #NEED TO OUTPUT THE INST MAPS AND THE IMAGES TO SEE WHATS GOING ON

                


                #Calculate the semantic segmentation metrics for original image
                calculate_semantic_seg_metrics(orig_pred_semantic_seg_mask, orig_truth_semantic_seg_mask, self.orig_results_dict, self.num_classes, self.background_index)
                #Calculate the semantic segmentation metrics for adversarial image
                calculate_semantic_seg_metrics(adv_pred_semantic_seg_mask, adv_truth_semantic_seg_mask, self.adv_results_dict, self.num_classes, self.background_index)
                

                

                #THERE IS A MISTAKE, NEED TO FIX IT. TAKE THE ZOOM IN FUNCTION AS AN EXAMPLE
                # print("")
                # print("orig")
                #Calculate the instance segmentation stats for original 

                calculate_instance_segmentation_stats(orig_pred_inst, orig_truth_inst, self.orig_results_dict)
                #Calculate the instance segmentation stats for original image
                # print("")
                # print("adv")                
                calculate_instance_segmentation_stats(adv_pred_inst, adv_truth_inst, self.adv_results_dict)
                

                #Calculate the type classification stats for the original image
                calculate_type_classification_stats(orig_pred_inst, orig_pred_inst_info_dict, orig_truth_inst, orig_truth_inst_info_dict, self.orig_results_dict)
                #Calculate the type classfication stats for the original image
                calculate_type_classification_stats(adv_pred_inst, adv_pred_inst_info_dict, adv_truth_inst, adv_truth_inst_info_dict, self.adv_results_dict)


                #Measure how large perturbation is
                if perturbation_measure is not None:
                    input_pert = (attack_results_dict["adv_inputs"].detach() - torch.tensor(cropped_inputs, dtype=torch.float32))
                    pert_measures.append(perturbation_measure(input_pert))

                #Does not work for Stochastic Search (stochastic search doesnt select the best transform)
                if weight_measure is not None:
                    weight_pert = (self.attack.transform.weights - self.attack.transform.base_weights).cpu().detach()
                    weight_measures.append(weight_measure(weight_pert))

        #THE SEMANTIC SEGMENTATION METRICS
        #pixel_wise IoU
        self.orig_results_dict["pixel_wise_nucleus_IoU"] = self.orig_results_dict["nucleus_pixel_tp"]/(self.orig_results_dict["nucleus_pixel_tp"]+self.orig_results_dict["nucleus_pixel_fp"]+self.orig_results_dict["nucleus_pixel_fn"]+1.0e-6)
        self.adv_results_dict["pixel_wise_nucleus_IoU"] = self.adv_results_dict["nucleus_pixel_tp"]/(self.adv_results_dict["nucleus_pixel_tp"]+self.adv_results_dict["nucleus_pixel_fp"]+self.adv_results_dict["nucleus_pixel_fn"]+1.0e-6)

        #pixel_wise IoU for each type
        self.orig_results_dict["pixel_wise_type_IoU"] = {}
        self.adv_results_dict["pixel_wise_type_IoU"] = {}
        for i in range(self.num_classes):
            self.orig_results_dict["pixel_wise_type_IoU"][i] = self.orig_results_dict["type_pixel_tp"][i]/(self.orig_results_dict["type_pixel_tp"][i]+self.orig_results_dict["type_pixel_fp"][i]+self.orig_results_dict["type_pixel_fn"][i]+1.0e-6)
            self.adv_results_dict["pixel_wise_type_IoU"][i] = self.adv_results_dict["type_pixel_tp"][i]/(self.adv_results_dict["type_pixel_tp"][i]+self.adv_results_dict["type_pixel_fp"][i]+self.adv_results_dict["type_pixel_fn"][i]+1.0e-6)

        
        #pixel_wise Dice coefficient
        self.orig_results_dict["pixel_wise_nucleus_Dice"] = 2*self.orig_results_dict["nucleus_pixel_tp"]/(2*self.orig_results_dict["nucleus_pixel_tp"]+self.orig_results_dict["nucleus_pixel_fp"]+self.orig_results_dict["nucleus_pixel_fn"]+1.0e-6)
        self.adv_results_dict["pixel_wise_nucleus_Dice"] = 2*self.adv_results_dict["nucleus_pixel_tp"]/(2*self.adv_results_dict["nucleus_pixel_tp"]+self.adv_results_dict["nucleus_pixel_fp"]+self.adv_results_dict["nucleus_pixel_fn"]+1.0e-6)

        #pixel_wise Dice for each type
        self.orig_results_dict["pixel_wise_type_Dice"] = {}
        self.adv_results_dict["pixel_wise_type_Dice"] = {}
        for i in range(self.num_classes):
            self.orig_results_dict["pixel_wise_type_Dice"][i] = 2*self.orig_results_dict["type_pixel_tp"][i]/(2*self.orig_results_dict["type_pixel_tp"][i]+self.orig_results_dict["type_pixel_fp"][i]+self.orig_results_dict["type_pixel_fn"][i]+1.0e-6)
            self.adv_results_dict["pixel_wise_type_Dice"][i] = 2*self.adv_results_dict["type_pixel_tp"][i]/(2*self.adv_results_dict["type_pixel_tp"][i]+self.adv_results_dict["type_pixel_fp"][i]+self.adv_results_dict["type_pixel_fn"][i]+1.0e-6)

        #THINK THAT THERE MAY BE A PROBLEM WITH THE PIXEL ACCURACY
        #pixel_wise type accuracy 
        self.orig_results_dict["all_type_pixel_accuracy"] = self.orig_results_dict["all_type_pixel_tp_tn"]/self.orig_results_dict["all_type_pixel_tp_tn_fp_fn"]
        self.adv_results_dict["all_type_pixel_accuracy"] = self.adv_results_dict["all_type_pixel_tp_tn"]/self.adv_results_dict["all_type_pixel_tp_tn_fp_fn"]



        #THE NUCLEUS INSTANCE SEGMENTATION METRICS
        self.orig_results_dict["instance_wise_Dice"] = self.orig_results_dict["nucleus_instance_tp"]/(self.orig_results_dict["nucleus_instance_tp"]+0.5*self.orig_results_dict["nucleus_instance_fp"]+0.5*self.orig_results_dict["nucleus_instance_fn"]+1.0e-6)
        self.adv_results_dict["instance_wise_Dice"] = self.adv_results_dict["nucleus_instance_tp"]/(self.adv_results_dict["nucleus_instance_tp"]+0.5*self.adv_results_dict["nucleus_instance_fp"]+0.5*self.adv_results_dict["nucleus_instance_fn"]+1.0e-6)
        
        self.orig_results_dict["instance_wise_segmentation_quality"] = self.orig_results_dict["paired_IoU_sum"]/(self.orig_results_dict["nucleus_instance_tp"] + 1.0e-6)
        self.adv_results_dict["instance_wise_segmentation_quality"] = self.adv_results_dict["paired_IoU_sum"]/(self.adv_results_dict["nucleus_instance_tp"] + 1.0e-6)

        self.orig_results_dict["instance_wise_panoptic_quality"] = self.orig_results_dict["instance_wise_Dice"]*self.orig_results_dict["instance_wise_segmentation_quality"]
        self.adv_results_dict["instance_wise_panoptic_quality"] = self.adv_results_dict["instance_wise_Dice"]*self.adv_results_dict["instance_wise_segmentation_quality"]

        #THE NUCLEUS WISE TYPE CLASSIFICATION METRICS
        self.orig_results_dict["instance_wise_type_classification_accuracy"] = self.orig_results_dict["nucleus_type_tp_tn"] / (self.orig_results_dict["nucleus_type_tp_tn"] + self.orig_results_dict["nucleus_type_fp_fn"]+1.0e-6)
        self.adv_results_dict["instance_wise_type_classification_accuracy"] = self.adv_results_dict["nucleus_type_tp_tn"] / (self.adv_results_dict["nucleus_type_tp_tn"] + self.adv_results_dict["nucleus_type_fp_fn"]+1.0e-6)



        #Add outputs, their labels, the advesarial image's results and other to the results dict.
        #CREATE A FUNCTION FOR LOGGING THESE, CAN BE USED HERE AND IN PREVOUS ONE FOR SEMANTIC SEGMENTATION. 
        return self.orig_results_dict, self.adv_results_dict
    
    def display_results_by_index(self, indices, targets, criterion):
        # for i in indices:

        
        pass



