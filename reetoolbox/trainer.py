from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
import time
import copy
import random
from reetoolbox.evaluator import Evaluator
from reetoolbox.metrics import get_metrics, rmse
from torchmetrics.classification import MulticlassAccuracy

#Optimises and applies transforms to image in a sequential order.
def apply_transforms(model, inputs, labels, adv_optimisers, k, reset_weights=True):
    num = len(adv_optimisers)
    if num > k:
        sub_opt = random.sample(adv_optimisers, k)
    else:
        sub_opt = adv_optimisers

    for adv_opt in sub_opt:
        adv_opt.model = model
        inputs = adv_opt.get_adversarial_images(inputs, targets=labels, reset_weights=reset_weights)
    return inputs

#Ran from training bout.
def run_val(model, loader, criterion, device="cuda:0"):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    num_examples = 0

    with torch.set_grad_enabled(False):
        for batch_no, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) 
            loss = criterion(outputs, labels)

            num_examples += len(preds)
            running_corrects += torch.sum(preds == labels.data)
            running_loss += loss.item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = running_corrects.double() / num_examples
    return epoch_loss, epoch_acc


def print_update(phase, epoch_loss, epoch_acc):
    print(f'{phase}: Loss: {round(epoch_loss, 3)} Acc: {round(epoch_acc.item(), 3)}')

#Run from adv_free
#DOES NOT HANDLE EVALUATE AND INFERENCE MODE VERY WELL.
def training_bout(model, train_loader, val_loader, optimiser, criterion, adversarial=False, attacks=[], epochs=10, m=10,
                  last_n=10, k=2, device="cuda:0"):
    acc_history = []

    #Not convinced that we need to deepcopy.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoch = 1
    num_examples = 0
    running_loss = 0.0
    running_corrects = 0

    print(f'Epoch {epoch}/{epochs}')
    print('-' * 10)
    t_full = time.time()
    t1 = time.time()

    for i in range(int(epochs / m)):
        model.train()

        batch_count = 0
        batches_per_epoch = len(train_loader)

        for batch_no, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            #Sample the attacks
            if attacks:
                num_attacks = len(attacks)
                if num_attacks > k:
                    sub_attacks = random.sample(attacks, k)
                else:
                    sub_attacks = attacks

            with torch.set_grad_enabled(True):

                #Repeat batch m times
                for repetition in range(m):
                    batch_count += 1

                    optimiser.zero_grad()

                    if adversarial and attacks:
                        adv_inputs = inputs.clone()
                        for attack in sub_attacks:
                            attack.model = model
                            if repetition == 0:
                                #Think about targets here.
                                adv_inputs = attack.get_adversarial_images(adv_inputs, targets=labels, reset_weights=True)
                            else:
                                adv_inputs = attack.get_adversarial_images(adv_inputs, targets=labels, reset_weights=False)

                        outputs = model(adv_inputs)
                    else:
                        outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    #DONT THINK WE NEED TO RETAIN GRAPH, Maybe for transforms, check?
                    loss.backward(retain_graph=True)
                    optimiser.step()

                    num_examples += len(preds)
                    running_corrects += torch.sum(preds == labels.data)
                    running_loss += loss.item()

                    if batch_count % batches_per_epoch == 0:
                        epoch_loss = running_loss / batches_per_epoch
                        epoch_acc = running_corrects.double() / num_examples
                        print_update("Train", epoch_loss, epoch_acc)

                        epoch_loss_v, epoch_acc_v = run_val(model, val_loader, criterion, device=device)
                        print_update("Val", epoch_loss_v, epoch_acc_v)

                        print(f"{round(time.time() - t1, 2)}s")

                        epoch += 1
                        if epoch <= epochs:
                            print(f'Epoch {epoch}/{epochs}')
                            print('-' * 10)
                            t1 = time.time()

                        num_examples = 0
                        running_loss = 0.0
                        running_corrects = 0

                        # deep copy the model
                        if epoch_acc_v >= best_acc and (last_n is None or epochs - epoch <= last_n):
                            best_acc = epoch_acc_v
                            best_model_wts = copy.deepcopy(model.state_dict())

                        acc_history.append(np.array([epoch_acc.cpu().detach(), epoch_acc_v.cpu().detach()]))

    print(f"Took: {round(time.time() - t_full, 2)}s")
    print(f'Best val Acc: {best_acc}')
    model.load_state_dict(best_model_wts)
    return model, np.array(acc_history)

#Run from train.

def train_adv_free(model, dataloaders, criterion, optimiser, initial_epochs=0, adv_epochs=30, m=10, last_n=10,
                   attacks=[], k=2, device="cuda:0"):
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    # Initial standard training
    print("Initial Epochs... ")
    model, initial_hist = training_bout(model, train_loader, val_loader, optimiser, criterion, m=1,
                                        epochs=initial_epochs, adversarial=False, attacks=[], last_n=last_n, k=k,
                                        device=device)
    # Free adversarial training
    print("Adversarial Epochs...")
    model, adv_hist = training_bout(model, train_loader, val_loader, optimiser, criterion, m=m, epochs=adv_epochs,
                                    adversarial=True, attacks=attacks, last_n=last_n, k=k, device=device)
    return model, initial_hist, adv_hist

#Currently not found where this is used.
def load_train(path=None, load_saved=False, pretrained=True, n_classes=2, hidden_size=512, device="cuda:0"):
    model = models.resnet18(pretrained=pretrained)

    if load_saved:
        model.fc = nn.Sequential(
            nn.Linear(hidden_size, n_classes),
            nn.LogSoftmax(dim=1))
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    else:
        model.fc = nn.Sequential(
            nn.Linear(hidden_size, n_classes),
            nn.LogSoftmax(dim=1))

    # Send the model to GPU
    model = model.to(device)

    model.train()

    # Gather the parameters to be optimized/updated in this run
    params_to_update = model.parameters()
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return model, optimizer_ft

#This method will no longer work. Could easily make this though.It currently is used in the training tutorial.Make one for semantic segmentation and classification.
#It goes over each of the evaluator params and uses it to produce and evaluator to perform the desired task.
def evaluation(model, all_evaluator_params, device="cuda:0"):
  for evaluator_params in all_evaluator_params:
    print(evaluator_params["Transform"])
    evaluator = Evaluator(model, **evaluator_params, device=device)
    results = evaluator.predict(adversarial=True, perturbation_measure=rmse, weight_measure=None)
    get_metrics(results)

#Optimiser_ft is for the training model
#Optimiser and params are for optimizing tranforms.
#Train_params: e.g.
# train_params = {
#     "dataloaders": dataloaders_dict,
#     "criterion": nn.CrossEntropyLoss(),
#     "initial_epochs": 0,
#     "adv_epochs": 10,
#     "m": 1,
#     "last_n": 10,
#     "k": 1
# }
def train_classification(model, model_optimiser, transform_optimisers_and_params, train_params, device):
    #Instastiate the transform optimisers and store in all attacks
    all_attacks = []
    for TransformOptimiser, attack_params in transform_optimisers_and_params:
        all_attacks.append(TransformOptimiser(model, **attack_params, device=device))


    new_model, initial_hist, adv_hist = train_adv_free(model=model, optimizer=model_optimiser, attacks=all_attacks,
                                                       **train_params, device=device)
    return new_model


#########################################################################################################################################################
#NEED TO ADD A SCALE FACTOR
def train_semantic_segmentation(model, model_optimiser, transform_optimisers_and_params, train_params, device, background_index = 0, num_classes=6):
    #Instastiate the transform optimisers and store in all attacks
    all_attacks = []
    for TransformOptimiser, attack_params in transform_optimisers_and_params:
        all_attacks.append(TransformOptimiser(model, **attack_params, device=device, scale_factor=255))


    new_model, initial_hist, adv_hist = train_adv_free_semantic_segmentation(model=model, optimiser=model_optimiser, attacks=all_attacks,
                                                       **train_params, device=device, background_index=background_index, num_classes=6)
    return new_model

#TotalNumberOfEpochs-last_n is the smallest epoch that our best model can be chosen from.
def train_adv_free_semantic_segmentation(model, dataloaders, criterion, optimiser, background_index = 0, initial_epochs=0, adv_epochs=30, m=10, last_n=None,
                   attacks=[], k=2, device="cuda:0", num_classes=6):
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    #Initial standard training
    adv_hist = []
    initial_hist = []
    if (initial_epochs>0): 
      print("Initial Epochs... ")
      model, initial_hist = training_bout_semantic_segmenation(model, train_loader, val_loader, optimiser, criterion, m=1,
                                          epochs=initial_epochs, adversarial=False, attacks=[], last_n=last_n, k=k,
                                        device=device, background_index=background_index, num_classes=num_classes)
    #Free adversarial training
    if (adv_epochs>0):
        print("Adversarial Epochs...")
        model, adv_hist = training_bout_semantic_segmenation(model, train_loader, val_loader, optimiser, criterion, m=m, epochs=adv_epochs,
                                      adversarial=True, attacks=attacks, last_n=last_n, k=k, device=device, background_index=background_index, num_classes=num_classes)
    return model, initial_hist, adv_hist



def training_bout_semantic_segmenation(model, train_loader, val_loader, optimiser, criterion, adversarial=False, attacks=[], epochs=10, m=10,
                  last_n=10, k=2, device="cuda:0", background_index=0, num_classes=6):
    acc_history = []

    #Not convinced that we need to deepcopy.
    #HOW ARE WE GOING TO DO THIS FOR SEMANTIC SEGMENTAION.
    best_model = copy.deepcopy(model)
    best_acc = 0.0

    epoch = 1
    num_examples = 0
    running_loss = 0.0
    running_corrects = 0
    running_accuracys = []

    print(f'Epoch {epoch}/{epochs}')
    print('-' * 10)
    t_full = time.time()
    t1 = time.time()

    pixel_acc_no_background = MulticlassAccuracy(num_classes, average='micro', ignore_index=background_index).to(device)

    for i in range(int(epochs / m)):
        model.train()

        batch_count = 0
        batches_per_epoch = len(train_loader)

        for batch_no, (inputs, masks) in enumerate(train_loader):
            inputs = inputs.to(device)
            masks = masks.to(device)

            #Sample the attacks
            if attacks:
                num_attacks = len(attacks)
                if num_attacks > k:
                    sub_attacks = random.sample(attacks, k)
                else:
                    sub_attacks = attacks


                #Repeat batch m times
            for repetition in range(m):

                with torch.set_grad_enabled(True):

                    batch_count += 1

                    optimiser.zero_grad()

                    if adversarial and attacks:
                        adv_inputs = inputs.clone()
                        for attack in sub_attacks:
                            attack.model = model
                            if repetition == 0:
                                #Think about targets here.
                                adv_inputs, masks = attack.get_adversarial_images(adv_inputs, targets=masks, reset_weights=True, return_masks = True)
                            else:
                                adv_inputs, masks = attack.get_adversarial_images(adv_inputs, targets=masks, reset_weights=False, return_masks = True)

                        outputs = model(adv_inputs)
                    else:
                        outputs = model(inputs)


                    loss = criterion(outputs, masks)

                    #DONT THINK WE NEED TO RETAIN GRAPH, 
                    #loss.backward(retain_graph=True)
                    loss.backward()
                    optimiser.step()

                masks = torch.argmax(masks, dim=1)
                outputs = torch.argmax(outputs, dim=1)



                num_examples += inputs.shape[0]
                for i in range(masks.shape[0]):
                    accuracy = pixel_acc_no_background(outputs[i], masks[i])
                    running_accuracys.append(accuracy.item())

                running_loss += loss.item()

                if batch_count == batches_per_epoch:
                    epoch_loss = running_loss / batches_per_epoch
                    epoch_acc = sum(running_accuracys) / num_examples
                    print(f'Training: Loss: {round(epoch_loss, 3)} Acc: {round(epoch_acc, 3)}')

                    epoch_loss_v, epoch_acc_v = run_val_semantic_segmentation(model, val_loader, criterion, device=device, metric = pixel_acc_no_background)
                    print(f'Validation: Loss: {round(epoch_loss_v, 3)} Acc: {round(epoch_acc_v, 3)}')

                    print(f"{round(time.time() - t1, 2)}s")

                    epoch += 1
                    if epoch <= epochs:
                        print(f'Epoch {epoch}/{epochs}')
                        print('-' * 10)
                        t1 = time.time()

                    num_examples = 0
                    running_loss = 0.0
                    running_corrects = 0
                    running_accuracys = []

                    # deep copy the model
                    #Think about this if statement.
                    if epoch_acc_v >= best_acc and (last_n is None or epochs - epoch <= last_n):
                        best_acc = epoch_acc_v
                        best_model = copy.deepcopy(model)

                    acc_history.append(np.array([epoch_acc, epoch_acc_v]))

    print(f"Took: {round(time.time() - t_full, 2)}s")
    print(f'Best val Acc: {best_acc}')

    return best_model, np.array(acc_history)


def run_val_semantic_segmentation(model, loader, criterion, metric, device="cuda:0"):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    num_examples = 0
    running_accuracys = []

    with torch.inference_mode():
        for batch_no, (inputs, masks) in enumerate(loader):
            inputs = inputs.to(device)
            masks = masks.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, masks)

            masks = torch.argmax(masks, dim=1)
            outputs = torch.argmax(outputs, dim=1)

            num_examples += inputs.shape[0]
            for i in range(inputs.shape[0]):
                    accuracy = metric(outputs[i], masks[i])
                    running_accuracys.append(accuracy.item())
            running_loss += loss.item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = sum(running_accuracys) / num_examples
    return epoch_loss, epoch_acc

