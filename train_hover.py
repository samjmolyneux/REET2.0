from numpy.matrixlib.defmatrix import N
from segmentation.unet import UNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np
import torch.nn.functional as F
from reetoolbox.train_loader import FileLoader
from reetoolbox.insert_trainers import hover_insert_trainer
from reetoolbox.utils import cropping_center, crop_center_2_3
import random
from nuc_inst_segmentation.hovernet.net_desc import create_model
from nuc_inst_segmentation.hovernet.utils import convert_pytorch_checkpoint, crop_to_shape, dice_loss, mse_loss, msge_loss, xentropy_loss
from nuc_inst_segmentation.hovernet.targets import gen_targets
from collections import OrderedDict

def get_cuda_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device



#PARAMETERS FOR TRAINING THE MODEL
num_classes = 5
batch_size = 4
#Set visualise to true if you want to print the images, these images are saved to test.png
visualise = True
#Set attack to true to train on transforms.
attack = True
setup_augmentor = False #This is for random augmentations, not those from REET
device = get_cuda_device()
num_epochs = 200
loss_plot_path = "nuc_inst_segmentation/output/plot.png"
#Where to save the model
model_save_path = "nuc_inst_segmentation/output/adv_hover_final"
attack_input_shape = (270, 270) #Should be atleast double model input shape if you are attacking, I use (540,540), make it the model shape if attack is set to false
model_input_shape = (270, 270)
if not attack:
    attack_input_shape = model_input_shape
training_patches_directory_list = ["Consep_patches_540x540/train/540x540_164x164"]
testing_patches_directory_list = ["Consep_patches_540x540/valid/540x540_164x164"]
#Attack_transform_list is the list of transforms to train on.
attack_transform_list = ["Pixel"]
transform_loss_function_key = "Default batch" #Use the keys given in maps.py for loss function to use with adversarial transform, have to specify if you are attacking by batch

#If you would like to train on a pretrained model, put the weights below
pretrained_path = None

#Datasets
train_FL = FileLoader(training_patches_directory_list, input_shape=attack_input_shape, setup_augmentor=setup_augmentor, mode='train', target_gen=gen_targets)
test_FL = FileLoader(testing_patches_directory_list, input_shape=model_input_shape, setup_augmentor=False, mode='valid', target_gen=gen_targets)


print(f"[INFO] found {len(train_FL)} examples in the training set...")
print(f"[INFO] found {len(test_FL)} examples in the test set...")

# create the training and test data loaders
trainLoader = DataLoader(train_FL, shuffle=True,
	batch_size=batch_size,
	num_workers=os.cpu_count())
testLoader = DataLoader(test_FL, shuffle=False,
	batch_size=int(8),
	num_workers=os.cpu_count())


# initialize our HoVer-Net model
hover = create_model(input_ch=3, nr_types=5, freeze=False, mode="original").to(device)

#The way you load the model will differ based on if it was train with parralelisation.
if pretrained_path is not None:
    
    net_state_dict = torch.load(pretrained_path)
    net_state_dict = convert_pytorch_checkpoint(net_state_dict)
    load_feedback = hover.load_state_dict(net_state_dict, strict=False)

    #IF YOU WISH TO TRAIN ON A PARRALLEL MODEL, LIKE THE ONES WE PROVIDED, REPLACE THE CODE ABOVE WITH THE CODE BELOW.
    # state_dict = torch.load(pretrained_path)["desc"]
    # state_dict = convert_pytorch_checkpoint(state_dict)
    # load_feedback = hover.load_state_dict(state_dict, strict=False)


loss_opts ={
            "np": {"bce": 1, "dice": 1},
            "hv": {"mse": 1, "msge": 1},
            "tp": {"bce": 1, "dice": 1},
}

loss_func_dict = {
    "bce": xentropy_loss,
    "dice": dice_loss,
    "mse": mse_loss,
    "msge": msge_loss,
}



opt = Adam(hover.parameters(), lr=1.0e-4)

#Initialise the learning rate scheduler
scheduler = ReduceLROnPlateau(opt, 'min', patience = 12)

# calculate steps per epoch for training and test set
trainSteps = len(train_FL) // batch_size
testSteps = len(test_FL) // 8





# totalTestLoss=0
# with torch.inference_mode():

#   # set the model in evaluation mode
#   hover.eval()

#   for batch in testLoader:

#     imgs = batch["img"]
#     true_np = batch["np_map"]
#     true_hv = batch["hv_map"]

#     #Crop to the correct shape
#     true_np = crop_center_2_3(true_np, (80,80))
#     true_hv = crop_center_2_3(true_hv, (80,80))

#     imgs = imgs.to("cuda").type(torch.float32)  # to NCHW
#     imgs = imgs.permute(0, 3, 1, 2).contiguous()

#     # HWC
#     true_np = true_np.to("cuda").type(torch.int64)
#     true_hv = true_hv.to("cuda").type(torch.float32)

#     true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
#     true_dict = {
#         "np": true_np_onehot,
#         "hv": true_hv,
#     }

#     true_tp = batch["tp_map"]
#     true_tp = crop_center_2_3(true_tp, (80,80))
#     true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
#     true_tp_onehot = F.one_hot(true_tp, num_classes=num_classes)
#     true_tp_onehot = true_tp_onehot.type(torch.float32)
#     true_dict["tp"] = true_tp_onehot              


#     pred_dict = hover(imgs)
#     pred_dict = OrderedDict(
#         [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
#     )
#     pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
#     pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)


#     ####
#     for branch_name in pred_dict.keys():
#         for loss_name, loss_weight in loss_opts[branch_name].items():
#             loss_func = loss_func_dict[loss_name]
#             loss_args = [true_dict[branch_name], pred_dict[branch_name]]
#             if loss_name == "msge":
#                 loss_args.append(true_np_onehot[..., 1])
#             term_loss = loss_func(*loss_args)
#             totalTestLoss += loss_weight * term_loss

# print(f"valid loss: {totalTestLoss/testSteps}")






# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
check = 0
currentBestLoss = 1000 #Could make this better

if attack:
    attacker = hover_insert_trainer(hover, attack_transform_list, model_input_shape, transform_loss_function_key, num_classes, device=device)



t1=time.time()
for e in tqdm(range(num_epochs)):

  # set the model in training mode
  hover.train()

  # initialize the total training and validation loss
  totalTrainLoss = 0
  totalTestLoss = 0

  # loop over the training se
  # print(f"time since start after {e} epochs {time.time()-t1}")
  t1= time.time()
  for batch in trainLoader:
    
    orig_imgs = batch["img"]
    true_np = batch["np_map"]
    true_hv = batch["hv_map"]

    imgs = orig_imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs = imgs.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = true_np.to("cuda").type(torch.int64)
    true_hv = true_hv.to("cuda").type(torch.float32)

    true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
    true_dict = {
        "np": true_np_onehot,
        "hv": true_hv,
    }

    # if model.module.nr_types is not None:
    true_tp = batch["tp_map"]
    true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
    true_tp_onehot = F.one_hot(true_tp, num_classes=num_classes)
    true_tp_onehot = true_tp_onehot.type(torch.float32)
    true_dict["tp"] = true_tp_onehot.to("cuda")
 

    if attack:
        true_dict["inst"] = batch["inst"]
        imgs, true_dict = attacker.attack_images(imgs, true_dict, 2)


    #Crop to the correct shape
    true_dict["np"] = crop_center_2_3(true_dict["np"], (80,80))
    true_dict["hv"] = crop_center_2_3(true_dict["hv"], (80,80))
    true_dict["tp"] = crop_center_2_3(true_dict["tp"], (80,80))



    if visualise:
      figure, axs = plt.subplots(2, 4, figsize=(14,14))
      index = random.randint(0, batch_size-1)

      adv = imgs[index].detach().cpu().permute(1,2,0).numpy().astype("uint8")
      orig = cropping_center(orig_imgs[index].numpy(), model_input_shape)
      pert = abs(adv-orig)
      np = torch.argmax(true_dict["np"][index],-1).detach().cpu().numpy().astype("uint8")
      tp = torch.argmax(true_dict["tp"][index],-1).detach().cpu().numpy().astype("uint8")
      true_hv = true_dict["hv"][index].detach().cpu().numpy()
      h = true_hv[...,0]
      v = true_hv[...,1]

      axs[0,0].imshow(orig)
      axs[0,1].imshow(adv)
      axs[0,2].imshow(pert)
      axs[0,3].imshow(cropping_center(adv, (80,80)))
      axs[1,0].imshow(np)
      axs[1,1].imshow(tp)
      axs[1,2].imshow(h)
      axs[1,3].imshow(v)

      plt.savefig("tests.png")
      breakpoint()

    opt.zero_grad() 
    pred_dict = hover(imgs)
    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
    )
    pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
    pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)


    ####
    loss = 0
    for branch_name in pred_dict.keys():
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
            loss_args = [true_dict[branch_name], pred_dict[branch_name]]
            if loss_name == "msge":
                loss_args.append(true_dict["tp"][..., 1])
            term_loss = loss_func(*loss_args)
            loss += loss_weight * term_loss



    loss.backward()
    opt.step()
    ####



  
    # add the loss to the total training loss so far
    totalTrainLoss += loss.item()
    l2 = time.time()
    # print(f"full loop time: {l2-l1}")


  # switch to inference mode (change back to no_grad if we get errors)
  with torch.inference_mode():

    # set the model in evaluation mode
    hover.eval()

    for batch in testLoader:

      imgs = batch["img"]
      true_np = batch["np_map"]
      true_hv = batch["hv_map"]

      #Crop to the correct shape
      true_np = crop_center_2_3(true_np, (80,80))
      true_hv = crop_center_2_3(true_hv, (80,80))

      imgs = imgs.to("cuda").type(torch.float32)  # to NCHW
      imgs = imgs.permute(0, 3, 1, 2).contiguous()

      # HWC
      true_np = true_np.to("cuda").type(torch.int64)
      true_hv = true_hv.to("cuda").type(torch.float32)

      true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
      true_dict = {
          "np": true_np_onehot,
          "hv": true_hv,
      }

      true_tp = batch["tp_map"]
      true_tp = crop_center_2_3(true_tp, (80,80))
      true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
      true_tp_onehot = F.one_hot(true_tp, num_classes=num_classes)
      true_tp_onehot = true_tp_onehot.type(torch.float32)
      true_dict["tp"] = true_tp_onehot              


      pred_dict = hover(imgs)
      pred_dict = OrderedDict(
          [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
      )
      pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
      pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)


      ####
      for branch_name in pred_dict.keys():
          for loss_name, loss_weight in loss_opts[branch_name].items():
              loss_func = loss_func_dict[loss_name]
              loss_args = [true_dict[branch_name], pred_dict[branch_name]]
              if loss_name == "msge":
                  loss_args.append(true_np_onehot[..., 1])
              term_loss = loss_func(*loss_args)
              totalTestLoss += loss_weight * term_loss

    

  # calculate the average training and validation loss
  avgTrainLoss = totalTrainLoss / trainSteps
  avgTestLoss = totalTestLoss / testSteps
  
  #Adjust scheduler if the test loss stagnates
  scheduler.step(avgTrainLoss)

  #Save the model
  currentBestLoss = avgTestLoss
  file_name = f"{model_save_path}_{e}.pth"
  torch.save(hover.state_dict(), file_name)

  # update our training history
  H["train_loss"].append(avgTrainLoss)
  H["test_loss"].append(avgTestLoss)

  # print the model training and validation information
  print("[INFO] EPOCH: {}/{}".format(e + 1, num_epochs))
  print("Train loss: {:.6f}, Test loss: {:.4f}".format(
    avgTrainLoss, avgTestLoss))

  #Print the current learning rate
  print(opt.state_dict()['param_groups'][0]['lr'])
    
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))


# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(loss_plot_path)