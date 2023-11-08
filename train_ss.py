from numpy.matrixlib.defmatrix import N
from segmentation.unet import UNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np
import torch.nn.functional as F
from reetoolbox.file_loader import FileLoader
from reetoolbox.insert_trainers import ss_insert_trainer
from reetoolbox.utils import cropping_center
from nuc_inst_segmentation.hovernet.utils import convert_pytorch_checkpoint
import random

def get_cuda_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device





#THESE ARE THE PARAMETERS FOR TRAINING YOUR MODEL
num_classes = 5
batch_size = 32
#Set visualise to true if you want to print the images, these images are saved to test.png
visualise = True
#Set attack to true if you want to do adversarial training.
attack = True
#Set setup_augmentor to True if you want random augmentations
setup_augmentor = False 
device = get_cuda_device()
num_epochs = 200
loss_plot_path = "segmentation/output/plot.png"
#Path to save model
model_save_path = "segmentation/output/real_adv_b64_epoch"
attack_input_shape = (540, 540) #Should be atleast double model input shape if you are attacking, I use (540,540), make it the model shape if attack is set to false
model_input_shape = (256, 256)
if not attack:
    attack_input_shape = model_input_shape
#Put a list of your dataset directorys below
training_patches_directory_list = ["Consep_patches_540x540/train/540x540_164x164"]
testing_patches_directory_list = ["Consep_patches_540x540/valid/540x540_164x164"]

#A list of transforms you wish to apply.
attack_transform_list = ["Pixel"]
transform_loss_function_key = "Default batch" #Use the keys given in maps.py for loss function to use with adversarial transform, have to specify if you are attacking by batch



#Datasets
train_FL = FileLoader(training_patches_directory_list, input_shape=attack_input_shape, setup_augmentor=setup_augmentor)
test_FL = FileLoader(testing_patches_directory_list, input_shape=model_input_shape, setup_augmentor=setup_augmentor)


print(f"[INFO] found {len(train_FL)} examples in the training set...")
print(f"[INFO] found {len(test_FL)} examples in the test set...")

# create the training and test data loaders
trainLoader = DataLoader(train_FL, shuffle=True,
	batch_size=batch_size,
	num_workers=os.cpu_count())
testLoader = DataLoader(test_FL, shuffle=False,
	batch_size=int(batch_size/2),
	num_workers=os.cpu_count())


# initialize our UNet model
unet = UNet(n_channels=3, n_classes=num_classes, bilinear=False).to(device)

#EXPERIMENT WITH IT
# state_dict = torch.load("nuc_inst_segmentation/output/ImageNet-ResNet50-Preact_pytorch.tar")["desc"]
# state_dict = convert_pytorch_checkpoint(state_dict)
# load_feedback = unet.load_state_dict(state_dict, strict=False)



# Cross entropy also performs softmax
lossFunc = CrossEntropyLoss()
opt = Adam(unet.parameters(), lr=1.0e-4)

#Initialise the learning rate scheduler
scheduler = ReduceLROnPlateau(opt, 'min', patience = 15)

# calculate steps per epoch for training and test set
trainSteps = len(train_FL) // batch_size
testSteps = len(test_FL) // batch_size

# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
check = 0
currentBestLoss = 1000 #Could make this better

if attack:
    attacker = ss_insert_trainer(unet, attack_transform_list, model_input_shape, transform_loss_function_key, num_classes, device=device)

#JUST FOR TESTING MODELS TEST LOSS, CAN REMOVE
# totalTestLoss = 0
# with torch.inference_mode():

#   # set the model in evaluation mode
#   unet.eval()

#   # loop over the validation set
#   for (imgs, type_masks) in testLoader:
#     # send the input to the device
#     (imgs, type_masks) = (imgs.to(device), type_masks.to(device))

#     imgs = imgs.permute(0,3,1,2).type(torch.float32)
#     type_masks = F.one_hot(type_masks.type(torch.int64), num_classes=num_classes).permute(0,3,1,2).type(torch.float32)

#     # make the predictions and calculate the validation loss
#     pred = unet(imgs)
#     totalTestLoss += lossFunc(pred, type_masks).item()

# print(f"test_loss = {totalTestLoss/testSteps}")


t1=time.time()
for e in tqdm(range(num_epochs)):

  # set the model in training mode
  unet.train()

  # initialize the total training and validation loss
  totalTrainLoss = 0
  totalTestLoss = 0

  # loop over the training se
  # print(f"time since start after {e} epochs {time.time()-t1}")
  t1= time.time()
  for (i, (orig_imgs, type_masks)) in enumerate(trainLoader):

    # send the input to the device
    (imgs, type_masks) = (orig_imgs.to(device), type_masks.to(device))
    
    imgs = imgs.permute(0,3,1,2).type(torch.float32)
    type_masks = F.one_hot(type_masks.type(torch.int64), num_classes=num_classes).permute(0, 3, 1, 2).type(torch.float32)

    if attack:
        imgs, type_masks = attacker.attack_images(imgs, type_masks, 1)


    # perform a forward pass and calculate the training loss
    if visualise:
        index = random.randint(0, batch_size-1)
        img = imgs[index].detach().cpu().permute(1,2,0).numpy().astype("uint8")
        msk = type_masks[index].detach().cpu().permute(1,2,0).numpy().astype("uint8")
        msk = np.argmax(msk, axis=-1)
        orig = cropping_center(orig_imgs[index], (256,256))
        figure, axs = plt.subplots(1, 4, figsize=(10,10))
        axs[0].imshow(img)
        axs[1].imshow(orig)
        axs[2].imshow(msk)
        axs[3].imshow(np.abs(orig-img)*100)
        plt.savefig("tests.png")
        breakpoint()

    pred = unet(imgs)

    loss = lossFunc(pred, type_masks)


    # first, zero out any previously accumulated gradients, then
    # perform backpropagation, and then update model parameters
    opt.zero_grad()
    loss.backward()
    opt.step()

    # add the loss to the total training loss so far
    totalTrainLoss += loss.item()
    l2 = time.time()
    # print(f"full loop time: {l2-l1}")


  # switch to inference mode (change back to no_grad if we get errors)
  with torch.inference_mode():

    # set the model in evaluation mode
    unet.eval()

    # loop over the validation set
    for (imgs, type_masks) in testLoader:
      # send the input to the device
      (imgs, type_masks) = (imgs.to(device), type_masks.to(device))

      imgs = imgs.permute(0,3,1,2).type(torch.float32)
      type_masks = F.one_hot(type_masks.type(torch.int64), num_classes=num_classes).permute(0,3,1,2).type(torch.float32)

      # make the predictions and calculate the validation loss
      pred = unet(imgs)
      totalTestLoss += lossFunc(pred, type_masks).item()

  # calculate the average training and validation loss
  avgTrainLoss = totalTrainLoss / trainSteps
  avgTestLoss = totalTestLoss / testSteps
  
  #Adjust scheduler if the test loss stagnates
  scheduler.step(avgTrainLoss)

  #Save the model
  if (avgTestLoss < currentBestLoss) and (e>10):
    currentBestLoss=avgTestLoss
    file_name = f"{model_save_path}_{e}.pth"
    torch.save(unet, file_name)

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

#Parts of this code were taken from pyimagesearch


