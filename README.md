# REET2.0
REET2.0 is the Robustness Evaulutation and Enhanchement Toolbox (https://github.com/alexjfoote/reetoolbox.git) extended to include nuclear semantic segmentation models and nuclear instance segmentation models.

The toolbox takes algorithms from robustness analysis and combines them with digital pathology augmentations, aiming to simulate the most challening clincial conditions that computational pathology CNNs are likely to face. The toolbox can use these algorithms to evaluate the clinical robustness of nuclear segmenation models and provide performance metrics for given clinical variations. For a detailed understanding of the algorithms developed for the toolbox, see [Dissertation.pdf](Dissertation.pdf).

### Instructions
For instructions on training nuclear segementaion models, see [Training_Instructions.ipynb](Training_Instructions.ipynb) (NEEDS UPDATING).

For instructions on evaluating the clinical robustness of nuclear segmentation models see [Evaluation_Instructions.ipynb](Evaluation_Instructions.ipynb) (NEEDS UPDATING).

### Large Files
Download pretrained U-Net models: https://drive.google.com/drive/folders/1Oo6HzRdfBAMPjB8rcH1dfBoWzg1meTOC?usp=drive_link.

Download pretrained HoVer-Net models: https://drive.google.com/drive/folders/1qlUyAaSwHVLf9iO8C1q4QAZ9rZpQWRgk?usp=drive_link.

Download the CoNSeP dataset, extracted in tiles as used in our experiments: https://drive.google.com/drive/folders/1e_dDLSmRFSwIyIBMFs9-hfQXjAiLUokF?usp=drive_link.

### Acknowledgements

This work builds off REET: https://github.com/alexjfoote/reetoolbox.git.

The HoVer-Net model and much of the training and evaluating code for instance segmentation models comes from: https://github.com/vqdang/hover_net.

The U-Net model was taken from: https://github.com/milesial/Pytorch-UNet.



