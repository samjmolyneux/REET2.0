import reetoolbox.semantic_segmentation_optimisers
import reetoolbox.classification_optimisers
import reetoolbox.constants
import reetoolbox.transforms
import reetoolbox.classification_optimisers
import reetoolbox.hover_optimisers
from reetoolbox.ss_eval_loader import SSFileLoader


# Dictionary from transform name to transform class
transform_dictionary = {
  "Pixel" : reetoolbox.transforms.PixelTransform,
  "Stain" : reetoolbox.transforms.StainTransform,
  "Mean" : reetoolbox.transforms.MeanTransform,
  "Cut Out" : reetoolbox.transforms.CutoutTransform,
  "Rotate" : reetoolbox.transforms.RotateTransform,
  "Crop" : reetoolbox.transforms.CropTransform,
  "Blur" : reetoolbox.transforms.BlurTransform,
  "Zoom In" : reetoolbox.transforms.ZoomInTransform,
  "Zoom Out" : reetoolbox.transforms.ZoomOutTransform,
  "HED Stain" : reetoolbox.transforms.HEDTransform,
  "Stochastic Stain" : reetoolbox.transforms.RandomStainTransform,
  "JPEG Compression" : reetoolbox.transforms.JPEGTransform
}

# Dictionary from transform to its default optimiser
default_classification_optimiser = {
  "Pixel" : reetoolbox.classification_optimisers.Classification_PGD,
  "Stain" : reetoolbox.classification_optimisers.Classification_PGD,
  "Mean" : reetoolbox.classification_optimisers.Classification_PGD,


  "Cut Out" : reetoolbox.classification_optimisers.Classification_StochasticSearch,
  "Rotate" : reetoolbox.classification_optimisers.Classification_StochasticSearch,
  "Crop" : reetoolbox.classification_optimisers.Classification_StochasticSearch,
  "Blur" : reetoolbox.classification_optimisers.Classification_StochasticSearch,
  "Zoom In" : reetoolbox.classification_optimisers.Classification_StochasticSearch,
  "Zoom Out" : reetoolbox.classification_optimisers.Classification_StochasticSearch,
  "HED Stain" : reetoolbox.classification_optimisers.Classification_StochasticSearch,
  "Stochastic Stain" : reetoolbox.classification_optimisers.Classification_StochasticSearch,
  "JPEG Compression" : reetoolbox.classification_optimisers.Classification_StochasticSearch
}

#Dictionary from transform to its optimiser for semantic segmentation
default_semantic_segmentation_optimiser = {
  "Pixel" : reetoolbox.semantic_segmentation_optimisers.Semantic_Segmentation_PGD,
  "Stain" : reetoolbox.semantic_segmentation_optimisers.Semantic_Segmentation_PGD,
  "Mean" : reetoolbox.semantic_segmentation_optimisers.Semantic_Segmentation_PGD,


  "Cut Out" : reetoolbox.semantic_segmentation_optimisers.Semantic_Segmentation_StochasticSearch,
  "Rotate" : reetoolbox.semantic_segmentation_optimisers.Semantic_Segmentation_StochasticSearch,
  "Crop" : reetoolbox.semantic_segmentation_optimisers.Semantic_Segmentation_StochasticSearch,
  "Blur" : reetoolbox.semantic_segmentation_optimisers.Semantic_Segmentation_StochasticSearch,
  "Zoom In" : reetoolbox.semantic_segmentation_optimisers.Semantic_Segmentation_StochasticSearch,
  "Zoom Out" : reetoolbox.semantic_segmentation_optimisers.Semantic_Segmentation_StochasticSearch,
  "HED Stain" : reetoolbox.semantic_segmentation_optimisers.Semantic_Segmentation_StochasticSearch,
  "Stochastic Stain" : reetoolbox.semantic_segmentation_optimisers.Semantic_Segmentation_StochasticSearch,
  "JPEG Compression" : reetoolbox.semantic_segmentation_optimisers.Semantic_Segmentation_StochasticSearch
}

#Dictionary from transform to its optimiser for HoVer-Net
default_hover_optimiser = {
  "Pixel" : reetoolbox.hover_optimisers.HoVer_PGD,
  "Stain" : reetoolbox.hover_optimisers.HoVer_PGD,
  "Mean" : reetoolbox.hover_optimisers.HoVer_PGD,


  "Cut Out" : reetoolbox.hover_optimisers.HoVer_StochasticSearch,
  "Rotate" : reetoolbox.hover_optimisers.HoVer_StochasticSearch,
  "Crop" : reetoolbox.hover_optimisers.HoVer_StochasticSearch,
  "Blur" : reetoolbox.hover_optimisers.HoVer_StochasticSearch,
  "Zoom In" : reetoolbox.hover_optimisers.HoVer_StochasticSearch,
  "Zoom Out" : reetoolbox.hover_optimisers.HoVer_StochasticSearch,
  "HED Stain" : reetoolbox.hover_optimisers.HoVer_StochasticSearch,
  "Stochastic Stain" : reetoolbox.hover_optimisers.HoVer_StochasticSearch,
  "JPEG Compression" : reetoolbox.hover_optimisers.HoVer_StochasticSearch,

}

default_optimiser_params = {
  "Pixel" : reetoolbox.constants.eval_pixel_optimiser_params,
  "Stain" : reetoolbox.constants.eval_stain_optimiser_params,
  "Mean" : reetoolbox.constants.eval_mean_optimiser_params,


  "Cut Out" : reetoolbox.constants.cutout_opt, #Needs to be updated
  "Rotate" : reetoolbox.constants.eval_rotate_optimiser_params,
  "Crop" : reetoolbox.constants.eval_crop_optimiser_params,
  "Blur" : reetoolbox.constants.eval_blur_optimiser_params,
  "Zoom In" : reetoolbox.constants.eval_zoom_in_optimiser_params,
  "Zoom Out" : reetoolbox.constants.eval_zoom_out_optimiser_params,
  "HED Stain" : reetoolbox.constants.eval_hed_optimiser_params,
  "Stochastic Stain" : reetoolbox.constants.eval_random_stain_optimiser_params,
  "JPEG Compression" : reetoolbox.constants.eval_jpeg_optimiser_params
}

default_transform_params = {
  "Pixel" : reetoolbox.constants.eval_pixel_transform_params,
  "Stain" : reetoolbox.constants.eval_stain_transform_params,
  "Mean" : reetoolbox.constants.eval_mean_transform_params,


  "Cut Out" : reetoolbox.constants.cutout_trans, #Needs to be updated
  "Rotate" : reetoolbox.constants.eval_rotate_transform_params,
  "Crop" : reetoolbox.constants.eval_crop_transform_params,
  "Blur" : reetoolbox.constants.eval_blur_transform_params,
  "Zoom In" : reetoolbox.constants.eval_zoom_in_transform_params,
  "Zoom Out" : reetoolbox.constants.eval_zoom_out_transform_params,
  "HED Stain" : reetoolbox.constants.eval_hed_transform_params,
  "Stochastic Stain" : reetoolbox.constants.eval_random_stain_transform_params,
  "JPEG Compression" : reetoolbox.constants.eval_jpeg_transform_params
}

#Need to specify what the last layer needs to be for these to work.
classification_loss = {
  "Default" : reetoolbox.classification_optimisers.untargeted_loss,
  "Untargeted" : reetoolbox.classification_optimisers.untargeted_loss,
  "Targeted" : reetoolbox.classification_optimisers.targeted_loss #How are targets selected again?
}

semantic_segmentation_loss = {
  "Default" : reetoolbox.semantic_segmentation_optimisers.untargeted_segmentation_CE_loss,
  "Default batch" : reetoolbox.semantic_segmentation_optimisers.multiple_untargeted_segmentation_CE_loss,
  "Untargeted CE" : reetoolbox.semantic_segmentation_optimisers.untargeted_segmentation_CE_loss
}

hover_net_loss = {
  "Default": reetoolbox.hover_optimisers.hover_untargeted_loss,
  "Default batch": reetoolbox.hover_optimisers.hover_untargeted_loss_by_batch,
}