from reetoolbox.transforms import *

# TODO make it easy to change all common parameters at once (e.g., change input_range from (0, 255) to (0, 1) everywhere
# Or make them specifiable in only one place

# Default evaluation parameters
eval_samples = 5
eval_steps = 5

input_range = (0, 255) #DO NOT CHANGE INPUT RANGE

#Make these more clear and rigorous.
cutout_opt = {
    "samples": eval_samples,

    "weight_ranges": {
      'corner_y' : (0,224),
      'corner_x' : (0,224)
    },
    "input_range": input_range
}
cutout_trans = {
    'height' :224,
    'width' : 224
}

eval_rotate_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "angle": (0, 360)
    },
    "input_range": input_range
}
eval_rotate_transform_params = {
}

eval_zoom_out_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "scale": (0.5, 0.95)
    },
    "input_range": input_range
}
eval_zoom_out_transform_params = {
}

sigma = 0.008
eval_hed_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "alpha": (1 - sigma, 1 + sigma),
        "beta": (-sigma, sigma)
    },
    "input_range": input_range
}
eval_hed_transform_params = {
}

eval_mean_optimiser_params = {
    "epsilon": 1,
    "steps": eval_steps,
    "constraint": "l_inf",
    "C": 7,
    "input_range": input_range
}
eval_mean_transform_params = {
    "input_range": input_range,
    "noise_range": (-0.2, 0.2)
}

eval_zoom_in_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "scale": (1.1, 2)
    },
    "input_range": input_range
}
eval_zoom_in_transform_params = {
}

eval_crop_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "top": (0, 100),
        "left": (0, 100),
        "height": (150, 200),
        "width": (150, 200)
    },
    "input_range": input_range
}
eval_crop_transform_params = {
}

eval_blur_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "corner_x": (100, 400),
        "corner_y": (100, 400),
        "height": (100, 300),
        "width": (100, 300),
        "kernel_size": (1, 10),
        "sigma": (3, 20)
    },
    "input_range": input_range
}
eval_blur_transform_params = {
}

eval_pixel_optimiser_params = {
    "epsilon": 0.1,
    "steps": eval_steps,
    "constraint": "l_inf",
    "C": 7,
    "input_range": input_range
}
eval_pixel_transform_params = {
    "input_range": input_range,
    "noise_range": (-0.2, 0.2)
}

eval_stain_optimiser_params = {
    "epsilon": 0.002,
    "steps": eval_steps,
    "constraint": "l_inf",
    "C": 7,
    "input_range": input_range
}
eval_stain_transform_params = {
    "input_range": input_range
}

eval_random_stain_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "weights": (-0.25, 0.25)
    },
    "input_range": input_range
}
eval_random_stain_transform_params = {
    "input_range": input_range
}

eval_identity_optimiser_params = {
    "epsilon": 0,
    "steps": 0,
    "constraint": "l_inf",
    "C": 0.2,
    "input_range": input_range
}
eval_identity_transform_params = {
    "input_range": input_range
}

eval_jpeg_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "quality": (5, 20)
    },
    "input_range": input_range
}
eval_jpeg_transform_params = {
}

# Default training parameters
train_samples = 5
train_steps = 2

train_rotate_optimiser_params = {
    "samples": train_samples,
    "weight_ranges": {
        "angle": (0, 360)
    },
    "input_range": input_range
}
train_rotate_transform_params = {
}

train_zoom_out_optimiser_params = {
    "samples": train_samples,
    "weight_ranges": {
        "scale": (0.5, 0.95)
    },
    "input_range": input_range
}
train_zoom_out_transform_params = {
}

sigma = 0.2
train_hed_optimiser_params = {
    "samples": train_samples,
    "weight_ranges": {
        "alpha": (1 - sigma, 1 + sigma),
        "beta": (-sigma, sigma)
    },
    "input_range": input_range
}
train_hed_transform_params = {
}

train_mean_optimiser_params = {
    "epsilon": 5,
    "steps": train_steps,
    "constraint": "l2",
    "C": 50,
    "input_range": input_range
}
train_mean_transform_params = {
    "input_range": input_range,
    "noise_range": (-0.2, 0.2)
}

train_zoom_in_optimiser_params = {
    "samples": train_samples,
    "weight_ranges": {
        "scale": (1.1, 2)
    },
    "input_range": input_range
}
train_zoom_in_transform_params = {
}

train_crop_optimiser_params = {
    "samples": train_samples,
    "weight_ranges": {
        "top": (0, 100),
        "left": (0, 100),
        "height": (150, 200),
        "width": (150, 200)
    },
    "input_range": input_range
}
train_crop_transform_params = {
}

train_blur_optimiser_params = {
    "samples": train_samples,
    "weight_ranges": {
        "corner_x": (0, 150),
        "corner_y": (0, 150),
        "height": (100, 150),
        "width": (100, 150),
        "kernel_size": (1, 10),
        "sigma": (3, 20)
    },
    "input_range": input_range
}
train_blur_transform_params = {
}

train_pixel_optimiser_params = {
    "epsilon": 0.1,
    "steps": train_steps,
    "constraint": "l2",
    "C": 2,
    "input_range": input_range
}
train_pixel_transform_params = {
    "input_range": input_range,
    "noise_range": (-0.2, 0.2)
}

train_stain_optimiser_params = {
    "epsilon": 0.05,
    "steps": train_steps,
    "constraint": "l2",
    "C": 0.5,
    "input_range": input_range
}
train_stain_transform_params = {
    "input_range": input_range
}

train_random_stain_optimiser_params = {
    "samples": train_samples,
    "weight_ranges": {
        "weights": (-0.25, 0.25)
    },
    "input_range": input_range
}
train_random_stain_transform_params = {
    "input_range": input_range
}

train_identity_optimiser_params = {
    "epsilon": 0,
    "steps": 0,
    "constraint": "l2",
    "C": 0.2,
    "input_range": input_range
}
train_identity_transform_params = {
    "input_range": input_range
}

train_jpeg_optimiser_params = {
    "samples": train_samples,
    "weight_ranges": {
        "quality": (5, 20)
    },
    "input_range": input_range
}
train_jpeg_transform_params = {
}

pixel_adv_opt_params = {
    "Transform": PixelTransform,
    "hyperparameters": train_pixel_optimiser_params,
    "transform_hyperparameters": train_pixel_transform_params
}
stain_adv_opt_params = {
    "Transform": StainTransform,
    "hyperparameters": train_stain_optimiser_params,
    "transform_hyperparameters": train_stain_transform_params
}
rotate_adv_opt_params = {
    "Transform": RotateTransform,
    "hyperparameters": train_rotate_optimiser_params,
    "transform_hyperparameters": train_rotate_transform_params
}
zoom_out_adv_opt_params = {
    "Transform": ZoomOutTransform,
    "hyperparameters": train_zoom_out_optimiser_params,
    "transform_hyperparameters": train_zoom_out_transform_params
}
hed_adv_opt_params = {
    "Transform": HEDTransform,
    "hyperparameters": train_hed_optimiser_params,
    "transform_hyperparameters": train_hed_transform_params
}
mean_adv_opt_params = {
    "Transform": MeanTransform,
    "hyperparameters": train_mean_optimiser_params,
    "transform_hyperparameters": train_mean_transform_params
}
zoom_in_adv_opt_params = {
    "Transform": ZoomInTransform,
    "hyperparameters": train_zoom_in_optimiser_params,
    "transform_hyperparameters": train_zoom_in_transform_params
}
crop_adv_opt_params = {
    "Transform": CropTransform,
    "hyperparameters": train_crop_optimiser_params,
    "transform_hyperparameters": train_crop_transform_params
}
blur_adv_opt_params = {
    "Transform": BlurTransform,
    "hyperparameters": train_blur_optimiser_params,
    "transform_hyperparameters": train_blur_transform_params
}
jpeg_adv_opt_params = {
    "Transform": JPEGTransform,
    "hyperparameters": train_jpeg_optimiser_params,
    "transform_hyperparameters": train_jpeg_transform_params
}
