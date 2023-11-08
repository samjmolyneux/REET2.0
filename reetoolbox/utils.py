import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch.nn.functional as F
from matplotlib import cm
import glob
import inspect
import logging
import os
import shutil
import cv2
from scipy import ndimage
from scipy.ndimage import measurements
from skimage import morphology as morph



def load_resnet(model_path, n_classes=2, hidden_size=512, pretrained=True, device="cuda:0"):
    model = models.resnet18(pretrained=pretrained)

    model.fc = nn.Sequential(
        nn.Linear(hidden_size, n_classes),
        nn.LogSoftmax(dim=1))

    model.load_state_dict(torch.load(model_path))

    model.to(device)
    model.eval()
    return model




#Makes a dataloader, more importantly it limits the dataloader to the first n values
def get_dataloader(dataset, n=100, batch_size=8, num_workers=2, shuffle=False):
    if n is None or n > len(dataset):
        n = len(dataset)
    X_y = dataset[:n]
    X = X_y[0]
    y = X_y[1]
    predict_data = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(predict_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def label_patch(patch, count):
    if count[0] >= 5:
        return patch, 1
    elif count[0] == 0:
        return patch, 0
    else:
        return patch, -1


def load_folds(folds):
    X = []
    y = []
    all_counts = []

    for images, counts in folds:
        for i, image in enumerate(images):
            cc = counts[i]
            image, label = label_patch(image, cc)

            if label == -1:
                continue

            im = Image.fromarray(np.uint8(image))
            X.append(np.transpose(np.array(im.resize((224, 224))), (2, 0, 1)))
            y.append(label)
            all_counts.append(cc)

    X = np.array(X)
    y = np.array(y)
    all_counts = np.array(all_counts)

    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    return X, y, all_counts


def load_pannuke(data_path, return_counts=False):
    F = np.load(data_path, allow_pickle=True)['F']

    Xtr, ytr, tr_counts = load_folds([F[1], F[2]])
    Xts, yts, ts_counts = load_folds([F[0]])

    if return_counts:
        return Xtr, ytr, Xts, yts, tr_counts, ts_counts
    return Xtr, ytr, Xts, yts


def plot_change(param_values, all_scores, xlabel="", ylabel="", title="", x_range=None, y_range=None, y_ticks=None):
    fig = plt.figure()
    plot = sns.lineplot(x=param_values, y=all_scores)
    plot.set_xlabel(xlabel)
    plot.set_ylabel(ylabel)
    plot.set_title(title)
    if x_range is not None:
        plot.set(xlim=x_range)
    if y_range is not None:
        plot.set(ylim=y_range)
    if y_ticks is not None:
        plot.set(yticks=y_ticks)
    return fig


def display_results(model, input, adv_input, classes):
    in_out = model(input.unsqueeze(0))
    adv_out = model(adv_input.unsqueeze(0))

    in_index = torch.argmax(in_out).item()
    adv_index = torch.argmax(adv_out).item()

    in_class = classes[in_index]
    adv_class = classes[adv_index]

    in_conf = round(torch.exp(in_out)[0][in_index].item() * 100, 1)
    adv_conf = round(torch.exp(adv_out)[0][adv_index].item() * 100, 1)

    perturbation = adv_input - input

    fig = plt.figure(figsize=[13, 9])

    input = input.permute(1,2,0).detach().cpu().numpy()
    adv_input = adv_input.permute(1,2,0).detach().cpu().numpy()
    perturbation = perturbation.permute(1,2,0).detach().cpu().numpy()

    plt.subplot(1, 3, 1)
    plt.imshow(input.astype(int))
    plt.title(in_class + ", " + str(in_conf) + "% confidence")
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])

    plt.subplot(1, 3, 2)
    rmse = np.sqrt(np.mean(np.power(perturbation, 2)))
    l_inf = np.max(np.abs(perturbation))
    #TAKE BREAK THEN READ THIS BIT

    min_val = np.min(np.abs(perturbation))
    max_val = np.max(np.abs(perturbation))

    if min_val < 1 and min_val != max_val:
        perturbation -= np.min(perturbation)
        scale_factor = 255/np.max(perturbation)
        perturbation *= scale_factor
    else:
        scale_factor = 1.0
        perturbation = np.abs(perturbation)

    plt.imshow(perturbation.astype(int))
    plt.title(f'Perturbation - scale factor: {scale_factor:.3f}')
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
    x_lab = f"L2: {rmse:.3f}, L infinity: {l_inf:.3f}"
    plt.xlabel(x_lab)

    plt.subplot(1, 3, 3)
    plt.imshow(adv_input.astype(int))
    plt.title(adv_class + ", " + str(adv_conf) + "% confidence")
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])

    return fig


def plot_hist(num_epochs, hist, title):
  plt.figure()
  plt.plot(range(num_epochs), hist)
  plt.title(title)
  plt.xlim([0, num_epochs])
  plt.ylim([0, 1])


# Differentiable compression utils from https://github.com/mlomnitz/DiffJPEG
# Used under the MIT license
"""
MIT License

Copyright (c) 2021 Michael R Lomnitz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x)) ** 3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality * 2
    return quality / 100.


# Standard libraries
import itertools
import numpy as np
# PyTorch
import torch
import torch.nn as nn

y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

y_table = nn.Parameter(torch.from_numpy(y_table))


class Utils:
    def __init__(self):
        c_table = np.empty((8, 8), dtype=np.float32)
        c_table.fill(99)
        c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                                    [24, 26, 56, 99], [47, 66, 99, 99]]).T
        self.c_table = nn.Parameter(torch.from_numpy(c_table)).to("cpu")


utils = Utils()


class rgb_to_ycbcr_jpeg(nn.Module):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(rgb_to_ycbcr_jpeg, self).__init__()
        matrix = np.array(
            [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0., 128., 128.]))
        #
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        #    result = torch.from_numpy(result)
        result.view(image.shape)
        return result


class chroma_subsampling(nn.Module):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """

    def __init__(self):
        super(chroma_subsampling, self).__init__()

    def forward(self, image):
        image_2 = image.permute(0, 3, 1, 2).clone()
        avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                                count_include_pad=False)
        cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
        cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class block_splitting(nn.Module):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output:
        patch(tensor):  batch x h*w/64 x h x w
    """

    def __init__(self):
        super(block_splitting, self).__init__()
        self.k = 8

    def forward(self, image):
        height, width = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)


class dct_8x8(nn.Module):
    """ Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """

    def __init__(self):
        super(dct_8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        #
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

    def forward(self, image):
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class y_quantize(nn.Module):
    """ JPEG Quantization for Y channel
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self, rounding, factor=1):
        super(y_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.y_table = y_table

    def forward(self, image):
        image = image.float() / (self.y_table * self.factor)
        image = self.rounding(image)
        return image


class c_quantize(nn.Module):
    """ JPEG Quantization for CrCb channels
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self, rounding, factor=1):
        super(c_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.c_table = utils.c_table

    def forward(self, image):
        image = image.float() / (self.c_table * self.factor)
        # image = image.float() / (self.c_table**2+eps * self.factor)
        image = self.rounding(image)
        return image


class compress_jpeg(nn.Module):
    """ Full JPEG compression algortihm
    Input:
        imgs(tensor): batch x 3 x height x width
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
    """

    def __init__(self, rounding=torch.round, factor=1):
        super(compress_jpeg, self).__init__()
        self.l1 = nn.Sequential(
            rgb_to_ycbcr_jpeg(),
            chroma_subsampling()
        )
        self.l2 = nn.Sequential(
            block_splitting(),
            dct_8x8()
        )
        self.c_quantize = c_quantize(rounding=rounding, factor=factor)
        self.y_quantize = y_quantize(rounding=rounding, factor=factor)

    def forward(self, image):
        y, cb, cr = self.l1(image * 255)
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp)
            else:
                comp = self.y_quantize(comp)

            components[k] = comp

        return components['y'], components['cb'], components['cr']


class y_dequantize(nn.Module):
    """ Dequantize Y channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width

    """

    def __init__(self, factor=1):
        super(y_dequantize, self).__init__()
        self.y_table = y_table
        self.factor = factor

    def forward(self, image):
        return image * (self.y_table * self.factor)


class c_dequantize(nn.Module):
    """ Dequantize CbCr channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width

    """

    def __init__(self, factor=1):
        super(c_dequantize, self).__init__()
        self.factor = factor
        self.c_table = utils.c_table

    def forward(self, image):
        return image * (self.c_table * self.factor)


class idct_8x8(nn.Module):
    """ Inverse discrete Cosine Transformation
    Input:
        dcp(tensor): batch x height x width
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self):
        super(idct_8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                (2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, image):
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class block_merging(nn.Module):
    """ Merge pathces into image
    Inputs:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self):
        super(block_merging, self).__init__()

    def forward(self, patches, height, width):
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class chroma_upsampling(nn.Module):
    """ Upsample chroma layers
    Input:
        y(tensor): y channel image
        cb(tensor): cb channel
        cr(tensor): cr channel
    Ouput:
        image(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(chroma_upsampling, self).__init__()

    def forward(self, y, cb, cr):
        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x

        cb = repeat(cb)
        cr = repeat(cr)

        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class ycbcr_to_rgb_jpeg(nn.Module):
    """ Converts YCbCr image to RGB JPEG
    Input:
        image(tensor): batch x height x width x 3
    Outpput:
        result(tensor): batch x 3 x height x width
    """

    def __init__(self):
        super(ycbcr_to_rgb_jpeg, self).__init__()

        matrix = np.array(
            [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128., -128.]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        # result = torch.from_numpy(result)
        result.view(image.shape)
        return result.permute(0, 3, 1, 2)


class decompress_jpeg(nn.Module):
    """ Full JPEG decompression algortihm
    Input:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        image(tensor): batch x 3 x height x width
    """

    def __init__(self, height, width, rounding=torch.round, factor=1):
        super(decompress_jpeg, self).__init__()
        self.c_dequantize = c_dequantize(factor=factor)
        self.y_dequantize = y_dequantize(factor=factor)
        self.idct = idct_8x8()
        self.merging = block_merging()
        self.chroma = chroma_upsampling()
        self.colors = ycbcr_to_rgb_jpeg()

        self.height, self.width = height, width

    def forward(self, y, cb, cr):
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k])
                height, width = int(self.height / 2), int(self.width / 2)
            else:
                comp = self.y_dequantize(components[k])
                height, width = self.height, self.width
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
            #
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)

        image = torch.min(255 * torch.ones_like(image),
                          torch.max(torch.zeros_like(image), image))
        return image / 255


####THE FOLLOWING FUNCTIONS ARE TAKEN DIRECTLY FROM THE HOVERNET GITHUB
####
def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)


####
def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


####
def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.
    Args:
        x: input array
        crop_shape: dimensions of cropped array
    
    Returns:
        x: cropped array
    
    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[-2] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[-1] - crop_shape[1]) * 0.5)
        x = x[..., h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x



####
def get_inst_centroid(inst_map):
    """Get instance centroids given an input instance map.
    Args:
        inst_map: input instance map
    
    Returns:
        array of centroids
    
    """
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)


####
def center_pad_to_shape(img, size, cval=255):
    """Pad input image."""
    # rounding down, add 1
    pad_h = size[0] - img.shape[0]
    pad_w = size[1] - img.shape[1]
    pad_h = (pad_h // 2, pad_h - pad_h // 2)
    pad_w = (pad_w // 2, pad_w - pad_w // 2)
    if len(img.shape) == 2:
        pad_shape = (pad_h, pad_w)
    else:
        pad_shape = (pad_h, pad_w, (0, 0))
    img = np.pad(img, pad_shape, "constant", constant_values=cval)
    return img


####
def color_deconvolution(rgb, stain_mat):
    """Apply colour deconvolution."""
    log255 = np.log(255)  # to base 10, not base e
    rgb_float = rgb.astype(np.float64)
    log_rgb = -((255.0 * np.log((rgb_float + 1) / 255.0)) / log255)
    output = np.exp(-(log_rgb @ stain_mat - 255.0) * log255 / 255.0)
    output[output > 255] = 255
    output = np.floor(output + 0.5).astype("uint8")
    return output



def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.
    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided. 
    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel. 
    
    Returns:
        out: output array with instances removed under min_size
    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

def crop_center_2_3(x, crop_shape):
    orig_shape = x.shape
    h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
    w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
    x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


####
def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image.
    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
        
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


####
def crop_to_shape(x, y, data_format="NCHW"):
    """Centre crop x so that x has shape of y. y dims must be smaller than x dims.
    Args:
        x: input array
        y: array with desired shape.
    """
    assert (
        y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    ), "Ensure that y dimensions are smaller than x dimensions!"

    x_shape = x.size()
    y_shape = y.size()
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])
    return crop_op(x, crop_shape, data_format)


####
def xentropy_loss(true, pred, reduction="mean"):
    """Cross entropy loss. Assumes NHWC!
    Args:
        pred: prediction array
        true: ground truth array
    
    Returns:
        cross entropy loss
    """
    epsilon = 10e-8
    # scale preds so that the class probs of each sample sum to 1
    pred = pred / torch.sum(pred, -1, keepdim=True)
    # manual computation of crossentropy
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
    loss = loss.mean() if reduction == "mean" else loss.sum()
    return loss


####
def dice_loss(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
    inse = torch.sum(pred * true, (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss


####
def mse_loss(true, pred):
    """Calculate mean squared error loss.
    Args:
        true: ground truth of combined horizontal
              and vertical maps
        pred: prediction of combined horizontal
              and vertical maps 
    
    Returns:
        loss: mean squared error
    """
    loss = pred - true
    loss = (loss * loss).mean()
    return loss


####
def msge_loss(true, pred, focus):
    """Calculate the mean squared error of the gradients of 
    horizontal and vertical map predictions. Assumes 
    channel 0 is Vertical and channel 1 is Horizontal.
    Args:
        true:  ground truth of combined horizontal
               and vertical maps
        pred:  prediction of combined horizontal
               and vertical maps 
        focus: area where to apply loss (we only calculate
                the loss within the nuclei)
    
    Returns:
        loss:  mean squared error of gradients
    """

    def get_sobel_kernel(size):
        """Get sobel kernel with a given size."""
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    ####
    def get_gradient_hv(hv):
        """For calculating gradient."""
        kernel_h, kernel_v = get_sobel_kernel(5)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    focus = (focus[..., None]).float()  # assume input NHW
    focus = torch.cat([focus, focus], axis=-1)
    true_grad = get_gradient_hv(true)
    pred_grad = get_gradient_hv(pred)
    loss = pred_grad - true_grad
    loss = focus * (loss * loss)
    # artificial reduce_mean with focused region
    loss = loss.sum() / (focus.sum() + 1.0e-8)
    return loss



####
def gen_instance_hv_map(ann, crop_shape):
    """Input annotation must be of original shape.
    
    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.
    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.
    """
    orig_ann = ann.copy()  # instance ID map
    fixed_ann = fix_mirror_padding(orig_ann)
    # re-cropping with fixed instance id map

    crop_ann = cropping_center(fixed_ann, crop_shape)
    # TODO: deal with 1 label warning
    crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(fixed_ann == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        # expand the box by 2px
        # Because we first pad the ann at line 207, the bboxes
        # will remain valid after expansion
        inst_box[0] -= 2
        inst_box[2] -= 2
        inst_box[1] += 2
        inst_box[3] += 2

        inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        # instance center of mass, rounded to nearest pixel
        inst_com = list(measurements.center_of_mass(inst_map))

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.dstack([x_map, y_map])
    return hv_map


#### #Added a way to work with batches
def gen_targets(ann, crop_shape, batch = False, **kwargs):
   
    """Generate the targets for the network."""
    if batch:
        hv_map = np.zeros((ann.shape[0], ann.shape[1], ann.shape[2], 2))
        for i in range (ann.shape[0]):
            hv_map[i] = gen_instance_hv_map(ann[i], crop_shape)    
    else:
        hv_map = gen_instance_hv_map(ann, crop_shape)
    np_map = ann.copy()
    np_map[np_map > 0] = 1

    hv_map = cropping_center(hv_map, crop_shape, batch)
    np_map = cropping_center(np_map, crop_shape, batch)

    target_dict = {
        "hv_map": hv_map,
        "np_map": np_map,
    }

    return target_dict


def gen_targets_batch(ann, crop_shape):
   
    hv_map = np.zeros((ann.shape[0], ann.shape[1], ann.shape[2], 2))
    for i in range (ann.shape[0]):
        hv_map[i] = gen_instance_hv_map(ann[i], crop_shape)    

    np_map = ann.copy()
    np_map[np_map > 0] = 1

    hv_map = crop_center_2_3(hv_map, crop_shape)
    np_map = crop_center_2_3(np_map, crop_shape)

    target_dict = {
        "hv_map": hv_map,
        "np_map": np_map,
    }

    return target_dict


####
def fix_mirror_padding(ann):
    """Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.).
    
    """
    current_max_id = np.amax(ann)
    inst_list = list(np.unique(ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(ann == inst_id, np.uint8)
        remapped_ids = measurements.label(inst_map)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(ann)
    return ann


################# POST PROCESSING #################################

from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)

from skimage.segmentation import watershed


def __proc_np_hv(pred):
    """Process Nuclei Prediction with XY Coordinate Map.
    Args:
        pred: prediction output, assuming 
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map
    """
    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    # processing
    
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)
    # plt.imshow(blb)
    # plt.show()
    # plt.imshow(h_dir_raw)
    # plt.show()
    # plt.imshow(v_dir_raw)
    # plt.show()

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    
    save1 = blb.copy()


    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    
    #SO I NEED TO COPY THIS BIT IN ORDER TO MAKE IT A FAIR TEST, USE THIS ON THE INSTANCE MAP
    #AND THEN USE IT TO REMOVE INSTANCES WHO HAVE ENIRETY OF INSTANCE LARGER THAN 0.4.
    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)




    marker = blb - overall
    # plt.imshow(marker)
    # plt.show()
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)


    proced_pred = watershed(dist, markers=marker, mask=blb)
    # plt.imshow(proced_pred)
    # plt.show()
    return proced_pred


####
#ADD A SECOND VERSION OF PROCESS THAT INCLUDES NUCLEI THAT DO NOT HAVE THERE CENTERS IN THE SQUARE
def process(pred_map, nr_types=None, return_centroids=False):
    """Post processing script for image tiles.
    Args:
        pred_map: commbined output of tp, np and hv branches, in the same order
        nr_types: number of types considered at output of nc branch
        overlaid_img: img to overlay the predicted instances upon, `None` means no
        type_colour (dict) : `None` to use random, else overlay instances of a type to colour in the dict
        output_dtype: data type of output
    
    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction 
    """
    #pred_map is a np arary of shape (1000,1000,4), tp, np, h and v.


    if nr_types is not None:
        pred_type = pred_map[..., :1]
        pred_inst = pred_map[..., 1:]
        pred_type = pred_type.astype(np.int32)
    else:
        pred_inst = pred_map

    pred_inst = np.squeeze(pred_inst)
    pred_inst = __proc_np_hv(pred_inst)


    inst_info_dict = None
    if return_centroids or nr_types is not None:
        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            # TODO: chane format of bbox output
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }

    if nr_types is not None:
        #### * Get class of each instance id, stored at index id-1
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                inst_map_crop == inst_id
            )  # TODO: duplicated operation, may be expensive
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

    # print('here')
    # ! WARNING: ID MAY NOT BE CONTIGUOUS
    # inst_id in the dict maps to the same value in the `pred_inst`

    #inst_info_dict has a key for each instance.
    #These are the dict_keys(['bbox', 'centroid', 'contour', 'type_prob', 'type'])
    return pred_inst, inst_info_dict

from collections import OrderedDict

#generates a map of e.g (1000,1000,4) where it is tp, np hv or something like that
def gen_inference_pred_map(imgs, model, device):
    
    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs)  
        imgs = imgs.to(device).contiguous().type(torch.float32)

    #check that it is in nchw    
    ####
    model.eval()  # infer mode

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(imgs)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:] 
        if "tp" in pred_dict:
            type_map = F.softmax(pred_dict["tp"], dim=-1) #Pretty sure we dont need this
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map
        pred_output = torch.cat(list(pred_dict.values()), -1).squeeze()

    # * Its up to user to define the protocol to process the raw output per step!
    return pred_output.detach().cpu().numpy()

#!VERY IMPORTANT!: We have to perform both the process on the truth maps so that 
#they both remove same nuclei (e.g. such as those that do not have there centers in the map)
def gen_hover_truth_map_from_dict(truth_dict):
    
    orig_shape = truth_dict["tp"].shape
    map_shape = (orig_shape[1], orig_shape[2], 4)
    truth_map = np.zeros(map_shape)
    
    truth_map[..., 0] = torch.argmax(truth_dict["tp"].detach().cpu().squeeze(), -1).numpy()
    truth_map[..., 1] = torch.argmax(truth_dict["np"].detach().cpu().squeeze(), -1).numpy()
    truth_map[..., 2:4] = truth_dict["hv"].detach().cpu().squeeze().numpy()

    return truth_map

#!VERY IMPORTANT!: We have to perform both the process on the truth maps so that 
#they both remove same nuclei (e.g. such as those that do not have there centers in the map)
def gen_hover_truth_map_from_ann(ann, map_shape):
    inst_map = ann[..., 0]
    type_map = ann[..., 1]

    #CHECK THAT GEN TARGETS IS DEFINITELY GETTING INST_MAP IN A GOOD ENOUGH SHAPE
    truth_dict = gen_targets(inst_map, map_shape, batch=False)

    truth_map = np.zeros((map_shape[0], map_shape[1], 4))

    truth_map[..., 0] = cropping_center(type_map, map_shape)
    truth_map[..., 1] = truth_dict["np_map"]
    truth_map[..., 2:4] = truth_dict["hv_map"]

    return truth_map #(80,80,4)tp, np, hv


def inst_to_semantic_segmentation_mask(inst_map, inst_info_dict, background_index=0):

    def map_instance_id_to_type(instance_id):
      try:
          return inst_info_dict[instance_id]["type"]
      except KeyError:
        # handle the case where the instance ID is not in the inst_info dictionary
          return background_index  

    # create a vectorized version of the mapping function
    vectorized_map = np.vectorize(map_instance_id_to_type)

    # apply the mapping function to the instance map to get the semantic segmentation map
    semantic_map = vectorized_map(inst_map).astype("int32") #Needs to be int32 to be same as the inst_maps

    return semantic_map

    # def count_instance_pixels(instance_map):
    #     instance_ids = np.unique(instance_map)
    #     counts = {}
    #     for instance_id in instance_ids:
    #         if instance_id == 0:  # ignore background
    #             continue
    #         counts[instance_id] = np.count_nonzero(instance_map == instance_id)
    #     return counts

    # print(count_instance_pixels(inst_map))

    # plt.imshow(inst_map)
    # plt.show()

    # plt.imshow(semantic_map)
    # plt.show()

    # type_map = np.zeros_like(inst_map)
    # for instance_id in np.unique(inst_map):
    #     if instance_id == 0:  # ignore background
    #         continue
    #     instance_type = inst_info_dict[instance_id]["type"]
    #     type_map[inst_map == instance_id] = instance_type
    # return type_map


from reetoolbox.hover_metrics import get_dice_1

#Takes masks as torch.tensor
def calculate_semantic_seg_metrics(results_dict, pred_mask, truth_mask, num_classes, background_index=0, device="cuda"):

    # pixel_acc = MulticlassAccuracy(num_classes, average='micro').to(device)
    # pixel_acc_no_background = MulticlassAccuracy(num_classes, average='micro', ignore_index=background_index).to(device)

    #dice_1 works for both
    #get_dice_1(truth_mask, pred_mask)
    #print(f"dice with semantic: {get_dice_1(truth_mask, pred_mask)}")
    


    
    #Check if the masks are in batches.
    #WOULD BE BETTER IF WE VECTORISE HERE SOMEHOW
    # if pred_masks.shape == 4:
    #     for i in range(pred_masks.shape[0]): 
    #         prediction_accuracy = pixel_acc(pred_masks[i], truth_masks[i])
    #         results_dict["pixel_accuracies"].append(prediction_accuracy.item())

    #         prediction_accuracy_no_background = pixel_acc_no_background(pred_masks[i], truth_masks[i])
    #         results_dict["pixel_accuracies_nb"].append(prediction_accuracy_no_background.item())
    # else:
    #         prediction_accuracy = pixel_acc(pred_masks, truth_masks)
    #         results_dict["pixel_accuracies"].append(prediction_accuracy.item())

    #         prediction_accuracy_no_background = pixel_acc_no_background(pred_masks, truth_masks)
    #         results_dict["pixel_accuracies_nb"].append(prediction_accuracy_no_background.item())


    #SHOULD ADD SOME MORE METRICS HERE.
    return

    
import cv2
#Returns the truth map for the inst_map entered. 
#WE COULD TAKE THE INST MAP, THE TRANSFORM AND OUTPUT SHAPE, APPLY THE TRANSFORM TO THE INST MAP (if we can actually do this), CALCULATE THE CENTROIDS OF THE TRANSFORMED
#INST MAP TRANS INST MAP. THEN TAKE OUR CROPPED CENTER AND GET OUR COORIDINATE BOUNDARIES FOR THE CROPPED CENTER. ANY CENTROIDS OUTSIDE OF THE THIS ARE REMOVED  
def gen_truth_inst_info(truth_inst_map, truth_type_map):
    #FIX MIRROR PADDING, THEN PERFORM SECOND HALF OF PROCESS TO GET THE INST INFO DICT

    inst_id_list = np.unique(truth_inst_map)[1:]  # exlcude background
    inst_info_dict = {}
    for inst_id in inst_id_list:
        inst_map = truth_inst_map == inst_id
        # TODO: chane format of bbox output
        rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
        inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
        inst_map = inst_map[
            inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
        ]
        inst_map = inst_map.astype(np.uint8)
        inst_moment = cv2.moments(inst_map)
        inst_contour = cv2.findContours(
            inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # * opencv protocol format may break
        inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
        # < 3 points dont make a contour, so skip, likely artifact too
        # as the contours obtained via approximation => too small or sthg
        #WE SHOULD ALSO TRY THIS WITH REMOVING SMALL OBJECTS FROM THE MASKS OF MODELS.
        # if inst_contour.shape[0] < 3:
        # #     print("3")
        #       continue
        #To make it a fair test when comparing to other models. 
        if len(inst_contour.shape) != 2:
            inst_info_dict[inst_id] = {  
            "bbox": inst_bbox,
            "centroid": [cmin, rmin],
            "contour": inst_contour,
            "type_prob": None,
            "type": None,
            }
            continue # ! check for trickery shape
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid = np.array(inst_centroid)
        inst_contour[:, 0] += inst_bbox[0][1]  # X
        inst_contour[:, 1] += inst_bbox[0][0]  # Y
        inst_centroid[0] += inst_bbox[0][1]  # X
        inst_centroid[1] += inst_bbox[0][0]  # Y
        inst_info_dict[inst_id] = {  # inst_id should start at 1
            "bbox": inst_bbox,
            "centroid": inst_centroid,
            "contour": inst_contour,
            "type_prob": None,
            "type": None,
        }
    
    for inst_id in list(inst_info_dict.keys()):
        rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
        inst_map_crop = truth_inst_map[rmin:rmax, cmin:cmax]
        inst_type_crop = truth_type_map[rmin:rmax, cmin:cmax]
        inst_map_crop = (
            inst_map_crop == inst_id
        )  # TODO: duplicated operation, may be expensive
        inst_type = inst_type_crop[inst_map_crop]
        type_list, type_pixels = np.unique(inst_type, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]
        if inst_type == 0:  # ! pick the 2nd most dominant if exist
            if len(type_list) > 1:
                inst_type = type_list[1][0]
        type_dict = {v[0]: v[1] for v in type_list}
        type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
        inst_info_dict[inst_id]["type"] = int(inst_type)
        inst_info_dict[inst_id]["type_prob"] = float(type_prob)

    return inst_info_dict



















