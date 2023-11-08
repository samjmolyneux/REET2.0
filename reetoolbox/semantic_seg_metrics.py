import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def calculate_semantic_seg_metrics(pred_type_mask, true_type_mask, results_dict, num_classes, background_index=0):
  
    nuclear_pixel_tp_fp_fn(pred_type_mask, true_type_mask, results_dict, background_index)
    type_pixel_tp_fp_fn(pred_type_mask, true_type_mask, num_classes, results_dict)
    return





# Calculate the IoU score
def nuclear_pixel_tp_fp_fn(pred, true, results_dict, background_index = 0):

    binary_pred_mask = np.zeros_like(pred)
    binary_true_mask = np.zeros_like(true)

    binary_pred_mask[pred != background_index] = 1
    binary_true_mask[true != background_index] = 1

    tp = np.sum((binary_true_mask == 1)&(binary_pred_mask == 1)).astype("int64")
    fp = np.sum((binary_true_mask == 0)&(binary_pred_mask == 1)).astype("int64")
    fn = np.sum((binary_true_mask == 1)&(binary_pred_mask == 0)).astype("int64")
    
    results_dict["nucleus_pixel_tp"] += tp
    results_dict["nucleus_pixel_fp"] += fp
    results_dict["nucleus_pixel_fn"] += fn
  
    return 



def type_pixel_tp_fp_fn(pred, true, num_classes, results_dict):

    for i in range(num_classes):
        tp = np.sum((true == i)&(pred == i))
        fp = np.sum((true != i)&(pred == i))
        fn = np.sum((true == i)&(pred != i))

        results_dict["type_pixel_tp"][i] += tp
        results_dict["type_pixel_fp"][i] += fp
        results_dict["type_pixel_fn"][i] += fn
    

    # acc = np.sum(pred == true)/(pred.shape[0]*pred.shape[1])
    # print(f"num: {np.sum(pred == true)}")
    # print(f"denom: {}")
    # print(f"aacuracy here is: {acc}")
    results_dict["all_type_pixel_tp_tn"] += np.sum(pred == true)
    results_dict["all_type_pixel_tp_tn_fp_fn"] += pred.shape[0]*pred.shape[1]
    
    
    return 



