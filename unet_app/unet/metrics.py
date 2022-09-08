import torch

def accuracy(y_pred, y_true, weight=None, axes=[1,2,3]):
    """
    Computes the accuracy along the given axes and then 
    takes the mean across the remaining axes. 
    """
    
    if weight is not None:
        y_pred = y_pred * weight
        y_true = y_true * weight
        
    # Accuracy computation
    if weight is not None:
        correct = torch.sum(y_true == y_pred, axis=axes) - torch.sum(1 - weight, axis=axes)
        counts = torch.sum(weight, axis=axes)
    else:
        correct = torch.sum(y_true == y_pred, axis=axes)
        counts = torch.prod(torch.take(torch.tensor(y_true.shape), torch.tensor(axes)))

    accuracy = correct / counts
        
    return accuracy

def crossentropy_loss(y_pred, y_true, weight=None, axes=[2,3]):
    """
    Computes the crossentropy loss along the given axes and then 
    takes the mean across the remaining axes. 
    """
    
    epsilon = 1e-12
    
    # Crossentropy computation
    if weight is not None:
        ce = weight * y_true * torch.log(y_pred + epsilon)
        counts = torch.sum(weight, axis=axes)
    else:
        ce = y_true * torch.log(y_pred + epsilon)
        counts = torch.prod(torch.take(torch.tensor(y_true.shape), torch.tensor(axes)))

    ce = - torch.sum(ce, axis=axes) / counts
        
    return torch.mean(ce)

def dice(y_pred, y_true, weight=None, axes=[2,3]):
    """
    Computes the dice score along the given axes and then 
    takes the mean across the remaining axes. 
    """
    
    epsilon = 1e-12
    
    if weight is not None:
        y_pred = y_pred * weight
        y_true = y_true * weight

    # Dice computation
    num = 2 * torch.sum(y_pred * y_true, axis=axes)
    den = torch.sum(y_pred + y_true, axis=axes)
    dice_score = (num + epsilon) / (den + epsilon)

    return torch.mean(dice_score)

def dice_loss(y_pred, y_true, weight=None, axes=[2,3]):
    """
    Computes the dice loss.
    """
    
    return 1 - dice(y_pred, y_true, weight, axes)

def iou(y_pred, y_true, weight=None, axes=[2,3]):
    """
    Computes the intersection over union (Jaccard index) along the given axes and then 
    takes the mean across the remaining axes. 
    """
    
    dice_score = dice(y_pred, y_true, weight, axes)
    iou_score = dice_score / (2.0 - dice_score)
    return iou_score

def iou_loss(y_pred, y_true, weight=None, axes=[2,3]):
    """
    Computes the intersection over union (Jaccard index) loss.
    """
    
    return 1 - iou(y_pred, y_true, weight, axes)


def mcc(y_pred, y_true, weight=None, axes=[2,3]):
    """
    Computes the Mathews correlation coefficient (mcc) along the given axes and then 
    takes the mean across the remaining axes. 
    """
    
    epsilon = 1e-12
    
    # Compute confusion matrix percentages
    tp = true_positives(y_pred, y_true, weight, axes)
    tn = true_negatives(y_pred, y_true, weight, axes)
    fp = false_positives(y_pred, y_true, weight, axes)
    fn = false_negatives(y_pred, y_true, weight, axes)

    # MCC computation
    num = (tp * tn) - (fp * fn)
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**0.5
    mcc_score = (num + epsilon) / (den + epsilon)

    return torch.mean(mcc_score)

def mcc_loss(y_pred, y_true, weight=None, axes=[2,3]):
    """
    Computes the Mathews correlation coefficient (mcc) loss.
    """
    
    return 1 - mcc(y_pred, y_true, weight, axes)

def true_positives(y_pred, y_true, weight=None, axes=[2,3]):
    """
    Computes percentage of true positives along the given axes.
    """
    
    tp = y_true * y_pred
    
    if weight is not None:
        tp = weight * tp
        counts = torch.sum(weight, axis=axes)
    else:
        counts = torch.prod(torch.take(torch.tensor(y_true.shape), torch.tensor(axes)))
        
    tp_per = torch.sum(tp, axis=axes) / counts
    return tp_per

def true_negatives(y_pred, y_true, weight=None, axes=[2,3]):
    """
    Computes percentage of true negatives along the given axes.
    """
    
    tn = (1 - y_pred) * (1 - y_true)
    
    if weight is not None:
        tn = weight * tn
        counts = torch.sum(weight, axis=axes)
    else:
        counts = torch.prod(torch.take(torch.tensor(y_true.shape), torch.tensor(axes)))
    
    tn_per = torch.sum(tn, axis=axes) / counts
    return tn_per

def false_positives(y_pred, y_true, weight=None, axes=[2,3]):
    """
    Computes percentage of false positives along the given axes.
    """
    
    fp = (1 - y_true) * y_pred
    
    if weight is not None:
        fp = weight * fp
        counts = torch.sum(weight, axis=axes)
    else:
        counts = torch.prod(torch.take(torch.tensor(y_true.shape), torch.tensor(axes)))
        
    fp_per = torch.sum(fp, axis=axes) / counts
    return fp_per

def false_negatives(y_pred, y_true, weight=None, axes=[2,3]):
    """
    Computes percentage of false negatives along the given axes.
    """
    
    fn = (1 - y_pred) * y_true
    
    if weight is not None:
        fn = weight * fn
        counts = torch.sum(weight, axis=axes)
    else:
        counts = torch.prod(torch.take(torch.tensor(y_true.shape), torch.tensor(axes)))
        
    fn_per = torch.sum(fn, axis=axes) / counts
    return fn_per

def dice_ce_loss(y_pred, y_true, weight=None, axes=[2,3]):
    """
    Computes the combined Dice and crossentropy loss.
    """
    
    return dice_loss(y_pred, y_true, weight, axes) + crossentropy_loss(y_pred, y_true, weight, axes)

def iou_ce_loss(y_pred, y_true, weight=None, axes=[2,3]):
    """
    Computes the combined intersection over union (Jaccard index) and crossentropy loss.
    """
    
    return iou_loss(y_pred, y_true, weight, axes) + crossentropy_loss(y_pred, y_true, weight, axes)

def mcc_ce_loss(y_pred, y_true, weight=None, axes=[2,3]):
    """
    Computes the combined Mathews correlation coefficient (mcc) and crossentropy loss.
    """
    
    return mcc_loss(y_pred, y_true, weight, axes) + crossentropy_loss(y_pred, y_true, weight, axes)