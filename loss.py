import numpy as np
import torch as th


def class_wise_mae_subscores(y_true, y_pred, bins=[-0.001, 0.5, 0.625, 0.75, 0.875, 1.0]):
    
    class_wise_maes = []
    y_true = np.array(y_true)/10.0
    y_pred = np.array(y_pred)/10.0
    
    for i in range(len(bins)-1):
        
        idx = (np.array(y_true) > bins[i]) & (np.array(y_true) <= bins[i+1])
        
        class_wise_maes.append(np.mean(np.abs(y_true[idx] - y_pred[idx]))) 
    
    return class_wise_maes

