from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd


def rmse(x, x_pred):
    return np.sqrt(((x - x_pred) ** 2).mean())


def pre_post_rmse_ratio(model, test_x, test_x_pred):
    '''Computes the pre-post root mean square prediction error for all
    in-place placebos and the treated units
    '''

    #Compute rmspe
    treated_pre = model.result.rmse
    treated_post = rmse(test_x, test_x_pred)

    #Store in dict
    rmse_dict = {"unit": model.data.label_treated.item(),
               "pre_rmspe": treated_pre,
               "post_rmspe": treated_post}
    
    #Compute post/pre rmspe ratio and add to rmse_df
    rmse_dict["post/pre"] = rmse_dict["post_rmspe"] / rmse_dict["pre_rmspe"]
    
    return rmse_dict

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
