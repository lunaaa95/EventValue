import numpy as np


class DataProcess:
    @staticmethod
    def process_input_data(treated_outcome, control_outcome):
        # treated_outcome: (T, 1)
        # control_outcome: (T, n_control)
        treated_outcome = treated_outcome / treated_outcome[0]
        control_outcome = control_outcome / control_outcome[0]

        # treated_covariates: (n_covariates, 1)
        treated_covariates = treated_outcome.mean(axis=0, keepdims=True)

        # control_covariates: (n_covariates, n_control)
        control_covariates = control_outcome.mean(axis=0, keepdims=True)

        n_covariates, n_controls = control_covariates.shape

        # Rescale covariates to be unit variance (helps with optimization)
        treated_covariates, control_covariates = DataProcess._rescale_covariate_variance(treated_covariates,
                                                                            control_covariates,
                                                                            n_covariates)
        
        # Get matrix of unitwise differences between control units to treated unit
        # pairwise_difference: (n_covariates, n_control)
        pairwise_difference = DataProcess._get_pairwise_difference_matrix(treated_covariates, 
                                                                   control_covariates)

        return {
            'n_controls': n_controls,
            'n_covariates':n_covariates,
            'treated_outcome': treated_outcome,
            'treated_covariates': treated_covariates,
            'control_outcome': control_outcome,
            'control_covariates': control_covariates,
            'pairwise_difference':pairwise_difference,
        }
    
    @staticmethod
    def _rescale_covariate_variance(treated_covariates, control_covariates, n_covariates):
        '''Rescale covariates to be unit variance'''

        #Combine control and treated into one big dataframe, over which we will compute variance for each covariate
        big_dataframe = np.concatenate((treated_covariates, control_covariates), axis=1)

        #Rescale each covariate to have unit variance
        big_dataframe /= np.std(big_dataframe, axis=1, keepdims=True)

        #Re-seperate treated and control from big dataframe
        treated_covariates = big_dataframe[:,0].reshape(n_covariates, 1) #First column is treated unit
        control_covariates = big_dataframe[:,1:] #All other columns are control units

        #Return covariate matices with unit variance
        return treated_covariates, control_covariates
    
    @staticmethod
    def _get_pairwise_difference_matrix(treated_covariates, control_covariates):
        '''
        Computes matrix of same shape as control_covariates, but with unit-wise difference from treated unit
        Used in optimization objective for both SC and DSC
        '''
        return treated_covariates - control_covariates
