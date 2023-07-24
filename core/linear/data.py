import numpy as np


class Data:
    @staticmethod
    def process_input_data(treated_outcome, control_outcome):
        # treated_outcome: (T, 1)
        # control_outcome: (T, n_control)

        # treated_covariates: (n_covariates, 1)
        treated_covariates = treated_outcome.mean(axis=0, keepdims=True)

        # control_covariates: (n_covariates, n_control)
        control_covariates = control_outcome.mean(axis=0, keepdims=True)

        n_covariates, n_controls = control_covariates.shape

        # Rescale covariates to be unit variance (helps with optimization)
        treated_covariates, control_covariates = Data._rescale_covariate_variance(treated_covariates,
                                                                            control_covariates,
                                                                            n_covariates)
        
        # Get matrix of unitwise differences between control units to treated unit
        # pairwise_difference: (n_covariates, n_control)
        pairwise_difference = Data._get_pairwise_difference_matrix(treated_covariates, 
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


class DataProcessor(object):
    '''Class that processes input data into variables and matrices needed for optimization'''
    
    def _process_input_data(self, dataset, 
                            outcome_var, id_var, time_var, 
                            treatment_period, treated_unit, 
                            pen, exclude_columns, random_seed,
                            **kwargs):
        '''
        Extracts processed variables, excluding v and w, from input variables.
        These are all the data matrices.
        '''
        #All columns not y, id or time must be predictors
        covariates = [col for col in dataset.columns if col not in [id_var, time_var] and col not in exclude_columns]

        #Extract quantities needed for pre-processing matrices
        periods_pre_treatment = dataset.loc[dataset[time_var]<treatment_period][time_var].nunique()
        #Number of control units, -1 to remove treated unit
        n_controls = dataset[id_var].nunique() - 1
        n_covariates = len(covariates)

        #All units that are not the treated unit are controls
        control_units = dataset.loc[dataset[id_var] != treated_unit][id_var].unique()

        ###Get treated unit matrices first###
        treated_outcome, unscaled_treated_covariates = self._process_treated_data(
            dataset, outcome_var, id_var, time_var, 
            treatment_period, treated_unit, 
            periods_pre_treatment, covariates, n_covariates
        )

        ### Now for control unit matrices ###
        control_outcome, unscaled_control_covariates = self._process_control_data(
            dataset, outcome_var, id_var, time_var, 
            treatment_period, treated_unit, n_controls, 
            periods_pre_treatment, covariates
        )
        
        #Rescale covariates to be unit variance (helps with optimization)
        treated_covariates, control_covariates = self._rescale_covariate_variance(unscaled_treated_covariates,
                                                                            unscaled_control_covariates,
                                                                            n_covariates)
        
        #Get matrix of unitwise differences between control units to treated unit
        pairwise_difference = self._get_pairwise_difference_matrix(treated_covariates, 
                                                                   control_covariates)

        return {
            'dataset': dataset,
            'outcome_var':outcome_var,
            'id_var':id_var,
            'time_var':time_var,
            'treatment_period':treatment_period,
            'treated_unit':treated_unit,
            'control_units':control_units,
            'covariates':covariates,
            'periods_pre_treatment':periods_pre_treatment,
            'n_controls': n_controls,
            'n_covariates':n_covariates,
            'treated_outcome': treated_outcome,
            'treated_covariates': treated_covariates,
            'unscaled_treated_covariates':unscaled_treated_covariates,
            'control_outcome': control_outcome,
            'control_covariates': control_covariates,
            'unscaled_control_covariates': unscaled_control_covariates,
            'pairwise_difference':pairwise_difference,
            'pen':pen,
            'random_seed':random_seed,
        }
    
    def _process_treated_data(self, dataset, outcome_var, id_var, time_var, treatment_period, treated_unit, 
                            periods_pre_treatment, covariates, n_covariates):
        '''
        Extracts and formats outcome and covariate matrices for the treated unit
        '''

        treated_data_all = dataset.loc[dataset[id_var] == treated_unit]
        
        #Only pre-treatment
        treated_data = treated_data_all.loc[dataset[time_var] < treatment_period]
        #Extract outcome and shape as matrix
        treated_outcome = np.array(treated_data[outcome_var]).reshape(periods_pre_treatment, 1)
        #Columnwise mean of each covariate in pre-treatment period for treated unit, shape as matrix
        treated_covariates = np.array(treated_data[covariates].mean(axis=0)).reshape(n_covariates, 1)

        return treated_outcome, treated_covariates
    

    def _process_control_data(self, dataset, outcome_var, id_var, time_var, treatment_period, treated_unit, n_controls, 
                            periods_pre_treatment, covariates):
        '''
        Extracts and formats outcome and covariate matrices for the control group
        '''

        #Every unit that is not the treated unit is control
        control_data_all = dataset.loc[dataset[id_var] != treated_unit]
        
        #Only pre-treatment
        control_data = control_data_all.loc[dataset[time_var] < treatment_period]
        #Extract outcome, then shape as matrix
        control_outcome = np.array(control_data[outcome_var]).reshape(n_controls, periods_pre_treatment).T
        
        #Extract the covariates for all the control units
        #Identify which rows correspond to which control unit by setting index,
        #then take the unitwise mean of each covariate
        #This results in the desired (n_control x n_covariates) matrix
        control_covariates = np.array(control_data[covariates].\
                set_index(np.arange(len(control_data[covariates])) // periods_pre_treatment).groupby(level=-1).mean()).T

        return control_outcome, control_covariates

    def _rescale_covariate_variance(self, treated_covariates, control_covariates, n_covariates):
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
    
    def _get_pairwise_difference_matrix(self, treated_covariates, control_covariates):
        '''
        Computes matrix of same shape as control_covariates, but with unit-wise difference from treated unit
        Used in optimization objective for both SC and DSC
        '''
        return treated_covariates - control_covariates