import numpy as np
from .data import DataProcessor, Data
from .optimize import Optimize


class Result:
    def __init__(self, n_controls, n_covariates, pen,
                 treated_outcome, control_outcome, treated_covariates, control_covariates, pairwise_difference,
                 w=None, v=None, random_seed=0,
                 ) -> None:
        # params
        self.n_controls = n_controls
        self.n_covariates = n_covariates
        self.pen = pen
        self.rng = np.random.default_rng(random_seed)

        # data
        self.treated_outcome = treated_outcome
        self.control_outcome = control_outcome
        self.treated_covariates = treated_covariates
        self.control_covariates = control_covariates
        self.pairwise_difference = pairwise_difference

        # results
        self.w = w
        self.v = v
        self.synth_outcome = None
        self.synth_covariates = None

        # used in optimization
        self.min_loss = float("inf")
        self.fail_count = 0

        # metric
        self.rmspe_df = None

        self.in_space_placebo_w = None
        self.in_time_placebo_w = None


class SynthBase(object):
    '''Class that stores all variables and results'''
    
    def __init__(self, dataset, outcome_var, id_var, time_var, treatment_period, treated_unit, control_units,
                covariates, periods_pre_treatment, n_controls, n_covariates,
                treated_outcome, control_outcome, treated_covariates, control_covariates, 
                unscaled_treated_covariates, unscaled_control_covariates,
                pairwise_difference, pen, random_seed=0,
                w=None, v=None, **kwargs):

        '''
        INPUT VARIABLES:
        
        dataset: the dataset for the synthetic control procedure.
        Should have the the following column structure:
        ID, Time, outcome_var, x0, x1,..., xn
        Each row in dataset represents one observation.
        The dataset should be sorted on ID then Time. 
        That is, all observations for one unit in order of Time, 
        followed by all observations by the next unit also sorted on time
        
        ID: a string containing a unique identifier for the unit associated with the observation.
        E.g. in the simulated datasets provided, the ID of the treated unit is "A".
        
        Time: an integer indicating the time period to which the observation corresponds.
        
        treated_unit: ID of the treated unit
        '''

        self.dataset = dataset
        self.outcome_var = outcome_var
        self.id = id_var
        self.time = time_var
        self.treatment_period = treatment_period
        self.treated_unit = treated_unit
        self.control_units = control_units
        self.covariates = covariates
        # self.periods_all = periods_all
        self.periods_pre_treatment = periods_pre_treatment
        self.n_controls = n_controls
        self.n_covariates = n_covariates
        self.pen = pen
        self.rng = np.random.default_rng(random_seed)

        
        '''
        PROCESSED VARIABLES:
        
        treated_outcome: a (1 x treatment_period) matrix containing the
        outcome of the treated unit for each observation in the pre-treatment period.
        Referred to as Z1 in Abadie, Diamond, Hainmueller.
        
        control_outcome: a ((len(unit_list)-1) x treatment_period) matrix containing the
        outcome of every control unit for each observation in the pre-treatment period
        Referred to as Z0 in Abadie, Diamond, Hainmueller.
        
        treated_outcome_all: a (1 x len(time)) matrix
        same as treated_outcome but includes all observations, including post-treatment
        
        control_outcome_all: a (n_controls x len(time)) matrix
        same as control_outcome but includes all observations, including post-treatment
        
        treated_covariates: a (1 x len(covariates)) matrix containing the
        average value for each predictor of the treated unit in the pre-treatment period
        Referred to as X1 in Abadie, Diamond, Hainmueller.
        
        control_covariates: a (n_controls x len(covariates)) matrix containing the
        average value for each predictor of every control unit in the pre-treatment period
        Referred to as X0 in Abadie, Diamond, Hainmueller.
        
        W: a (1 x n_controls) matrix containing the weights assigned to each
        control unit in the synthetic control. W is contstrained to be convex,
        that is sum(W)==1 and ∀w∈W, w≥0, each weight is non-negative and all weights sum to one.
        Referred to as W in Abadie, Diamond, Hainmueller.
        
        V: a (len(covariates) x len(covariates)) matrix representing the relative importance
        of each covariate. V is contrained to be diagonal, positive semi-definite. 
        Pracitcally, this means that the product V.control_covariates and V.treated_covariates
        will always be non-negative. Further, we constrain sum(V)==1, otherwise there will an infinite
        number of solutions V*c, where c is a scalar, that assign equal relative importance to each covariate
        Referred to as V in Abadie, Diamond, Hainmueller.
        '''
        ###Post processing quantities
        self.treated_outcome = treated_outcome
        self.control_outcome = control_outcome
        self.treated_covariates = treated_covariates
        self.control_covariates = control_covariates
        self.unscaled_treated_covariates = unscaled_treated_covariates
        self.unscaled_control_covariates = unscaled_control_covariates
        # self.treated_outcome_all = treated_outcome_all
        # self.control_outcome_all = control_outcome_all
        self.pairwise_difference = pairwise_difference

        ###Post inference quantities
        self.w = w #Can be provided
        self.v = v #Can be provided
        self.weight_df = None
        self.comparison_df = None
        self.synth_outcome = None
        self.synth_constant = None
        self.synth_covariates = None
        self.rmspe_df = None
        #used in optimization
        self.min_loss = float("inf")
        self.fail_count = 0 #Used to limit number of optimization attempts

        ###Validity tests
        self.in_space_placebos = None
        self.in_space_placebo_w = None
        self.pre_post_rmspe_ratio = None
        self.in_time_placebo_outcome = None
        self.in_time_placebo_treated_outcome = None
        self.in_time_placebo_w = None
        self.placebo_treatment_period = None
        self.placebo_periods_pre_treatment = None


class Synth():
    '''Class implementing the Synthetic Control Method'''

    def __init__(self, dataset, 
                outcome_var, id_var, time_var, 
                treatment_period, treated_unit, 
                n_optim=10, pen=0, exclude_columns=[], random_seed=0,
                **kwargs):
        '''
        data: 
          Type: Pandas dataframe. 
          A pandas dataframe containing the dataset. Each row should contain one observation for a unit at a time, 
          including the outcome and covariates. Dataset should be ordered by unit then time.

        outcome_var: 
          Type: str. 
          Name of outcome column in data, e.g. "gdp"

        id_var: 
          Type: str. 
          Name of unit indicator column in data, e.g. "country"

        time_var: 
          Type: str. 
          Name of time column in data, e.g. "year"

        treatment_period: 
          Type: int. 
          Time of first observation after the treatment took place, i.e. first observation affected by the treatment effect.
          E.g. 1990 for german reunification.

        treated_unit: 
          Type: str. 
          Name of the unit that recieved treatment, e.g. "West Germany"
          data["id_var"] == treated_unit

        n_optim: 
          Type: int. Default: 10. 
          Number of different initialization values for which the optimization is run. 
          Higher number means longer runtime, but a higher chance of a globally optimal solution.

        pen:
          Type: float. Default: 0.
          Penalization coefficient which determines the relative importance of minimizing the sum of the pairwise difference of each individual
          control unit in the synthetic control and the treated unit, vis-a-vis the difference between the synthetic control and the treated unit.
          Higher number means pairwise difference matters more.
        
        '''
        self.method = "SC"

        original_checked_input = self._process_input_data(
            dataset, outcome_var, id_var, time_var, treatment_period, treated_unit, pen, 
            exclude_columns, random_seed, **kwargs
        )



        self.original_data = SynthBase(**original_checked_input)

        #Get synthetic Control
        self.optimize(self.original_data, False, pen, n_optim)
        
        #Compute rmspe_df
        # self._pre_post_rmspe_ratios(None, False)

        #Prepare weight_df with unit weights
        # self.original_data.weight_df = self._get_weight_df(self.original_data)
        # self.original_data.comparison_df = self._get_comparison_df(self.original_data)

    def _process_input_data(self, dataset, 
                        outcome_var, id_var, time_var, 
                        treatment_period, treated_unit, 
                        pen, exclude_columns, random_seed,
                        **kwargs):
        try:
            self.data_processor
        except AttributeError:
            self.data_processor = DataProcessor()
        return self.data_processor._process_input_data(
            dataset, outcome_var, id_var, time_var, treatment_period, treated_unit, pen, 
            exclude_columns, random_seed, **kwargs
            )
    
    def optimize(self,
            data,
            placebo,
            pen, steps=8
            ):
        try:
            self.optimizer
        except AttributeError:
            self.optimizer = Optimize()
        return self.optimizer.optimize(
            data,
            placebo,
            pen, steps=8
            )
    

class SynthControl:
    """Class implementing the Synthetic Control Method"""
    def __init__(self, pen='auto', n_optim=10, random_seed=0) -> None:
        """
        n_optim: 
          Type: int. Default: 10. 
          Number of different initialization values for which the optimization is run. 
          Higher number means longer runtime, but a higher chance of a globally optimal solution.

        pen:
          Type: float. Default: 0.
          Penalization coefficient which determines the relative importance of minimizing the sum of the pairwise difference of each individual
          control unit in the synthetic control and the treated unit, vis-a-vis the difference between the synthetic control and the treated unit.
          Higher number means pairwise difference matters more.
        """
        self.pen = pen
        self.n_optim = n_optim
        self.random_seed = random_seed
        self.optimizer = Optimize()

    def generate(self, x_treated, x_control,
                 label_treated, label_control, time
                 ):
        # x_treated: (T, 1)
        # x_control: (T, n_control)
        x_input = Data.process_input_data(treated_outcome=x_treated, control_outcome=x_control)

        # label_treated: (1,)
        # label_control: (n_control,)
        # time: (T,)
        self.label_treated = label_treated
        self.label_control = label_control
        self.time = time

        self.result = Result(
            pen=self.pen, random_seed=self.random_seed,
            **x_input
        )

        #Get synthetic Control
        self.optimizer.optimize(
            data=self.result,
            placebo=False, pen=self.pen, steps=self.n_optim)
        
        x_synth = self.result.synth_outcome

        return x_synth