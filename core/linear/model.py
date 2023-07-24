import numpy as np
from .data import Data
from .optimize import Optimize
from .metric import rmse, moving_average


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
        self.rmse = None

        self.in_space_placebo_w = None
        self.in_time_placebo_w = None
    

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
        # label_treated: (1,)
        # label_control: (n_control,)
        # time: (T,)
        self.data = Data(x_treated, x_control, label_treated, label_control, time)
        x_input = Data.process_input_data(treated_outcome=x_treated, control_outcome=x_control)

        self.result = Result(
            pen=self.pen, random_seed=self.random_seed,
            **x_input
        )

        #Get synthetic Control
        self.optimizer.optimize(
            data=self.result,
            placebo=False, pen=self.pen, steps=self.n_optim)
        
        # x_synth: (T, 1)
        x_synth = self.result.synth_outcome * self.data.treated_outcome[0]

        self.result.rmse = rmse(x_treated, x_synth)

        return x_synth
    
    def sample(self, x_control):
        # x_control: (T, n_control)
        x_control = x_control / self.data.control_outcome[0]

        # x_synth: (T, 1)
        x_synth = (self.result.w.T @ x_control.T).T
        x_synth = x_synth * self.data.treated_outcome[0]

        return x_synth

    def band(self, period=24):
        full = self.data.treated_outcome.squeeze()
        center = moving_average(full, period)
        full = full[period//2:-(period//2-1)]
        std = np.std(full - center)
        band = std + self.result.rmse
        return band
