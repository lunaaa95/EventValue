# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import cvxpy as cvx
from scipy.optimize import minimize, differential_evolution

class Optimize(object):
    '''
    This class is where the Synthetic Controls weights are solved for,
    More precisely, it contains all methods for parameter estimation, such as:

      W: a (1 x n_controls) matrix 
        containing the weights assigned to each
        control unit in the synthetic control. W is contstrained to be convex,
        that is sum(W)==1 and ∀w∈W, w≥0, each weight is non-negative and all weights sum to one.
        Referred to as W in Abadie, Diamond, Hainmueller.
        
      V: a (len(covariates) x len(covariates)) matrix 
        representing the relative importance of each covariate. V is constrained to be diagonal, positive semi-definite. 
        Practcally, this means that the product V.control_covariates and V.treated_covariates
        will always be non-negative. Further, we constrain sum(V)==1, otherwise there will an infinite
        number of solutions V*c, where c is a scalar, that assign equal relative importance to each covariate
        Referred to as V in Abadie, Diamond, Hainmueller.
        
      pen: float
        Coefficient representing the relative importance of minimizing differences between control units and treated unit 
        BEFORE weighting them (i.e. pairwise difference between control units and treated unit) as compared to
        AFTER weighting them (synthetic control vs. treated unit).
        A higher value means that pairwise differences are more important.
        If pen==0, then pairwise differences do not matter. Thus, if pen==0 optimize will solve for a normal synthetic control.

      constant: float
        Differenced synthetic controls allow for a constant offset (difference) between the treated unit and the synthetic control
        this constant is solved for here.
    '''

    def optimize(self, data, placebo, pen, steps=8):
        '''
        Solves the nested optimization function of finding the optimal synthetic control

        placebo: bool
            indicates whether the optimization is ran for finding the real synthetic control
            or as part of a placebo-style validity test. If True, only placebo class attributes are affected.

        steps: int
            The number of different initializations of v_0 the gradient descent optimization is ran for
            Higher values mean longer running time but higher chances of finding a globally optimal solution

        verbose: bool, default=False
            If true, prints additional detail regarding the state of the optimization
        '''
        args = (data, placebo, pen)
        
        for step in range(steps):

            #Approach for selecting initial v matrix:
            #First time, try a uniform v matrix, assigning equal weight to all covariates
            #Subsequent times, sample a random pmf using the dirichlet distribution
            if step == 0:
                v_0 = np.full(data.n_covariates, 1/data.n_covariates)
                if pen == "auto":
                    #if pen =="auto", we have an additional parameter to optimize over, so we append it
                    v_0 = np.append(v_0, 0)
            else:
                #Dirichlet distribution returns a valid pmf over n_covariates states
                v_0 = data.rng.dirichlet(np.ones(data.n_covariates), size=1)
                if pen == "auto":
                    #if pen =="auto", we have an additional parameter to optimize over, so we append it
                    v_0 = np.append(v_0, data.rng.lognormal(1.5, 1, size=1)) #Still experimenting with what distribution is appropriate
            

            #Required to have non negative values
            if pen != "auto":
                bnds = tuple((0,1) for _ in range(data.n_covariates))
            else:
                #if pen =="auto", we have an additional parameter to optimize over, and we need to bound it to be non-negative
                bnds = tuple((0,20) if ((pen=="auto") and (x==data.n_covariates)) else (0,1) for x in range(data.n_covariates + 1))

            #Optimze
            res = minimize(self.total_loss, v_0,  args=(args),
                            method='L-BFGS-B', bounds=bnds, 
                            options={'gtol': 1e-8,'disp':False, 'iprint':3})
        
        #If sampler did not converge, try again up to times before admitting defeat
        try:
            res.x
        except:
            data.fail_count += 1
            if data.fail_count <= 1:
                self.optimize(*args)

    def total_loss(self, v_0,
                   data, placebo, pen, verbose=False):
        '''
        Solves for w*(v) that minimizes loss function 1 given v,
        Returns loss from loss function 2 with w=w*(v)

        placebo: bool
            indicates whether the optimization is ran for finding the real synthetic control
            or as part of a placebo-style validity test. If True, only placebo class attributes are affected.
        '''

        assert placebo in [False, "in-time", "in-space"], "TypeError: Placebo must False, 'in-time' or 'in-space'"

        n_controls = data.control_outcome.shape[1]
        
        if pen == "auto":
            V = np.diag(v_0[:-1])
            pen_coef = v_0[-1]
        else:
            V = np.diag(v_0)
            pen_coef = pen
        
        # Construct the problem - constrain weights to be non-negative
        w = cvx.Variable((n_controls, 1), nonneg=True)
        
        #Define the objective

        #PROBLEM: treated_synth_difference = cvx.sum(V @ cvx.square(treated_covariates.T - control_covariates @ w)) runs better for normal sc,
        #but it doesnt work at all for in-time placebos, this probably means I am messing up the dimensionality somewhere in the processing
        #This is a work-around that works, but it ain't pretty
        if placebo == 'in-time':
            treated_synth_difference = cvx.sum(V @ cvx.square(data.treated_covariates - data.control_covariates @ w))
        else:
            treated_synth_difference = cvx.sum(V @ cvx.square(data.treated_covariates.T - data.control_covariates @ w))
        
        pairwise_difference = cvx.sum(V @ (cvx.square(data.pairwise_difference) @ w))
        objective = cvx.Minimize(treated_synth_difference + pen_coef*pairwise_difference)

        #Add constraint sum of weights must equal one
        constraints = [cvx.sum(w) == 1]
        
        #Solve problem
        problem = cvx.Problem(objective, constraints)
        
        try: #Try solving using current value of V, if it doesn't work return infinite loss
            result = problem.solve(verbose=verbose)
            loss = (data.treated_outcome - data.control_outcome @ w.value).T @ (data.treated_outcome - data.control_outcome @ w.value)
        except:
            return float(np.inf)
       
        #If loss is smaller than previous minimum, update loss, w and v
        if not placebo:
            if loss < data.min_loss:
                data.min_loss = loss
                data.w = w.value
                data.v = np.diagonal(V) / np.sum(np.diagonal(V)) #Make sure its normailzed (sometimes the optimizers diverge from bounds)
                data.pen = pen_coef
                data.synth_outcome = (data.w.T @ data.control_outcome.T).T #Transpose to make it (n_periods x 1)
                data.synth_covariates = data.control_covariates @ data.w

        elif placebo == "in-space":
            data.in_space_placebo_w = w.value

        elif placebo == "in-time":
            data.in_time_placebo_w = w.value
            
        #Return loss
        return loss
