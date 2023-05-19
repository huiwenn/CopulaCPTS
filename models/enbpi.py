
 # from https://github.com/hamrel-cxu/EnbPI/

import importlib
import warnings
#from utils_EnbPI import generate_bootstrap_samples, strided_app, weighted_quantile
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import os
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
import sys



def generate_bootstrap_samples(n, m, B):
    '''
      Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
    '''
    samples_idx = np.zeros((B, m), dtype=int)
    for b in range(B):
        sample_idx = np.random.choice(n, m)
        samples_idx[b, :] = sample_idx
    return(samples_idx)

class prediction_interval():
    '''
        Create prediction intervals using different methods (Ensemble, LOO, ICP, weighted...)
    '''

    def __init__(self, fit_func, X_train, X_predict, Y_train, Y_predict):
        '''
            Fit_func: ridge, lasso, linear model, data
        '''
        self.regressor = fit_func
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
        # it will be updated with a list of bootstrap models, fitted on subsets of training data
        self.Ensemble_fitted_func = []
        # it will store residuals e_1, e_2,... from Ensemble
        self.Ensemble_online_resid = np.array([])
        self.ICP_fitted_func = []  # it only store 1 fitted ICP func.
        # it will store residuals e_1, e_2,... from ICP
        self.ICP_online_resid = np.array([])
        self.WeightCP_online_resid = np.array([])
    '''
        Algorithm: Ensemble (online)
            Main difference from earlier is
            1. We need to store these bootstrap estimators f^b
            2. when aggregating these stored f^b to make prediction on future points,
            do not aggregate all of them but randomly select B*~Binom(B,e^-1 ~= (1-1/k)^k) many f^b
            3. the residuals grow in length, so that a new point uses all previous residuals to create intervals
            (Thus intervals only get wider, not shorter)
    '''

    def fit_bootstrap_models_online(self, alpha, B, miss_test_idx):
        '''
          Train B bootstrap estimators from subsets of (X_train, Y_train) and calculate predictions on original data X_train
          Return 1-\alpha quantile of each prdiction on self.X_predict, also
          1. update self.Ensemble_fitted_func with bootstrap estimators and
          2. update self.Ensemble_online_resid with LOO online residuals (from training)
          Update:
           Include tilt option (only difference is using a different test data, so just chaneg name from predict to predict_tilt)
        '''
        n = len(self.X_train)
        n1 = len(self.X_predict)
        # hold indices of training data for each f^b
        boot_samples_idx = generate_bootstrap_samples(n, n, B)
        # hold predictions from each f^b
        boot_predictions = np.zeros((B, (n+n1)), dtype=float)
        # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
        in_boot_sample = np.zeros((B, n), dtype=bool)
        out_sample_predict = np.zeros((n, n1))
        ind_q = int((1-alpha)*n)
        for b in range(B):
            model = self.regressor()

            model.train(self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b], :],
                      epochs=100, batch_size=100)

            boot_predictions[b] = model.predict(np.r_[self.X_train, self.X_predict]).flatten()
            self.Ensemble_fitted_func.append(model)
            
            in_boot_sample[b, boot_samples_idx[b]] = True
        
        for i in range(n):
            b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1) # index of ensemble not including sample i
            # calibration
            if(len(b_keep) > 0):
                resid_LOO = np.abs(self.Y_train[i] - boot_predictions[b_keep, i].mean())
                self.Ensemble_online_resid = np.append(self.Ensemble_online_resid, resid_LOO)
                out_sample_predict[i] = boot_predictions[b_keep, n:].mean(0)
            else:  # if aggregating an empty set of models, predict zero everywhere
                resid_LOO = np.abs(self.Y_train[i])
                self.Ensemble_online_resid = np.append(self.Ensemble_online_resid, resid_LOO)
                out_sample_predict[i] = np.zeros(n1)

        sorted_out_sample_predict = np.sort(out_sample_predict, axis=0)[ind_q]  # length n1
        # TODO: Change this, because ONLY minus the non-missing predictions
        # However, need to make sure same length is maintained, o/w sliding cause problem
        
        resid_out_sample = np.abs(sorted_out_sample_predict-self.Y_predict)
        self.Ensemble_online_resid = np.append(self.Ensemble_online_resid, resid_out_sample)
        return(sorted_out_sample_predict)

    
    def compute_PIs_Ensemble(self, alpha, B, stride, miss_test_idx, density_est=False):
        '''
            Note, this is not online version, so all test points have the same width
        '''
        n = len(self.X_train)
        n1 = len(self.Y_predict)
        # Now f^b and LOO residuals have been constructed from earlier
        out_sample_predict = self.fit_bootstrap_models_online(
            alpha, B, miss_test_idx)  # length of n1
        ind_q = int(100*(1-alpha))
        # start = time.time()
        if density_est:
            blocks = int(n1/stride)
            ind_q = np.zeros(blocks)
            p_vals = self.Ensemble_online_resid[:n]  # This will be changing
            p_vals = np.array([np.sum(i > p_vals)/len(p_vals) for i in p_vals])
            # Fill in first (block) of estimated quantiles:
            ind_q[0] = 100*beta_percentile(p_vals, alpha)
            # Fill in consecutive blocks
            for block in range(blocks-1):
                p_vals = p_vals[stride:]
                new_p_vals = self.Ensemble_online_resid[n+block*stride:n+(block+1)*stride]
                new_p_vals = np.array([np.sum(i > new_p_vals)/len(new_p_vals) for i in new_p_vals])
                p_vals = np.hstack((p_vals, new_p_vals))
                ind_q[block+1] = 100*beta_percentile(p_vals, alpha)
            ind_q = ind_q.astype(int)
            width = np.zeros(blocks)
            strided_resid = strided_app(self.Ensemble_online_resid[:-1], n, stride)
            for i in range(blocks):
                width[i] = np.percentile(strided_resid[i], ind_q[i], axis=-1)
        else:
            width = np.percentile(strided_app(
                self.Ensemble_online_resid[:-1], n, stride), ind_q, axis=-1)
        width = np.abs(np.repeat(width, stride))  # This is because |width|=T/stride.
        PIs_Ensemble = pd.DataFrame(np.c_[out_sample_predict-width,
                                          out_sample_predict+width], columns=['lower', 'upper'])
        # print(time.time()-start)
        return PIs_Ensemble
    
    def run_experiments(self, alpha, B, stride, data_name, itrial,  miss_test_idx, true_Y_predict=[], density_est=False, get_plots=False, none_CP=False, methods=['Ensemble', 'ICP', 'Weighted_ICP']):
        '''
            Note, it is assumed that real data has been loaded, so in actual execution,
            generate data before running this
            Default is for running real-data
            NOTE: I added a "true_Y_predict" option, which will be used for calibrating coverage under missing data
            In particular, this is needed when the Y_predict we use for training is NOT the same as true Y_predict
        '''
        train_size = len(self.X_train)
        np.random.seed(98765+itrial)
       
        results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                        'method', 'train_size', 'coverage', 'width'])
        PIs = []
        for method in methods:
            print(f'Runnning {method}')
            if method == 'JaB':
                B_ = B
                n = len(self.X_train)
                B = int(np.random.binomial(int(B_/(1-1./(1+train_size))**n),
                                           (1-1./(1+train_size))**n, size=1))
                PI = self.compute_PIs_JaB(alpha, B)
            elif method == 'Ensemble':
                PI = eval(f'compute_PIs_{method}_online({alpha},{B},{stride},{miss_test_idx},{density_est})',
                          globals(), {k: getattr(self, k) for k in dir(self)})
            else:
                l = int(0.5*len(self.X_train))
                PI = eval(f'compute_PIs_{method}_online({alpha},{l},{density_est})',
                          globals(), {k: getattr(self, k) for k in dir(self)})
            PIs.append(PI)
            coverage = ((np.array(PI['lower']) <= self.Y_predict) & (
                np.array(PI['upper']) >= self.Y_predict)).mean()
            if len(true_Y_predict) > 0:
                coverage = ((np.array(PI['lower']) <= true_Y_predict) & (
                    np.array(PI['upper']) >= true_Y_predict)).mean()
            print(f'Average Coverage is {coverage}')
            width = (PI['upper'] - PI['lower']).mean()
            print(f'Average Width is {width}')
            results.loc[len(results)] = [itrial, data_name,
                                         self.regressor.__class__.__name__, method, train_size, coverage, width]
        
        return(results)
