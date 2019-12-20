import numpy as np
#from sklearn.linear_model import LassoCV
#from sklearn.linear_model import Lasso
from statsmodels.api import OLS

from rpy2.robjects import r as ror
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as numpy2ri

# get ordering of the nodes by computing the conditional variance
# X is a n by p design matrix
def EqVarDAG_TD_internal(X):
    n, p = X.shape
    done = []
    S = np.cov(X.T)
    Sinv = np.linalg.inv(S)    
    for i in range(p):
        varmap = np.delete(np.array(range(p)), done)
        v = np.diag(np.linalg.inv(np.delete(np.delete(Sinv,done,axis=0),done,axis=1))).argmin()
        done.append(varmap[v])
    return done

# main function
# X is a n by p design matrix
def EqVarDAG_TD(X):
    n, p = X.shape
    r = ror
    numpy2ri.activate()
    try:
        glmnet = importr('glmnet')
    except:
        utils = importr('utils')
        utils.install_packages('glmnet')
        glmnet = importr('glmnet')
    # get ordering of the nodes
    rr = EqVarDAG_TD_internal(X)[::-1]
    # adjacency matrix
    result = np.zeros([p,p])
    for i in range(p-1):
        now = rr[i]
        this = sorted(rr[(i+1):])
        if len(this) > 1:
            lassoModel = glmnet.cv_glmnet(X[:,this],X[:,now][:,np.newaxis])
            betaFit = np.array(r['as.matrix'](r['coefficients'](lassoModel)))[1:,0]
            
            '''sklean    
            # variable selection by lasso, not estimate coefficients
            # model selection by 1se rule
            lassoModel = LassoCV(cv = 10).fit(X[:,this],X[:,now])
            indi = np.where(lassoModel.alphas_ == lassoModel.alpha_)[0]
            maxcv = lassoModel.mse_path_[indi,:].mean() + lassoModel.mse_path_[indi,:].std()/np.sqrt(10)
            best = np.where(lassoModel.mse_path_.mean(axis=1) < maxcv)[0][0]
            lassoModel = Lasso(alpha = lassoModel.alphas_[best]).fit(X[:,this],X[:,now])
            betaFit = lassoModel.coef_
            
            '''
            for j in range(len(this)):
                if betaFit[j]!=0:
                    result[this[j],now] = 1
        else:
            # for last one, use OLS t test
            ols = OLS(X[:,now],X[:,this]).fit()
            if ols.pvalues < 0.05:
                result[this,now] = 1
    return result, rr[::-1]

