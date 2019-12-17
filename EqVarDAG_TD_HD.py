import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from statsmodels.api import OLS

from rpy2.robjects import r as ror
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as numpy2ri
#import itertools

# implement best subset regression by self-written function
'''
def BestSubsetReg(X,Y,J):
    n = X.shape[0]
    J = min(J, X.shape[1])
    RSS = []
    sigmahat = []
    for k in range(J):
        RSStemp = []
        for combo in itertools.combinations(range(X.shape[1]),k+1):
            #no intercept
            ols = OLS(Y,X[:,list(combo)]).fit()
            RSStemp.append(ols.ssr)
        RSS.append(min(RSStemp))
        sigmahat.append(RSS[k]/(n-1-k))
    return min(RSS), min(sigmahat)
'''


# implement best subset regression by R package 'leaps'
def BestSubsetReg(X,Y,J,r,leaps):
    n = X.shape[0]
    res = leaps.regsubsets( X, Y,
                            method='exhaustive',
                            nbest = 1,
                            nvmax = J,
                            **{'really.big': True})
    rss = np.array(r.summary(res)[2])
    sigmahat = rss/(n-1-rss/(n-2-np.array(range(len(rss)))))
    return min(rss), min(sigmahat)


# get ordering of the nodes by best subset regression with subset size < J (indegree)
# every step chooses the smallest estimated conditional variance
# for here (and the original code), use RSS to estimate, may be not appropriate
# X is n by p design matrix, J is the max indegree
def getOrdering(X, J):
    ### use r package leaps for BestSubsetReg
    try:
        leaps = importr('leaps')
    except:
        utils = importr('utils')
        utils.install_packages('leaps')
        leaps = importr('leaps')
    r = ror
    numpy2ri.activate()
    ###
    p = X.shape[1]
    theta = []
    rest = list(range(p))
    for i in range(p-1):
        if i==0:
            #NO.1
            theta.append(np.argmin(X.var(axis=0)))
            rest.remove(theta[0])
        elif i==1:
            #NO.2
            out = np.apply_along_axis(lambda z: OLS(z,X[:,theta]).fit().ssr, 0, np.delete(X,theta,axis=1))
            theta.append(rest[np.argmin(out)]) 
            rest.remove(theta[1])
        else:
            #NO.3 to NO.(p-1)
            #out return with a matrix with first row RSS; second row sigma hat, here use RSS
            out = np.apply_along_axis(lambda z:BestSubsetReg(X[:,theta],z,J=J,r=r,leaps=leaps), 0, np.delete(X,theta,axis=1))
            theta.append(rest[np.argmin(out[0,:])])
            rest.remove(theta[i])
    theta.append(rest[0])
    return theta


# main function
# X is n by p design matrix, J is the max indegree
def EqVarDAG_HD_TD(X, J):
    n, p = X.shape
    # get ordering
    rr = getOrdering(X, J)[::-1]
    # adjacancy matrix
    result = np.zeros([p,p])
    for i in range(p-1):
        now = rr[i]
        this = sorted(rr[(i+1):])
        if len(this) > 1:
            if n > 100:
                # n > 100, lasso select model by 1se rule
                lassoModel = LassoCV(cv=10).fit(X[:,this],X[:,now])
                indi = np.where(lassoModel.alphas_ == lassoModel.alpha_)[0]
                maxcv = lassoModel.mse_path_[indi,:].mean() + lassoModel.mse_path_[indi,:].std()/np.sqrt(10)
                best = np.where(lassoModel.mse_path_.mean(axis=1) < maxcv)[0][0]
                lassoModel = Lasso(alpha = lassoModel.alphas_[best]).fit(X[:,this],X[:,now])
                betaFit = lassoModel.coef_
            else:
                # if n < 100, use a BIC-like criterian to selection model
                _,lassoCoefs,_ = Lasso().path(X[:,this],X[:,now])
                resid = np.apply_along_axis(lambda z:z-X[:,now], 0, X[:,this]@lassoCoefs)
                rss = (resid**2).sum(axis=0)
                df = (lassoCoefs!=0).sum(axis=0)
                bic = n*np.log(rss/n) + df*np.log(n) + 2*df*np.log(p-i-1)
                betaFit = lassoCoefs[:,np.argmin(bic)]
            for j in range(len(this)):
                if betaFit[j]!=0:
                    result[this[j],now] = 1    
        else:
            # for last one, use OLS t test
            ols = OLS(X[:,now],X[:,this]).fit()
            if ols.pvalues < 0.05:
                result[this,now] = 1

    return result, rr[::-1]