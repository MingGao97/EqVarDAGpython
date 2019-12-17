import numpy as np 
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings("ignore")
import EqVarDAG_TD

# generate adjacency matrix
# p dimension, probConnect probability to connect nodes
def randomDAG(p, probConnect):
    DAG = np.zeros((p,p))
    causalOrder = np.random.permutation(range(p))
    for i in range(p-2):
        node = causalOrder[i]
        possibleParents = causalOrder[(i+1):]
        numberParents = np.random.binomial(p-i-1,probConnect,1)
        Parents = np.random.choice(possibleParents, numberParents, replace = False)
        Parents = np.concatenate(([causalOrder[i+1]],Parents))
        DAG[Parents,node] = np.ones(numberParents+1)
    node = causalOrder[p-2]
    DAG[causalOrder[p-1],node] = 1
    return DAG, causalOrder[::-1]

Bmin = 0.3
# generate coefficients from uniform (Bmin,1) and (-1,-Bmin)
# from adjacency matrix generate n by p matrix
# return truth adjacency matrix; X design matrix; TO ordering
def get_DAGdata(n,p,pc):
    truth, TO = randomDAG(p,pc)
    errs = np.random.normal(size = (p,n))
    B = truth.copy().T
    B[B==1] = np.random.uniform(Bmin,1,int(truth.sum()))*(2*np.random.binomial(1,0.5,int(truth.sum()))-1)
    X = np.linalg.solve(np.eye(p)-B, errs)
    X = X.T
    return truth, B, X, TO


N = 500
ns = [100,500,1000]
ps = [5,20,40]


# dense setting
pc = 0.3

for n in ns:
    for p in ps:
        res = np.zeros((N,4))
        for j in range(N):
            print('n: '+str(n)+' p: '+str(p)+' no. '+str(j))
            truth,_,X,TO = get_DAGdata(n,p,pc)
            sx,to = EqVarDAG_TD(X)
            res[j,0] = kendalltau([np.where(TO==i) for i in range(p)],[to.index(i) for i in range(p)]).correlation
            res[j,1] = (truth*sx).sum()/truth.sum()
            res[j,2] = (truth*sx.T).sum()/truth.sum()
            res[j,3] = 1-(truth*sx).sum()/sx.sum()
        np.savetxt('resDense'+str(n)+'_'+str(p)+'.csv', res)

# sparse setting
for n in ns:
    for p in ps:
        pc = 3/(2*p-2)
        res = np.zeros((N,4))
        for j in range(N):
            print('n: '+str(n)+' p: '+str(p)+' no. '+str(j))
            truth,_,X,TO = get_DAGdata(n,p,pc)
            sx,to = EqVarDAG_TD(X)
            res[j,0] = kendalltau([np.where(TO==i) for i in range(p)],[to.index(i) for i in range(p)]).correlation
            res[j,1] = (truth*sx).sum()/truth.sum()
            res[j,2] = (truth*sx.T).sum()/truth.sum()
            res[j,3] = 1-(truth*sx).sum()/sx.sum()
        np.savetxt('resSparse'+str(n)+'_'+str(p)+'.csv', res)
