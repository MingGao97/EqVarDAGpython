import numpy as np 
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings("ignore")
import EqVarDAG_TD_HD

# generate adjacency matrix

# p dimension, probConnect probability to connect nodes
def randomDAG_er(p, probConnect):
    DAG = np.zeros((p,p))
    causalOrder = np.random.permutation(range(p))
    for i in range(2,p):
        node = causalOrder[i]
        possibleParents = causalOrder[:i]
        Parents = possibleParents[np.random.binomial(1,probConnect,len(possibleParents))==1]
        DAG[Parents,node] = np.ones(len(Parents))
    DAG[causalOrder[0], causalOrder[1]] = 1
    return DAG, causalOrder

# indegree <= 3
# outdegree < 4
def randomDAG_chain(p, probConnect):
    DAG = np.zeros((p,p))
    causalOrder = np.random.permutation(range(p))
    for i in range(2,p):
        node = causalOrder[i]
        possibleParents =  causalOrder[:i]
        possibleParents = possibleParents[DAG[possibleParents,:].sum(axis=1) < 4]
        if len(possibleParents)>0:
            Parents = np.random.choice(possibleParents, min(len(possibleParents),2), replace = False)
            DAG[Parents,node] = np.ones(len(Parents))
        DAG[causalOrder[i-1], node] = 1
    DAG[causalOrder[0], causalOrder[1]] = 1
    return DAG, causalOrder

# No. <10 hub
def randomDAG_hub(p):
    DAG = np.zeros((p,p))
    causalOrder = np.random.permutation(range(p))
    Z = 10
    for i in range(2,p):
        node = causalOrder[i]
        DAG[causalOrder[i-1],node] = 1
        if i > 1:
            DAG[causalOrder[np.random.choice(range(min(i,Z)), 2, replace = False)], node] = 1
    DAG[causalOrder[0], causalOrder[1]] = 1
    return DAG, causalOrder

Bmin = 0.5

# generate coefficients from uniform (Bmin,1) and (-1,-Bmin)
# from adjacency matrix generate n by p matrix
# return truth adjacency matrix; X design matrix; TO ordering
def get_DAGdata(n, p, pc, gtype):
    if gtype == 'hub':
        truth, TO = randomDAG_hub(p)
    elif gtype == 'chain':
        truth, TO = randomDAG_chain(p, pc)
    else:
        truth, TO = randomDAG_er(p, pc)
    errs = np.random.normal(size = (p,n))
    B = truth.copy().T
    B[B==1] = np.random.uniform(Bmin,1,int(truth.sum()))*(2*np.random.binomial(1,0.5,int(truth.sum()))-1)
    X = np.linalg.solve(np.eye(p)-B, errs)
    X = X.T
    return truth, B, X, TO


ns = [80,100,200]
p_percent= [0.5,0.75,1,1.5,2]
pc = 0.5/p
N = 100

# type chain
for n in ns:
    ps = [int(n*z) for z in p_percent]
    for p in ps:
        res = np.zeros((N,4))
        for j in range(N):
            print('n: '+str(n)+' p: '+str(p)+' no. '+str(j))
            truth,_,X,TO = get_DAGdata(n,p,pc,'chain')
            sx,to = EqVarDAG_HD_TD(X,3)
            res[j,0] = kendalltau([np.where(TO==i) for i in range(p)],[to.index(i) for i in range(p)]).correlation
            res[j,1] = (truth*sx).sum()/truth.sum()
            res[j,2] = (truth*sx.T).sum()/truth.sum()
            res[j,3] = 1-(truth*sx).sum()/sx.sum()
        np.savetxt('resChain'+str(n)+'_'+str(p)+'.csv', res)

# type hub
for n in ns:
    ps = [int(n*z) for z in p_percent]
    for p in ps:
        res = np.zeros((N,4))
        for j in range(N):
            print('n: '+str(n)+' p: '+str(p)+' no. '+str(j))
            truth,_,X,TO = get_DAGdata(n,p,pc,'hub')
            sx,to = EqVarDAG_HD_TD(X,3)
            res[j,0] = kendalltau([np.where(TO==i) for i in range(p)],[to.index(i) for i in range(p)]).correlation
            res[j,1] = (truth*sx).sum()/truth.sum()
            res[j,2] = (truth*sx.T).sum()/truth.sum()
            res[j,3] = 1-(truth*sx).sum()/sx.sum()
        np.savetxt('resHub'+str(n)+'_'+str(p)+'.csv', res)
