from scipy import optimize
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

ytrain = np.load('ytrain 99 4.npy')
yvalidlist = np.load('yvalidlist 99 4.npy')
ypredlist = np.load('ypredlist 99 4.npy')
sigmalist = np.load('sigmalist 99 4.npy')

def utility(init_w, delta, blacklit_r, blacklit_sigma):
    def objective(w):
    
        return -np.dot(w,blacklit_r)+0.5*delta*np.dot(np.dot(w,blacklit_sigma),w)
    
    constraints = []
    for i in range(np.array(init_w).size):
        
        constraints.append(lambda w: 1 - w[i] )
        
    constraints.append(lambda w: 3-sum(abs(w)))            
    constraints.append(lambda w: sum(w)-1)        
    constraints.append(lambda w: -sum(w)+1)

    return optimize.fmin_cobyla(objective,init_w,constraints,rhoend=1e-7,maxfun=1000)
    
def blacklitterman(delta, weq, sigma, tau, P, Q, Omega):
  # Reverse optimize and back out the equilibrium returns
  # This is formula (12) page 6.
  pi = weq.dot(sigma * delta)
  # We use tau * sigma many places so just compute it once
  ts = tau * sigma
  # Compute posterior estimate of the mean
  # This is a simplified version of formula (8) on page 4.
  middle = linalg.inv(np.dot(np.dot(P,ts),P.T) + Omega)
  er = np.expand_dims(pi,axis=0).T + np.dot(np.dot(np.dot(ts,P.T),middle),(Q - np.expand_dims(np.dot(P,pi.T),axis=1)))
  # Compute posterior estimate of the uncertainty in the mean
  # This is a simplified and combined version of formulas (9) and (15)
  posteriorSigma = sigma + ts - ts.dot(P.T).dot(middle).dot(P).dot(ts)
  # Compute posterior weights based on uncertainty in mean
  # dit geeft een optimale w ZONDER constraints mbt shorten en zo
  w = utility(weq,delta,er,posteriorSigma)
  #w = er.T.dot(linalg.inv(delta * posteriorSigma)).T
  # Compute lambda value
  # We solve for lambda from formula (17) page 7, rather than formula (18)
  # just because it is less to type, and we've already computed w*.
  lmbda = np.dot(linalg.pinv(P).T,(w.T * (1 + tau) - weq).T)
  #return [er, w, lmbda]
  return w

DSdet = np.load('det.npy') #wekelijkse data vanaf 8 feb 93
US3mo = DSdet[39]
US3mo = US3mo[523:]
US3mo = US3mo/5200. #omzetten naar wekelijkse return


delta = 2.5
tau = 0.01
sigmablacklit = np.cov(ytrain, y=None, rowvar=0, bias=0, ddof=None)
tausigma = tau * sigmablacklit



weq = np.ones((yvalidlist.shape[1]))/(yvalidlist.shape[1])
brutorend = 1.
RPlist = []
nettorend = 1.
buyholdbrutorend = 1.
buyholdRPlist = []
wlist = np.zeros((yvalidlist.shape[1]))
c = 0.001
for i in range(yvalidlist.shape[0]):
    
    P = np.eye((yvalidlist.shape[1]))
    Q = ypredlist[i,:].reshape((yvalidlist.shape[1]),1)
    Omega = np.eye((yvalidlist.shape[1]))
    for j in range(yvalidlist.shape[1]):
        Omega[j][j] = sigmalist[i][j]**2.
        #Omega[j][j] = (np.std(ypredlist[:,j])**2)
    w_opt = blacklitterman(delta, weq, sigmablacklit, tau, P, Q, Omega)
    wlist = np.vstack((wlist,w_opt))
    RP = 0.
    buyholdRP = 0.
    for k in range(yvalidlist.shape[1]):
        RP += w_opt[k]*yvalidlist[i][k]
        buyholdRP += weq[k]*yvalidlist[i][k]
    brutorend = brutorend*(1+RP)
    RPlist.append(RP)
    
    buyholdbrutorend = buyholdbrutorend*(1+buyholdRP)
    buyholdRPlist.append(buyholdRP)  
    
    nettorend = nettorend*(1+RP)*(1-c*np.sum(abs(wlist[i+1]-wlist[i])))
    
RParray = np.asarray(RPlist)
buyholdRParray = np.asarray(buyholdRPlist)
US3mo = US3mo[:yvalidlist.shape[0]]
Sharpe = np.sqrt(52)*np.mean(RParray - US3mo)/np.std(RParray - US3mo)
buyholdSharpe = np.sqrt(52)*np.mean(buyholdRParray - US3mo)/np.std(buyholdRParray - US3mo)

brutocumrend = brutorend - 1.
nettocumrend = nettorend - 1.
buyholdbrutocumrend = buyholdbrutorend - 1.


#print Omega

print "Het Bruto Cumulatief Rendement :",brutocumrend
print "Het Netto Cumulatief Rendement :",nettocumrend
print "De Sharpe Ratio :",Sharpe

print "Het Bruto Cumulatief Rendement (buy hold):",buyholdbrutocumrend
print "De Sharpe Ratio (buy hold):",buyholdSharpe

#print np.sum(abs(wlist[10+1]-wlist[10]))
print wlist
