from scipy import optimize
import numpy as np
#github

r = np.array([[0.1],[0.2],[-0.05],[0.5]]) #array of predicted returns, transformed with Black Lit
init = np.array([-0.3,0.4,0.3,0.2]) #initial solution (portfolio weights)
delta = 2. #risk aversion
sigma = np.array([[1,2,3,3],[4,5,6,6],[7,8,9,9],[10,11,12,13]]) #covariance matrix of the assets, transformed with Black Lit

def objective(w):
    print(w)
    return -np.dot(w,r)+0.5*delta*np.dot(np.dot(w,sigma),w)
  
    
#constr 1 and 2 : sum(w) = 1
def con1(w):
    return -sum(w)+1
    
def con2(w):
    return sum(w)-1  
    
#constr 3 to 5: every w should be < 1  

constraints = []
for i in range(init.size):
    constraints.append(lambda w:1-w[i] )
def con3(w):
    return 1-w[0]
   
def con4(w):
    return 1-w[1]   

def con5(w):
    return 1-w[2]
   
# shorting constrain: sum of negative weights < 1
def con6(w):
    return 3-sum(abs(w))
    
a = optimize.fmin_cobyla(objective,init,[con1,con2,con3,con4,con5,con6],rhoend=1e-7,maxfun=1000)

print a

def utility(init_w, delta, blacklit_r, blacklit_sigma):
    def objective(w):
    
        return -np.dot(w,r)+0.5*delta*np.dot(np.dot(w,sigma),w)
    
    constraints = []
    for i in range(np.array(init_w).size):
        
        constraints.append(lambda w: 1 - w[i] )
        
    constraints.append(lambda w: 3-sum(abs(w)))            
    constraints.append(lambda w: sum(w)-1)        
    constraints.append(lambda w: -sum(w)+1)

    return optimize.fmin_cobyla(objective,init_w,constraints,rhoend=1e-7,maxfun=1000)