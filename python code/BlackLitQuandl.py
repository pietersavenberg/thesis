# Black-Litterman example code for python (hl.py)
# Copyright (c) Jay Walters, blacklitterman.org, 2012.
#
# Redistribution and use in source and binary forms, 
# with or without modification, are permitted provided 
# that the following conditions are met:
#
# Redistributions of source code must retain the above 
# copyright notice, this list of conditions and the following 
# disclaimer.
# 
# Redistributions in binary form must reproduce the above 
# copyright notice, this list of conditions and the following 
# disclaimer in the documentation and/or other materials 
# provided with the distribution.
#  
# Neither the name of blacklitterman.org nor the names of its
# contributors may be used to endorse or promote products 
# derived from this software without specific prior written
# permission.
#  
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND 
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR 
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH 
# DAMAGE.
#
# This program uses the examples from the paper "The Intuition 
# Behind Black-Litterman Model  Portfolios", by He and Litterman,
# 1999.  You can find a copy of this  paper at the following url.
#     http://papers.ssrn.com/sol3/papers.cfm?abstract_id=334304
#
# For more details on the Black-Litterman model you can also view
# "The BlackLitterman Model: A Detailed Exploration", by this author
# at the following url.
#     http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1314585
#

import Quandl
import numpy as np
from pandas import *
from scipy import linalg
from sklearn.gaussian_process import GaussianProcess



# blacklitterman
#   This function performs the Black-Litterman blending of the prior
#   and the views into a new posterior estimate of the returns as
#   described in the paper by He and Litterman.
# Inputs
#   delta  - Risk tolerance from the equilibrium portfolio
#   weq    - Weights of the assets in the equilibrium portfolio
#   sigma  - Prior covariance matrix
#   tau    - Coefficiet of uncertainty in the prior estimate of the mean (pi)
#   P      - Pick matrix for the view(s)
#   Q      - Vector of view returns
#   Omega  - Matrix of variance of the views (diagonal)
# Outputs
#   Er     - Posterior estimate of the mean returns
#   w      - Unconstrained weights computed given the Posterior estimates
#            of the mean and covariance of returns.
#   lambda - A measure of the impact of each view on the posterior estimates.
#
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
  w = er.T.dot(linalg.inv(delta * posteriorSigma)).T
  # Compute lambda value
  # We solve for lambda from formula (17) page 7, rather than formula (18)
  # just because it is less to type, and we've already computed w*.
  lmbda = np.dot(linalg.pinv(P).T,(w.T * (1 + tau) - weq).T)
  #return [er, w, lmbda]
  return w

# Function to display the results of a black-litterman shrinkage
# Inputs
#   title	- Displayed at top of output
#   assets	- List of assets
#   res		- List of results structures from the bl function
#
def display(title,assets,res):
  er = res[0]
  w = res[1]
  lmbda = res[2]
  print('\n' + title)
  line = 'Country\t\t'
  for p in range(len(P)):
	line = line + 'P' + str(p) + '\t'
  line = line + 'mu\tw*'
  print(line)

  i = 0;
  for x in assets:
	line = '{0}\t'.format(x)
	for j in range(len(P.T[i])):
		line = line + '{0:.1f}\t'.format(100*P.T[i][j])

	line = line + '{0:.3f}\t{1:.3f}'.format(100*er[i][0],100*w[i][0])
	print(line)
        i = i + 1

  line = 'q\t\t'
  i = 0
  for q in Q:
    line = line + '{0:.2f}\t'.format(100*q[0])
    i = i + 1
  print(line)

  line = 'omega/tau\t'
  i = 0
  for o in Omega:
	line = line + '{0:.5f}\t'.format(o[i]/tau)
	i = i + 1
  print(line)

  line = 'lambda\t\t'
  i = 0
  for l in lmbda:
	line = line + '{0:.5f}\t'.format(l[0])
	i = i + 1
  print(line)


def splittraintest(list,ratio):
    return list[:ratio*len(list)],list[ratio*len(list):]
    
#doel
bel20 = np.array(Quandl.get("YAHOO/INDEX_BFX",collapse="monthly",trim_start ="July 2009", trim_end="July 2013",transformation = "rdiff")["Close"])
cac40 = np.array(Quandl.get("YAHOO/INDEX_FCHI",collapse="monthly",trim_start ="July 2009", trim_end="July 2013", transformation = "rdiff")["Close"])
aex = np.array(Quandl.get("YAHOO/INDEX_AEX",collapse="monthly",trim_start ="July 2009", trim_end="July 2013", transformation = "rdiff")["Close"])


#determinanten
koers = np.array(Quandl.get("YAHOO/INDEX_BFX",collapse="monthly",trim_start ="June 2009", trim_end="June 2013")["Close"])
rendement = np.array(Quandl.get("YAHOO/INDEX_BFX",collapse="monthly",trim_start ="May 2009", trim_end="June 2013",transformation = "rdiff")["Close"])
oil = np.array(Quandl.get("DOE/RBRTE",collapse="monthly",trim_start ="June 2009", trim_end="June 2013")["Value"])
bill = np.array(Quandl.get("USTREASURY/BILLRATES",collapse="monthly",trim_start ="June 2009", trim_end="June 2013")["13 Weeks Coupon Equivalent"])
cci = np.array(Quandl.get("BCB/4393",collapse="monthly",trim_start ="June 2009", trim_end="June 2013")["Value"])
koper = np.array(Quandl.get("IMF/PCOPP_USD",collapse="monthly",trim_start ="June 2009", trim_end="June 2013")["Value"])
goud = np.array(Quandl.get("BUNDESBANK/BBK01_WT5511",collapse="monthly",trim_start ="June 2009", trim_end="June 2013")["Value"])
'''
normalize(bel20), normalize(koers), normalize(rendement), normalize(oil), normalize(bill), normalize(cci), normalize(koper), normalize(goud)
moet in principe niet omdat sk learn dat al doet tijdens de fitting
'''
aextrain,aextest = splittraintest(aex,0.8)
cac40train,cac40test = splittraintest(cac40,0.8)
bel20train,bel20test = splittraintest(bel20,0.8)
oiltrain,oiltest = splittraintest(oil,0.8)
billtrain,billtest = splittraintest(bill,0.8)
koerstrain,koerstest = splittraintest(koers,0.8)
rendementtrain,rendementtest = splittraintest(rendement,0.8)
ccitrain,ccitest = splittraintest(cci,0.8)
kopertrain,kopertest = splittraintest(koper,0.8)
goudtrain,goudtest = splittraintest(goud,0.8)


X = np.vstack((oiltrain,billtrain,koerstrain,rendementtrain,ccitrain,kopertrain,goudtrain)).T
x = np.vstack((oiltest,billtest,koerstest,rendementtest,ccitest,kopertest,goudtest)).T

gp = GaussianProcess( random_start=1000)
gp.fit(X, bel20train)
belpred, belMSE = gp.predict(x, eval_MSE=True)

gp.fit(X,cac40train)
cacpred, cacMSE = gp.predict(x, eval_MSE=True)

gp.fit(X,aextrain)
aexpred,aexMSE = gp.predict(x, eval_MSE=True) #moet iteratief

'''
bel20 = Quandl.get("YAHOO/INDEX_BFX",collapse="weekly",trim_start ="July 2011", trim_end="July 2013",transformation = "rdiff") #rdiff geeft %change weer, met andere woorden de return
cac40 = Quandl.get("YAHOO/INDEX_FCHI",collapse="weekly",trim_start ="July 2011", trim_end="July 2013", transformation = "rdiff") 
aex = Quandl.get("YAHOO/INDEX_AEX",collapse="weekly",trim_start ="July 2011", trim_end="July 2013", transformation = "rdiff") 
'''

d = {'bel20' : bel20train, 'cac40' : cac40train, 'aex' : aextrain}
df = DataFrame(d)
sigma = df.cov()
sigma = np.array(sigma) #maak van een gewone lijst of matrix eentje die compatibel is met numpy


divisorbel20 = 25180438.8125485
divisorcac40 = 191695324.297767
divisoraex = 762621789.602243

koersbel20 = Quandl.get("YAHOO/INDEX_BFX",collapse="weekly",trim_start ="2013-07-28")["Open"][0]
koerscac40 = Quandl.get("YAHOO/INDEX_FCHI",collapse="weekly",trim_start ="2013-07-28")["Open"][0]
koersaex = Quandl.get("YAHOO/INDEX_AEX",collapse="weekly",trim_start ="2013-07-28")["Open"][0]

MPbel20 = koersbel20*divisorbel20 #MP = marketcap
MPcac40 = koerscac40*divisorcac40
MPaex = koersaex*divisoraex

somMP= MPbel20 + MPcac40 + MPaex
weq = [MPbel20/somMP,MPcac40/somMP,MPaex/somMP]
weq = np.array(weq)

# Risk aversion of the market 
delta = 2.5

# Coefficient of uncertainty in the prior estimate of the mean
# from footnote (8) on page 11
tau = 0.01
tausigma = tau * sigma
assets= {'bel20   ','cac40   ','aex       '}

# Define view 1
list = []
brutorend = 0.0
for i in range(len(belpred)):
    
    P = np.array([[1,0,0],[0,1,0],[0,0,1]])
    Q = np.array([[belpred[i]],[cacpred[i]],[aexpred[i]]])
    Omega = np.array([[belMSE[i],0,0],[0,cacMSE[i],0],[0,0,aexMSE[i]]])
    res = blacklitterman(delta, weq, sigma, tau, P, Q, Omega)
    brutorend += res[0][0]*bel20test[i] + res[1][0]*cac40test[i] + res[2][0]*aextest[i] 
print brutorend
