import Quandl
import numpy as np
from pandas import *
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
from data import *

def splittraintest(list,ratio):
    return list[:ratio*len(list)],list[ratio*len(list):]

#doel
BELtrain,BELtest = splittraintest(BEL,0.9)
PXtrain,PXtest = splittraintest(PX,0.9)
FINtrain,FINtest = splittraintest(FIN,0.9)
CACtrain,CACtest = splittraintest(CAC,0.9)
DAXtrain,DAXtest = splittraintest(DAX,0.9)
BUXtrain,BUXtest = splittraintest(BUX,0.9)
ITAtrain,ITAtest = splittraintest(ITA,0.9)
IBEXtrain,IBEXtest = splittraintest(IBEX,0.9)
SWEtrain,SWEtest = splittraintest(SWE,0.9)
SMItrain,SMItest = splittraintest(SMI,0.9)
UKtrain,UKtest = splittraintest(UK,0.9)
UKRtrain,UKRtest = splittraintest(UKR,0.9)
AEXtrain,AEXtest = splittraintest(AEX,0.9)



# 16 algemene det
SP500train,SP500test = splittraintest(SP500,0.9)
Niktrain,Niktest = splittraintest(Nik,0.9)
DJIAtrain, DJIAtest = splittraintest(DJIA,0.9)
cpitrain, cpitest =splittraintest(cpi,0.9)
defltrain, defltest = splittraintest(defl,0.9)
usleadtrain, usleadtest = splittraintest(uslead,0.9)
ccitrain, ccitest = splittraintest(cci,0.9)
bcitrain, bcitest = splittraintest(bci,0.9)
oiltrain, oiltest =splittraintest(oil,0.9)
coppertrain, coppertest = splittraintest(copper,0.9)
goldtrain, goldtest = splittraintest(gold,0.9)
energytrain, energytest = splittraintest(energy,0.9) 
metalstrain, metalstest = splittraintest(metals,0.9)
bill3motrain, bill3motest = splittraintest(bill3mo,0.9)
bill2ytrain, bill2ytest = splittraintest(bill2y,0.9)
bill10ytrain, bill10ytest = splittraintest(bill10y,0.9)
billdiftrain,billdiftest = splittraintest(billdif,0.9)


# 10 nationaal det
BELlagtrain, BELlagtest = splittraintest(BELlag,0.9)
PXlagtrain, PXlagtest = splittraintest(PXlag,0.9)
FINlagtrain, FINlagtest = splittraintest(FINlag,0.9)
CAClagtrain, CAClagtest = splittraintest(CAClag,0.9)
DAXlagtrain, DAXlagtest = splittraintest(DAXlag,0.9)
BUXlagtrain, BUXlagtest = splittraintest(BUXlag,0.9)
ITAlagtrain, ITAlagtest = splittraintest(ITAlag,0.9)
IBEXlagtrain, IBEXlagtest = splittraintest(IBEXlag,0.9)
SWElagtrain, SWElagtest = splittraintest(SWElag,0.9)
SMIlagtrain, SMIlagtest = splittraintest(SMIlag,0.9)
UKlagtrain, UKlagtest = splittraintest(UKlag,0.9)
UKRlagtrain, UKRlagtest = splittraintest(UKRlag,0.9)
AEXlagtrain, AEXlagtest = splittraintest(AEXlag,0.9)


cpibeltrain, cpibeltest = splittraintest(cpibel,0.9)
cpipxtrain, cpipxtest = splittraintest(cpipx,0.9)
cpifintrain, cpifintest = splittraintest(cpifin,0.9)
cpicactrain, cpicactest = splittraintest(cpicac,0.9)
cpidaxtrain, cpidaxtest = splittraintest(cpidax,0.9)
cpibuxtrain, cpibuxtest = splittraintest(cpibux,0.9)
cpiitatrain, cpiitatest = splittraintest(cpiita,0.9)
cpiibextrain, cpiibextest = splittraintest(cpiibex,0.9)
cpiswetrain, cpiswetest = splittraintest(cpiswe,0.9)
cpismitrain, cpismitest = splittraintest(cpismi,0.9)
cpiuktrain, cpiuktest = splittraintest(cpiuk,0.9)
cpiukrtrain, cpiukrtest = splittraintest(cpiukr,0.9)
cpiaextrain, cpiaextest = splittraintest(cpiaex,0.9)


indbeltrain, indbeltest = splittraintest(indbel,0.9)

empbeltrain, empbeltest = splittraintest(empbel,0.9)
clibeltrain, clibeltest = splittraintest(clibel,0.9)
neerbeltrain, neerbeltest = splittraintest(neerbel,0.9)
reerbeltrain, reerbeltest  = splittraintest(reerbel,0.9)
debtbeltrain, debtbeltest = splittraintest(debtbel,0.9)
gdpbeltrain, gdpbeltest  = splittraintest(gdpbel,0.9)
monbeltrain, monbeltest =  splittraintest(monbel,0.9)

'''
print SP500train.shape, Niktrain.shape, DJIAtrain.shape,cpitrain.shape,defltrain.shape, usleadtrain.shape,ccitrain.shape,bcitrain.shape
print oiltrain.shape,coppertrain.shape,goldtrain.shape,energytrain.shape, metalstrain.shape, bill3motrain.shape, bill2ytrain.shape, bill10ytrain.shape
print BELlagtest.shape, cpibeltest.shape,indbeltest.shape, empbeltest.shape, clibeltest.shape,neerbeltest.shape,reerbeltest.shape, debtbeltest.shape
print gdpbeltest.shape, monbeltest.shape
'''

#BEL20 prediction
X = np.vstack((SP500train, Niktrain, DJIAtrain, cpitrain, defltrain, usleadtrain, ccitrain, bcitrain, oiltrain, coppertrain, goldtrain, energytrain, metalstrain, 
bill3motrain, bill2ytrain, bill10ytrain, billdiftrain,
BELlagtrain, PXlagtrain, FINlagtrain,CAClagtrain,DAXlagtrain,BUXlagtrain,ITAlagtrain, IBEXlagtrain,SWElagtrain,SMIlagtrain,UKlagtrain,UKRlagtrain, AEXlagtrain,
cpibeltrain, cpipxtrain, cpifintrain,cpicactrain,cpidaxtrain,cpibuxtrain,cpiitatrain,cpiibextrain,cpiswetrain,cpismitrain,cpiuktrain, cpiukrtrain,cpiaextrain,
indbeltrain, indpxtrain,indfintrain,indcactrain,inddaxtrain,indbuxtrain,inditatrain,indibextrain,indswetrain,indsmitrain,induktrain,indukrtrain,indaextrain,
empbeltrain, 
clibeltrain, 
neerbeltrain, 
reerbeltrain, 
debtbeltrain, 
gdpbeltrain, 
monbeltrain)).T
y = np.vstack((BELtrain,PXtrain,FINtrain,CACtrain,DAXtrain,BUXtrain,ITAtrain,IBEXtrain,SWEtrain,SMItrain,UKtrain,UKRtrain,AEXtrain)).T
'''
BELtrain,BELtest = splittraintest(BEL,0.9)
PXtrain,PXtest = splittraintest(PX,0.9)
FINtrain,FINtest = splittraintest(FIN,0.9)
CACtrain,CACtest = splittraintest(CAC,0.9)
DAXtrain,DAXtest = splittraintest(DAX,0.9)
BUXtrain,BUXtest = splittraintest(BUX,0.9)
ITAtrain,ITAtest = splittraintest(ITA,0.9)
IBEXtrain,IBEXtest = splittraintest(IBEX,0.9)
SWEtrain,SWEtest = splittraintest(SWE,0.9)
SMItrain,SMItest = splittraintest(SMI,0.9)
UKtrain,UKtest = splittraintest(UK,0.9)
UKRtrain,UKRtest = splittraintest(UKR,0.9)
AEXtrain,AEXtest = splittraintest(AEX,0.9)
'''

x = np.vstack((SP500test, Niktest, DJIAtest, cpitest, defltest, usleadtest, ccitest, bcitest, oiltest, coppertest, goldtest, energytest, metalstest, 
bill3motest, bill2ytest, bill10ytest, 
BELlagtest, cpibeltest, indbeltest, empbeltest, clibeltest, neerbeltest, reerbeltest, debtbeltest, gdpbeltest, monbeltest)).T

'''

theta0 = []
for i in range( X.shape[1]):
    theta0.append(1e-2)
theta0 = np.array(theta0)    

thetaL = []
for i in range( X.shape[1]):
    thetaL.append(1e-4)
thetaL = np.array(thetaL)  

thetaU = []
for i in range( X.shape[1]):
    thetaU.append(10.)
thetaU = np.array(thetaU)  

print x
'''

gp = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4,thetaU=1.,normalize = True, random_start=10)
gp.fit(X, y)
y_pred, MSE = gp.predict(x, eval_MSE=True)


