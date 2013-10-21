import Quandl
import numpy as np
from pandas import *


bel20 = Quandl.get("YAHOO/INDEX_BFX",collapse="weekly",trim_start ="July 2011", trim_end="July 2013",transformation = "rdiff") #rdiff geeft %change weer, met andere woorden de return
cac40 = Quandl.get("YAHOO/INDEX_FCHI",collapse="weekly",trim_start ="July 2011", trim_end="July 2013", transformation = "rdiff") 
aex = Quandl.get("YAHOO/INDEX_AEX",collapse="weekly",trim_start ="July 2011", trim_end="July 2013", transformation = "rdiff") 


d = {'bel20' : bel20["Open"], 'cac40' : cac40["Open"], 'aex' : aex["Open"]}
df = DataFrame(d)
sigma = df.cov() #historische covariantiematrix
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

print bel20["Open"]