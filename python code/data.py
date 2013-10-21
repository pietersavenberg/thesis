import Quandl
import numpy as np
from pandas import *

#assets

start = "December 2001"
end = "December 2012"
freq = "monthly"

#ik moet telkens het laatste object van de array verwijderen aangezien Quandl niet overweg kan met een eindmaand van November.
BEL = np.delete(np.array(Quandl.get("YAHOO/INDEX_BFX",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"]),132)
PX = np.delete(np.array(Quandl.get("PRAGUESE/PX",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Index"]),132)
FIN = np.delete(np.array(Quandl.get("NASOMXNORDIC/FI0008900212",collapse= freq,trim_start = start, trim_end= end,transformation = "rdiff")["Closing Price"]),132)
CAC = np.delete(np.array(Quandl.get("YAHOO/INDEX_FCHI",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"]),132)
DAX = np.delete(np.array(Quandl.get("YAHOO/INDEX_GDAXI",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"]),132)
BUX = np.delete(np.array(Quandl.get("BUDAPESTSE/BUX",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"]),132)
ITA = np.delete(np.array(Quandl.get("YAHOO/INDEX_FTSEMIB_MI",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"]),132)
IBEX = np.delete(np.array(Quandl.get("YAHOO/INDEX_IBEX",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"]),132)
SWE = np.delete(np.array(Quandl.get("NASOMXNORDIC/SE0000337842",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Closing Price"]),132)
SMI = np.delete(np.array(Quandl.get("YAHOO/INDEX_SSMI",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"]),132)
UK = np.delete(np.array(Quandl.get("YAHOO/INDEX_FTSE",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"]),132)
UKR = np.delete(np.array(Quandl.get("PFTS/INDEX",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Index"]),132)
AEX = np.delete(np.array(Quandl.get("YAHOO/INDEX_AEX",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"]),132)

#determinants, de basisset voor voorspellingen

#technische determinanten:
start = "November 2001"
end = "November 2012"
freq = "monthly"
BELlag = np.array(Quandl.get("YAHOO/INDEX_BFX",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"])
PXlag = np.array(Quandl.get("PRAGUESE/PX",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Index"])
FINlag = np.array(Quandl.get("NASOMXNORDIC/FI0008900212",collapse= freq,trim_start = start, trim_end= end,transformation = "rdiff")["Closing Price"])
CAClag = np.array(Quandl.get("YAHOO/INDEX_FCHI",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"])
DAXlag = np.array(Quandl.get("YAHOO/INDEX_GDAXI",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"])
BUXlag = np.array(Quandl.get("BUDAPESTSE/BUX",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"])
ITAlag = np.array(Quandl.get("YAHOO/INDEX_FTSEMIB_MI",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"])
IBEXlag = np.array(Quandl.get("YAHOO/INDEX_IBEX",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"])
SWElag = np.array(Quandl.get("NASOMXNORDIC/SE0000337842",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Closing Price"])
SMIlag = np.array(Quandl.get("YAHOO/INDEX_SSMI",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"])
UKlag = np.array(Quandl.get("YAHOO/INDEX_FTSE",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"])
UKRlag = np.array(Quandl.get("PFTS/INDEX",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Index"])
AEXlag = np.array(Quandl.get("YAHOO/INDEX_AEX",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"])

SP500 = np.array(Quandl.get("YAHOO/INDEX_GSPC",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"])
Nik = np.array(Quandl.get("YAHOO/INDEX_N225",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"])
DJIA = np.array(Quandl.get("BCB/UDJIAD1",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Value"])


#fundamentele determinanten
st = "December 2001"
en = "November 2012"
freq = "monthly"

def jaarlijks(arr):
    new = []
    for i in range(arr.size):
        for j in range(12):
            new.append(arr[i])
    return np.array(new)
    
def kwartaal(arr):
    new = []
    for i in range(arr.size):
        for j in range(3):
            new.append(arr[i])
    return np.array(new)
    
    

#inflation related: 
#global inflation based on CPI, 
cpi = jaarlijks(np.array(Quandl.get("WORLDBANK/WLD_FP_CPI_TOTL_ZG",collapse=freq,trim_start = st, trim_end= en)["Value"]))
#misschien moet collapse monthly hier weggelaten worden?

#global inflation based on GDP deflator
defl = jaarlijks(np.array(Quandl.get("WORLDBANK/WLD_NY_GDP_DEFL_KD_ZG",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"]))


#CPI per country
cpibel = jaarlijks(np.array(Quandl.get("WORLDBANK/BEL_CPTOTSAXNZGY",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"]))
cpipx = jaarlijks(np.array(Quandl.get("WORLDBANK/CZE_CPTOTSAXNZGY",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"]))
cpifin = jaarlijks(np.array(Quandl.get("WORLDBANK/FIN_CPTOTSAXNZGY",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"]))
cpicac = jaarlijks(np.array(Quandl.get("WORLDBANK/FRA_CPTOTSAXNZGY",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"]))
cpidax = jaarlijks(np.array(Quandl.get("WORLDBANK/DEU_CPTOTSAXNZGY",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"]))
cpibux = jaarlijks(np.array(Quandl.get("WORLDBANK/HUN_CPTOTSAXNZGY",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"]))
cpiita = jaarlijks(np.array(Quandl.get("WORLDBANK/ITA_CPTOTSAXNZGY",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"]))
cpiibex = jaarlijks(np.array(Quandl.get("WORLDBANK/ESP_CPTOTSAXNZGY",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"]))
cpiswe = jaarlijks(np.array(Quandl.get("WORLDBANK/SWE_CPTOTSAXNZGY",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"]))
cpismi = jaarlijks(np.array(Quandl.get("WORLDBANK/CHE_CPTOTSAXNZGY",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"]))
cpiuk = jaarlijks(np.array(Quandl.get("WORLDBANK/GBR_CPTOTSAXNZGY",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"]))
cpiukr = jaarlijks(np.array(Quandl.get("WORLDBANK/UKR_CPTOTSAXNZGY",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"]))
cpiaex = jaarlijks(np.array(Quandl.get("WORLDBANK/NLD_CPTOTSAXNZGY",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"]))
#transformation rdiff weglaten?

#industrial production per country
indbel = jaarlijks(np.array(Quandl.get("WORLDBANK/BEL_IPTOTSAKD",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
indpx = jaarlijks(np.array(Quandl.get("WORLDBANK/CZE_IPTOTSAKD",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
indfin = jaarlijks(np.array(Quandl.get("WORLDBANK/FIN_IPTOTSAKD",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
indcac = jaarlijks(np.array(Quandl.get("WORLDBANK/FRA_IPTOTSAKD",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
inddax = jaarlijks(np.array(Quandl.get("WORLDBANK/DEU_IPTOTSAKD",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
indbux = jaarlijks(np.array(Quandl.get("WORLDBANK/HUN_IPTOTSAKD",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
indita = jaarlijks(np.array(Quandl.get("WORLDBANK/ITA_IPTOTSAKD",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
indibex = jaarlijks(np.array(Quandl.get("WORLDBANK/ESP_IPTOTSAKD",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
indswe = jaarlijks(np.array(Quandl.get("WORLDBANK/SWE_IPTOTSAKD",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
indsmi = jaarlijks(np.array(Quandl.get("WORLDBANK/CHE_IPTOTSAKD",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
induk = jaarlijks(np.array(Quandl.get("WORLDBANK/GBR_IPTOTSAKD",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
indukr = jaarlijks(np.array(Quandl.get("WORLDBANK/UKR_IPTOTSAKD",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
indaex = jaarlijks(np.array(Quandl.get("WORLDBANK/NLD_IPTOTSAKD",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))



#unemployment rate per country
empbel = jaarlijks(np.array(Quandl.get("IMF/MAP_WEO_UNEMP_BEL",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
emppx = jaarlijks(np.array(Quandl.get("IMF/MAP_WEO_UNEMP_CZE",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
empfin = jaarlijks(np.array(Quandl.get("IMF/MAP_WEO_UNEMP_FIN",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
empcac = jaarlijks(np.array(Quandl.get("IMF/MAP_WEO_UNEMP_FRA",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
empdax = jaarlijks(np.array(Quandl.get("IMF/MAP_WEO_UNEMP_DEU",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
empbux = jaarlijks(np.array(Quandl.get("IMF/MAP_WEO_UNEMP_HUN",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
empita = jaarlijks(np.array(Quandl.get("IMF/MAP_WEO_UNEMP_ITA",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
empibex = jaarlijks(np.array(Quandl.get("IMF/MAP_WEO_UNEMP_ESP",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
empswe = jaarlijks(np.array(Quandl.get("IMF/MAP_WEO_UNEMP_SWE",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
empsmi = jaarlijks(np.array(Quandl.get("IMF/MAP_WEO_UNEMP_CHE",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
empuk = jaarlijks(np.array(Quandl.get("IMF/MAP_WEO_UNEMP_GBR",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
empukr = jaarlijks(np.array(Quandl.get("IMF/MAP_WEO_UNEMP_UKR",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
empaex = jaarlijks(np.array(Quandl.get("IMF/MAP_WEO_UNEMP_NLD",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))


#composite leading indicator per country (Ukraine unavailable)
clibel = np.array(Quandl.get("OECD/MEI_CLI_LOLITOAA_BEL_M",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
clipx = np.array(Quandl.get("OECD/MEI_CLI_LOLITOAA_CZE_M",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
clifin = np.array(Quandl.get("OECD/MEI_CLI_LOLITOAA_FIN_M",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
clicac = np.array(Quandl.get("OECD/MEI_CLI_LOLITOAA_FRA_M",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
clidax = np.array(Quandl.get("OECD/MEI_CLI_LOLITOAA_DEU_M",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
clibux = np.array(Quandl.get("OECD/MEI_CLI_LOLITOAA_HUN_M",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
cliita = np.array(Quandl.get("OECD/MEI_CLI_LOLITOAA_ITA_M",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
cliibex = np.array(Quandl.get("OECD/MEI_CLI_LOLITOAA_ESP_M",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
cliswe = np.array(Quandl.get("OECD/MEI_CLI_LOLITOAA_SWE_M",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
clismi = np.array(Quandl.get("OECD/MEI_CLI_LOLITOAA_CHE_M",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
cliuk = np.array(Quandl.get("OECD/MEI_CLI_LOLITOAA_GBR_M",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
#cliukr = np.array(Quandl.get("OECD/MEI_CLI_LOLITOAA_UKR_M",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
cliaex = np.array(Quandl.get("OECD/MEI_CLI_LOLITOAA_NLD_M",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])


#USA leading indicator
uslead = np.array(Quandl.get("ECRI/USLEADING",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Level"])

#Consumer confidence index
cci = np.array(Quandl.get("BCB/4393",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])

#Business confidence index
bci = kwartaal(np.array(Quandl.get("BCB/7341",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))


#Nominal Effective Exchange Rate per country
neerbel = jaarlijks(np.array(Quandl.get("WORLDBANK/BEL_NEER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
neerpx = np.array(Quandl.get("WORLDBANK/CZE_NEER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
neerfin = np.array(Quandl.get("WORLDBANK/FIN_NEER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
neercac = np.array(Quandl.get("WORLDBANK/FRA_NEER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
neerdax = np.array(Quandl.get("WORLDBANK/DEU_NEER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
neerbux = np.array(Quandl.get("WORLDBANK/HUN_NEER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
neerita = np.array(Quandl.get("WORLDBANK/ITA_NEER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
neeribex = np.array(Quandl.get("WORLDBANK/ESP_NEER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
neerswe = np.array(Quandl.get("WORLDBANK/SWE_NEER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
neersmi = np.array(Quandl.get("WORLDBANK/CHE_NEER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
neeruk = np.array(Quandl.get("WORLDBANK/GBR_NEER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
neerukr = np.array(Quandl.get("WORLDBANK/UKR_NEER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
neeraex = np.array(Quandl.get("WORLDBANK/NLD_NEER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])

#Real Effective Exchange Rate per country
reerbel = jaarlijks(np.array(Quandl.get("WORLDBANK/BEL_REER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
reerpx = np.array(Quandl.get("WORLDBANK/CZE_REER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
reerfin = np.array(Quandl.get("WORLDBANK/FIN_REER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
reercac = np.array(Quandl.get("WORLDBANK/FRA_REER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
reerdax = np.array(Quandl.get("WORLDBANK/DEU_REER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
reerbux = np.array(Quandl.get("WORLDBANK/HUN_REER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
reerita = np.array(Quandl.get("WORLDBANK/ITA_REER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
reeribex = np.array(Quandl.get("WORLDBANK/ESP_REER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
reerswe = np.array(Quandl.get("WORLDBANK/SWE_REER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
reersmi = np.array(Quandl.get("WORLDBANK/CHE_REER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
reeruk = np.array(Quandl.get("WORLDBANK/GBR_REER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
reerukr = np.array(Quandl.get("WORLDBANK/UKR_REER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
reeraex = np.array(Quandl.get("WORLDBANK/NLD_REER",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])



#General government net debt, Percent of GDP, per country, Czech rep. unavailable
debtbel = jaarlijks(np.array(Quandl.get("IMF/GGXWDN_NGDP_124",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
#debtpx = np.array(Quandl.get("IMF/GGXWDN_NGDP_935",collapse=freq,trim_start = st, trim_end= en,transformation = "none")["Value"])
debtfin = np.array(Quandl.get("IMF/GGXWDN_NGDP_172",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
debtcac = np.array(Quandl.get("IMF/GGXWDN_NGDP_132",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
debtdax = np.array(Quandl.get("IMF/GGXWDN_NGDP_134",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
debtbux = np.array(Quandl.get("IMF/GGXWDN_NGDP_944",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
debtita = np.array(Quandl.get("IMF/GGXWDN_NGDP_136",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
debtibex = np.array(Quandl.get("IMF/GGXWDN_NGDP_184",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
debtswe = np.array(Quandl.get("IMF/GGXWDN_NGDP_144",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
debtsmi = np.array(Quandl.get("IMF/GGXWDN_NGDP_146",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
debtuk = np.array(Quandl.get("IMF/GGXWDN_NGDP_112",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
debtukr = np.array(Quandl.get("IMF/GGXWDN_NGDP_926",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
debtaex = np.array(Quandl.get("IMF/GGXWDN_NGDP_138",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])


#Real GDP per country
gdpbel = jaarlijks(np.array(Quandl.get("IMF/NGDP_R_124",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
gdppx = np.array(Quandl.get("IMF/NGDP_R_935",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
gdpfin = np.array(Quandl.get("IMF/NGDP_R_172",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
gdpcac = np.array(Quandl.get("IMF/NGDP_R_132",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
gdpdax = np.array(Quandl.get("IMF/NGDP_R_134",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
gdpbux = np.array(Quandl.get("IMF/NGDP_R_944",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
gdpita = np.array(Quandl.get("IMF/NGDP_R_136",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
gdpibex = np.array(Quandl.get("IMF/NGDP_R_184",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
gdpswe = np.array(Quandl.get("IMF/NGDP_R_144",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
gdpsmi = np.array(Quandl.get("IMF/NGDP_R_146",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
gdpuk = np.array(Quandl.get("IMF/NGDP_R_112",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
gdpukr = np.array(Quandl.get("IMF/NGDP_R_926",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
gdpaex = np.array(Quandl.get("IMF/NGDP_R_138",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])


#M2 Money Supply per country
monbel = jaarlijks(np.array(Quandl.get("WORLDBANK/BEL_FM_LBL_MQMY_GD_ZS",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"]))
monpx = np.array(Quandl.get("WORLDBANK/CZE_FM_LBL_MQMY_GD_ZS",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
monfin = np.array(Quandl.get("WORLDBANK/FIN_FM_LBL_MQMY_GD_ZS",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
moncac = np.array(Quandl.get("WORLDBANK/FRA_FM_LBL_MQMY_GD_ZS",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
mondax = np.array(Quandl.get("WORLDBANK/DEU_FM_LBL_MQMY_GD_ZS",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
monbux = np.array(Quandl.get("WORLDBANK/HUN_FM_LBL_MQMY_GD_ZS",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
monita = np.array(Quandl.get("WORLDBANK/ITA_FM_LBL_MQMY_GD_ZS",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
monibex = np.array(Quandl.get("WORLDBANK/ESP_FM_LBL_MQMY_GD_ZS",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
monswe = np.array(Quandl.get("WORLDBANK/SWE_FM_LBL_MQMY_GD_ZS",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
monsmi = np.array(Quandl.get("WORLDBANK/CHE_FM_LBL_MQMY_GD_ZS",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
monuk = np.array(Quandl.get("WORLDBANK/GBR_FM_LBL_MQMY_GD_ZS",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
monukr = np.array(Quandl.get("WORLDBANK/UKR_FM_LBL_MQMY_GD_ZS",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])
monaex = np.array(Quandl.get("WORLDBANK/NLD_FM_LBL_MQMY_GD_ZS",collapse=freq,trim_start = st, trim_end= en,transformation = "rdiff")["Value"])

#Europe Brent oil spot price
oil = np.array(Quandl.get("DOE/RBRTE",collapse=freq,trim_start = st, trim_end= en,transformation= "rdiff")["Value"])

#copper price
copper = np.array(Quandl.get("WORLDBANK/WLD_COPPER",collapse=freq,trim_start = st, trim_end= en,transformation= "rdiff")["Value"])

#gold price
gold = np.array(Quandl.get("BUNDESBANK/BBK01_WT5511",collapse=freq,trim_start = st, trim_end= en,transformation= "rdiff")["Value"])

#energy index
energy = np.array(Quandl.get("WORLDBANK/WLD_IENERGY",collapse=freq,trim_start = st, trim_end= en,transformation= "rdiff")["Value"])

#metals and minerals index
metals = np.array(Quandl.get("WORLDBANK/WLD_IMETMIN",collapse=freq,trim_start = st, trim_end= en,transformation= "rdiff")["Value"])

#treasury bill yields
bill3mo = np.array(Quandl.get("USTREASURY/YIELD",collapse=freq,trim_start = st, trim_end= en)["3 Mo"])
bill2y = np.array(Quandl.get("USTREASURY/YIELD",collapse=freq,trim_start = st, trim_end= en)["2 Yr"])
bill10y = np.array(Quandl.get("USTREASURY/YIELD",collapse=freq,trim_start = st, trim_end= en)["10 Yr"])
billdif = bill10y-bill2y

