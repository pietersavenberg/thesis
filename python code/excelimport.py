import numpy as np
from xlrd import open_workbook
import Quandl


"""
det = open_workbook('keyind.xlsx')
belgium = det.sheet_by_index(0)


x = np.zeros(1057)
for j in range(39,101):
    a = []
    for i in range(6,1063):
        a.append(belgium.cell(i,j).value) #cell(1,0) = tweede rij, eerste kolom
        
    x = np.vstack((x,np.array(a)))


     
               
      
denmark = det.sheet_by_index(1)
  
for j in range(56,142):
    a = []
    for i in range(3,1060):
        a.append(denmark.cell(i,j).value) #cell(1,0) = tweede rij, eerste kolom
        
    x = np.vstack((x,np.array(a)))
     
        
            
            
x = x[1:]  #remove the zeros (first row)
    
x = x.T #transpose

print x.shape
#print x[0]



np.save('x', x)

"""


asset = open_workbook('asset15feb.xlsx')
sheet = asset.sheet_by_index(0)

y = np.zeros(1057)
for j in range(2,62,4):
    a = []
    for i in range(9,1066):
        a.append(sheet.cell(i,j).value) #cell(1,0) = tweede rij, eerste kolom
        
    y = np.vstack((y,np.array(a)))
      
y = y[1:]  #remove the zeros (first row)
    
y = y.T #transpose

print y.shape


det = open_workbook('keyind.xlsx')
belgium = det.sheet_by_index(0)

q = np.zeros(83)
for j in range(1,12):
    lala = []
    for i in range(2,85):
        lala.append(belgium.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))

m = np.zeros(245)
for j in range(15,35):
    lala = []
    for i in range(5,250):
        lala.append(belgium.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))
    
    
denmark = det.sheet_by_index(1)

tijdquart = []
for i in range(2,85):
    tijdquart.append(denmark.cell(i,53).value) 
    
tijdmaan = []
for i in range(2,247):
    tijdmaan.append(denmark.cell(i,54).value)   
    
    
for j in range(1,20):
    lala = []
    for i in range(2,85):
        lala.append(denmark.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(23,50):
    lala = []
    for i in range(5,250):
        lala.append(denmark.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))   
    
    
finland = det.sheet_by_index(2)

for j in range(1,14):
    lala = []
    for i in range(2,85):
        lala.append(finland.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(17,47):
    lala = []
    for i in range(5,250):
        lala.append(finland.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))     
    
france = det.sheet_by_index(3)

for j in range(1,21):
    lala = []
    for i in range(2,85):
        lala.append(france.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(24,53):
    lala = []
    for i in range(5,250):
        lala.append(france.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))      
    
germany = det.sheet_by_index(4)

for j in range(1,26):
    lala = []
    for i in range(2,85):
        lala.append(germany.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(29,78):
    lala = []
    for i in range(5,250):
        lala.append(germany.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))          
    
greece = det.sheet_by_index(5)


for j in range(1,18):
    lala = []
    for i in range(5,250):
        lala.append(greece.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))      
    
hungary = det.sheet_by_index(6)

for j in range(1,25):
    lala = []
    for i in range(5,250):
        lala.append(hungary.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))     
    
ireland = det.sheet_by_index(7)

for j in range(1,6):
    lala = []
    for i in range(2,85):
        lala.append(ireland.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(9,41):
    lala = []
    for i in range(5,250):
        lala.append(ireland.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))                 
                  
                         
netherlands = det.sheet_by_index(8)

for j in range(2,21):
    lala = []
    for i in range(2,85):
        lala.append(netherlands.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(25,74):
    lala = []
    for i in range(5,250):
        lala.append(netherlands.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))    
    
norway = det.sheet_by_index(9)

for j in range(1,35):
    lala = []
    for i in range(2,85):
        lala.append(norway.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(38,83):
    lala = []
    for i in range(5,250):
        lala.append(norway.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))                                                                            

portugal = det.sheet_by_index(10)

for j in range(1,8):
    lala = []
    for i in range(2,85):
        lala.append(portugal.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(11,38):
    lala = []
    for i in range(5,250):
        lala.append(portugal.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))  
                                               
spain = det.sheet_by_index(11)

for j in range(1,9):
    lala = []
    for i in range(2,85):
        lala.append(spain.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(12,50):
    lala = []
    for i in range(5,250):
        lala.append(spain.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))   
    
                                                                                                      
sweden = det.sheet_by_index(12)

for j in range(4,17):
    lala = []
    for i in range(2,85):
        lala.append(sweden.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(21,66):
    lala = []
    for i in range(5,250):
        lala.append(sweden.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))  
    
                                                                    
                                                                                                                                    
swiss = det.sheet_by_index(13)

for j in range(1,15):
    lala = []
    for i in range(2,85):
        lala.append(swiss.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(21,48):
    lala = []
    for i in range(5,250):
        lala.append(swiss.cell(i,j).value) 
    m = np.vstack((m,np.array(lala))) 
uk = det.sheet_by_index(14)

for j in range(1,26):
    lala = []
    for i in range(2,85):
        lala.append(uk.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(29,60):
    lala = []
    for i in range(5,250):
        lala.append(uk.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))          
                                                                                                                                                                                                                                                                             

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
kwartaaldata = q[1:] # remove zeros first row
maandata = m[1:]
  
#b verder aanvullen met kwartale det van de andere landen

#c of iets dergelijks aanmaken dat hetzelfde doet als b maar dan voor maandelijks  
np.save('a',np.asarray(tijdquart)) #tijdsas voor kwartaal data
np.save('b',np.asarray(tijdmaan))

np.save('q',kwartaaldata)
np.save('m',maandata)

np.save('y', y)


print maandata.shape
print kwartaaldata.shape



'''
#----------------------------------------------------------------------------


freq = "weekly"
st = "15/02/1993"
en = "20/05/2013"
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


print oil.shape, copper.shape, gold.shape, energy.shape, metals.shape, bill3mo.shape, bill2y.shape, bill10y.shape, billdif.shape
'''