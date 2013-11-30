import numpy as np
from xlrd import open_workbook

asset = open_workbook('asset15feb.xlsx')
sheet = asset.sheet_by_index(0)

y = np.zeros(1005)
for j in range(2,62,4):
    a = []
    for i in range(9,1014):
        a.append(sheet.cell(i,j).value) #cell(1,0) = tweede rij, eerste kolom
        
    y = np.vstack((y,np.array(a)))
      
y = y[1:]  #remove the zeros (first row)
    
y = y.T #transpose

#print y.shape

yalsdet = np.zeros(1005)
for j in range(2,62,4):
    a = []
    for i in range(8,1013):
        a.append(sheet.cell(i,j).value) #cell(1,0) = tweede rij, eerste kolom
        
    yalsdet = np.vstack((yalsdet,np.array(a)))
      
yalsdet = yalsdet[1:]  #remove the zeros (first row)
    
yalsdet = yalsdet.T #transpose


det = open_workbook('keyind.xlsx')
belgium = det.sheet_by_index(0) #31 determinanten

q = np.zeros(79)
for j in range(1,12):
    lala = []
    for i in range(2,81):
        lala.append(belgium.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))

m = np.zeros(233)
for j in range(15,35):
    lala = []
    for i in range(5,238):
        lala.append(belgium.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))
    
    
denmark = det.sheet_by_index(1) #46 determinanten

tijdquart = []
for i in range(2,81):
    tijdquart.append(denmark.cell(i,53).value) 
    
tijdmaan = []
for i in range(2,235):
    tijdmaan.append(denmark.cell(i,54).value)   
    
    
for j in range(1,20):
    lala = []
    for i in range(2,81):
        lala.append(denmark.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(23,50):
    lala = []
    for i in range(5,238):
        lala.append(denmark.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))   
    
    
finland = det.sheet_by_index(2)

for j in range(1,14):
    lala = []
    for i in range(2,81):
        lala.append(finland.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(17,47):
    lala = []
    for i in range(5,238):
        lala.append(finland.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))     
    
france = det.sheet_by_index(3)

for j in range(1,21):
    lala = []
    for i in range(2,81):
        lala.append(france.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(24,53):
    lala = []
    for i in range(5,238):
        lala.append(france.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))      
    
germany = det.sheet_by_index(4)

for j in range(1,26):
    lala = []
    for i in range(2,81):
        lala.append(germany.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(29,78):
    lala = []
    for i in range(5,238):
        lala.append(germany.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))          
    
greece = det.sheet_by_index(5)


for j in range(1,18):
    lala = []
    for i in range(5,238):
        lala.append(greece.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))      
    
hungary = det.sheet_by_index(6)

for j in range(1,25):
    lala = []
    for i in range(5,238):
        lala.append(hungary.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))     
    
ireland = det.sheet_by_index(7)

for j in range(1,6):
    lala = []
    for i in range(2,81):
        lala.append(ireland.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(9,41):
    lala = []
    for i in range(5,238):
        lala.append(ireland.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))                 
                  
                         
netherlands = det.sheet_by_index(8)

for j in range(2,21):
    lala = []
    for i in range(2,81):
        lala.append(netherlands.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(25,74):
    lala = []
    for i in range(5,238):
        lala.append(netherlands.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))    
    
norway = det.sheet_by_index(9)

for j in range(1,35):
    lala = []
    for i in range(2,81):
        lala.append(norway.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(38,83):
    lala = []
    for i in range(5,238):
        lala.append(norway.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))                                                                            

portugal = det.sheet_by_index(10)

for j in range(1,8):
    lala = []
    for i in range(2,81):
        lala.append(portugal.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(11,38):
    lala = []
    for i in range(5,238):
        lala.append(portugal.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))  
                                               
spain = det.sheet_by_index(11)

for j in range(1,9):
    lala = []
    for i in range(2,81):
        lala.append(spain.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(12,50):
    lala = []
    for i in range(5,238):
        lala.append(spain.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))   
    
                                                                                                      
sweden = det.sheet_by_index(12)

for j in range(4,17):
    lala = []
    for i in range(2,81):
        lala.append(sweden.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(21,66):
    lala = []
    for i in range(5,238):
        lala.append(sweden.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))  
    
                                                                    
                                                                                                                                    
swiss = det.sheet_by_index(13)

for j in range(1,15):
    lala = []
    for i in range(2,81):
        lala.append(swiss.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(21,48):
    lala = []
    for i in range(5,238):
        lala.append(swiss.cell(i,j).value) 
    m = np.vstack((m,np.array(lala))) 
uk = det.sheet_by_index(14)

for j in range(1,26):
    lala = []
    for i in range(2,81):
        lala.append(uk.cell(i,j).value) 
    q = np.vstack((q,np.array(lala)))
    

for j in range(29,60):
    lala = []
    for i in range(5,238):
        lala.append(uk.cell(i,j).value) 
    m = np.vstack((m,np.array(lala)))          
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
kwartaaldata = q[1:] # remove zeros first row
maandata = m[1:]
  
com = open_workbook('commodities.xlsx')
sheet = com.sheet_by_index(0)

det = np.zeros(1006)
for j in range(1,43):
    lala = []
    for i in range(2,1008):
        lala.append((sheet.cell(i,j).value)) 
    det = np.vstack((det,np.array(lala)))
det = det[1:]


np.save('a',np.asarray(tijdquart)) #tijdsas voor kwartaal data
np.save('b',np.asarray(tijdmaan))

np.save('q',kwartaaldata)
np.save('m',maandata)
np.save('det',det)

print kwartaaldata.shape
print maandata.shape
print det.shape

np.save('y', y)
np.save('yalsdet', yalsdet)
