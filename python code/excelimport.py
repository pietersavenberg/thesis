import numpy as np
from xlrd import open_workbook
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


np.save('y', y)
np.save('x', x)