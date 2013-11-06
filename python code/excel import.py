import numpy as np
from xlrd import open_workbook
book = open_workbook('asset15feb.xlsx')
sheet = book.sheet_by_index(0)

y = np.zeros(1057)
for j in range(2,62,4):
    a = []
    for i in range(9,1066):
        a.append(sheet.cell(i,j).value) #cell(1,0) = tweede rij, eerste kolom
        
    y = np.vstack((y,np.array(a)))
      
y = y[1:]  #remove the zeros (first row)
    
y = y.T #transpose

book = open_workbook('keyind.xlsx')
sheet = book.sheet_by_index(0)

x = np.zeros(1057)
for j in range(2,13):
    a = []
    for i in range(91,1148):
        a.append(sheet.cell(i,j).value) #cell(1,0) = tweede rij, eerste kolom
        
    x = np.vstack((x,np.array(a)))
      
x = x[1:]  #remove the zeros (first row)
    
x = x.T #transpose

    