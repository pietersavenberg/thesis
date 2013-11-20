import matplotlib.pyplot as plt
from matplotlib  import cm
import numpy as np

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.set_title("X vs Y AVG",fontsize=14)
ax.set_xlabel("XAVG",fontsize=12)
ax.set_ylabel("YAVG",fontsize=12)
ax.grid(True,linestyle='-',color='0.75')
x = np.random.random(30)
y = np.random.random(30)
z = np.random.random(30)/1000

# scatter with colormap mapping to z value
print x,y,z
ax.scatter(x,y,s=40,c=z, marker = 'o', cmap = cm.jet );
plt.colobar()
plt.show()