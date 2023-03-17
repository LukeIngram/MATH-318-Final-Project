# samples_image_vis.py
# Image Visualization for Challenge_1 notebooks section 1

import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import numpy as np
import cv2


basepath = "../data/archive/Garbage_classification/"
fnames = ["cardboard1.jpg","glass1.jpg","metal1.jpg",
          "paper1.jpg","plastic1.jpg","trash1.jpg"]
    
samples = [
    cv2.imread(f"{basepath}cardboard/{fnames[0]}"),
    cv2.imread(f"{basepath}glass/{fnames[1]}"),
    cv2.imread(f"{basepath}metal/{fnames[2]}"),
    cv2.imread(f"{basepath}paper/{fnames[3]}"),
    cv2.imread(f"{basepath}plastic/{fnames[4]}"),
    cv2.imread(f"{basepath}trash/{fnames[5]}")
]

fig,axes = plt.subplots(2,3,subplot_kw={'xticks':(),'yticks':()})

for sample,ax,fname in zip(samples,axes.flatten(),fnames):
    ax.imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
    ax.set_title(fname)

plt.show()

