# CellAnnotate
Scripts used to create training data and annotations of cells from fluorescence microscopy images.

3D Annotations from 2D Annotated Data

Author: Christopher Z. Eddy, eddych@oregonstate.edu

One of the challenges in obtaining 3D annotated data (defined as 3D annotated data in either a z-stack of binary-labeled object/background, or a complete set of vertices and faces that form a triangular mesh around a 3D object) is the time-consuming nature of labeling each individual image slice. Instead, it is possible to obtain full 3D annotations from annotations made on maximum-projected [2D] annotated data. These data are far easier to annotate, as only a single slice per object is necessary to label.

Each cell-object containing a set of labeled x, y pixels has an axial intensity profile (Iobj) that is approximately Gaussian in appearance. By fitting the region, we obtain a profile (FIobj) for that object. For each x, y pixel belonging in the object, we take its axial intensity profile (Ixy) which may contain multiple peaks as a result of many objects dispersed in 3D. To suppress axial signatures that do not belong to the object x-y pixel, we multiply Ixy by FIobj, resulting in a small subset of remaining axial signatures that also appear normally distributed (Rxy). By applying a small threshold to limit blurring at the edges in the z-dimension (1/10 of the peak in Rxy), a final set of z-pixels to be paired with the object x-y pixel is determined. By performing this analysis over all x-y pixels in the cell-object, a complete 3D annotation is achieved. It should be noted that only the x-y pixels in the 2D annotated region are examined.

The program expects a binary mask input with shape H x W x N, where N channels is the number of annotated cells; each channel should contain only 1 labeled cell object [the maximum projected annotated cell] and the rest of the image is black-background. The input image should be H x W x Z, a typical Z stack with only 1 channel (gray). 

Possible Fail Cases:  Multiple peaks with approximately the same intensity and width may appear in the Iobj profile as a result of other cell-objects being in different z-slice but overlapping almost completely. To overcome this challenge, initial parameters given to the Guassian fitting algorithm must be very close to the peak-of-interest. In this case, it is recommended that you provide a list of approximate z-locations for each cell (if they are close to overlapping with another object) and pass this as an argument in run_analysis. The algorithm will handle the remaining analysis and should provide a good 3D annotation, however, I recommend you examine the fitted Iobj profile.
