# CellAnnotate
Scripts used to create training data and annotations of cells from fluorescence microscopy images.

## 3D Annotations from 2D Annotated Data
*Author: Christopher Z. Eddy, eddych@oregonstate.edu*

**Description**:
One of the challenges in obtaining 3D annotated data (defined as 3D annotated data in either a z-stack of binary-labeled object/background, or a complete set of vertices and faces that form a triangular mesh around a 3D object) is the time-consuming nature of labeling each individual image slice. Instead, it is possible to obtain full 3D annotations from annotations made on maximum-projected [2D] annotated data. These data are far easier to annotate, as only a single slice per object is necessary to label.

Each cell-object containing a set of labeled x, y pixels has an axial intensity profile (I<sub>obj</sub>) that is approximately Gaussian in appearance. By fitting the region, we obtain a profile (FI<sub>obj</sub>) for that object. For each x, y pixel belonging in the object, we take its axial intensity profile (I<sub>xy</sub>) which may contain multiple peaks as a result of many objects dispersed in 3D. To suppress axial signatures that do not belong to the object x-y pixel, we multiply I<sub>xy</sub> by FI<sub>obj</sub>, resulting in a small subset of remaining axial signatures that also appear normally distributed (R<sub>xy</sub>). By applying a small threshold to limit blurring at the edges in the z-dimension (1/10 of the peak in R<sub>xy</sub>), a final set of z-pixels to be paired with the object x-y pixel is determined. By performing this analysis over all x-y pixels in the cell-object, a complete 3D annotation is achieved. It should be noted that only the x-y pixels in the 2D annotated region are examined.

The program expects a binary mask input with shape H x W x N, where N channels is the number of annotated cells; each channel should contain only 1 labeled cell object [the maximum projected annotated cell] and the rest of the image is black-background. The input image should be H x W x Z, a typical Z stack with only 1 channel (gray).

**Possible Fail Cases**:  Multiple peaks with approximately the same intensity and width may appear in the I<sub>obj</sub> profile as a result of other cell-objects being in different z-slice but overlapping almost completely. To overcome this challenge, initial parameters given to the Guassian fitting algorithm must be very close to the peak-of-interest. In this case, it is recommended that you provide a list of approximate z-locations for each cell (if they are close to overlapping with another object) and pass this as an argument in run_analysis. The algorithm will handle the remaining analysis and should provide a good 3D annotation, however, I recommend you examine the fitted I<sub>obj</sub> profile.

**Tutorial**:

Example of application on cell images:

| - | - | - |
|---|---|---|
| ![Max Proj](/images/Tutorial_6.png) | [Z-stack](/images/Tutorial_6_image.gif) | ![Output binary](/images/Tutorial_6_output.gif) |
| ![Z-stack](/images/Tutorial_6_image.gif) | I am text to the right |
<!-- ![Max Proj](/images/Tutorial_6.png) -->
<!-- ![Z-stack](/images/Tutorial_6_image.gif) -->
<!-- ![Output binary](/images/Tutorial_6_output.gif) -->

**Coding Practices**:

First, save your z-stack images as a single .ome.tif file, a z-stack type image loadable in both ImageJ and Python:

![Save .ome.tif files](/images/Tutorial_1.png)

Load the saved .ome.tif file into Python:

![Load .ome.tif file](/images/Tutorial_2.png)


Load the 2D annotations into a 3D numpy array called ‘mask’ (shape H x W x N) with the binary mask of the nth cell placed into shape H x W x n channel. There are many ways to do this.
  - Load a single maximum projected binary image containing many cells where each cell is separated by a thin line. Run a label scheme such as from skimage.measure.label, and place each cell into its own channel. Email if you have questions.
  - Load a JSON file with annotations, the setup of your script depends on the structure of the JSON file. An example is shown below:

![Load mask json file](/images/Tutorial_3.png)

Produce the 3D annotations from the 2D annotated ‘mask’ data and save the output into a z-stack:

![Find 3D annotations](/images/Tutorial_4.png)

Evaluating test cases:

![Test cases](/images/Tutorial_5.png)

**See bottom of Find_3D_annotation.py for code examples!**
