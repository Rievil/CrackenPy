![](https://github.com/Rievil/CrackPy/blob/main/Plots/Example.png)
Image segmentation of building material surfaces using deep learning (CrackPy)
=================================================================

This package is dedicated to segment cracks, matrix and pores in testing specimens of different building materials. The project was developed under the ressearch project of Grant Agency of Czech Republic No. 22-02098S with title: "Experimental analysis of the shrinkage, creep and cracking mechanism of the materials based on the alkali-activated slag".

_Please cite our research paper using the sidebar button when using CrackPy in your research project._
> Dvorak, R., xxx https://doi.org/10.xxxx

Features
=================================================================
Image segmentation of given surface of the test specimen. The photo of the specimen must have minimum resolution of 416 $\times$ 416 pixels.
The specimens should be placed on close-to-black background. The surface plane of the specimen should be parallel to the objective, to have minimal lens distortion. It is possible to give the main axis dimensions of the speicmens to calculate pixels to mm ratio.
The CrackPy package is able do generate mask with classes "background", "matrix", "cracks" and "pores". On these classes the CrackPy package introduce couple of metrics, which are the intersection of practises in image processing regarding the evaluation of building materials in the current state of the art. The most basic metrics are:
- Areas of each class
- Crack ratio
- Average pore size
- Average pore distance

The advance metrics of the cracks are:
- Distance map
- Skeleton of crack networks
- Node network
- Crack distribution

Tese metrics can be observed all at once, or just some of the metrics can be picked. If the time evolution of one speicmen is adressed the acquring of the metrics can be optimilized, to focus only for the metrics important for the given experiment.

The instalation of the package
============================
$pip install crackpy

The usage of the package 
=============================

```Python
from cracks import cracks as cr
#Cration of the class used for the segmentation
cp=cr.CrackPy()
```


