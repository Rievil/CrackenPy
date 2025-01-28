![](https://github.com/Rievil/CrackPy/blob/main/Examples/Plots/Example.png)
Image segmentation of building material surfaces using deep learning (CrackPy)
=================================================================

This package is dedicated to segment cracks, matrix and pores in testing specimens of different building materials. The project was developed under the ressearch project of Grant Agency of Czech Republic No. 22-02098S with title: "Experimental analysis of the shrinkage, creep and cracking mechanism of the materials based on the alkali-activated slag".

[![DOI](https://zenodo.org/badge/734478397.svg)](https://doi.org/10.5281/zenodo.13969747)

- [Library](https://github.com/Rievil/CrackenPy)
- [Model](https://huggingface.co/rievil/crackenpy)
- [Dataset](https://huggingface.co/datasets/rievil/crackenpy_dataset)

_Please cite our research paper using the sidebar button when using CrackPy in your research project._

The videos documenting the source training images are represented via videos on youtube playlist [Cracks in alkali activated binders](https://youtube.com/playlist?list=PLE4UJv3O5jqNp2Oaqw1vnGiE2zu80q-Xe&si=aJCcwjeDd9LQq3NA).

Features
============================
Image segmentation of given surface of the test specimen. The photo of the specimen must have minimum resolution of 416 $\times$ 416 pixels.
The specimens should be placed on close-to-black background. The surface plane of the specimen should be parallel to the objective, to have minimal lens distortion. It is possible to give the main axis dimensions of the speicmens to calculate pixels to mm ratio.
The CrackPy package is able do generate mask with classes "background", "matrix", "cracks" and "pores". On these classes the CrackPy package introduce couple of metrics, which are the intersection of practises in image processing regarding the evaluation of building materials in the current state of the art. The most basic metrics are:

- edge_per_node
- crack_tot_length
- average_angle
- spec_area
- mat_area
- crack_ratio
- crack_length
- crack_thickness
- pore_area
- avg_pore_distance
- avg_pore_size

Tese metrics can be observed all at once, or just some of the metrics can be picked. If the time evolution of one speicmen is adressed the acquring of the metrics can be optimilized, to focus only for the metrics important for the given experiment.

The instalation of the package (private package)
============================
The package works optimal with NVIDIA GPU together with installed [CUDA](https://developer.nvidia.com/cuda-toolkit) 11.8 or newer. It was also tested on mac OS using MPS backend. The package was tested in Google Colab.
```
pip install crackenpy

```

The basic usage of the package 
=============================

```Python
from crackest.cracks import CrackPy
 
#%Model 1 optimized also for pores
#Model 0 is optimized for fine cracks
cp=CrackPy(model=1) #

#Read a file from examples
imfile=r'Examples\Img\ID14_940_Image.png' 
cp.get_mask(imfile)

#Plot the example
cp.overlay()
```

After instance initilaization, a pre-trained model is downloaded from HuggingFace repository, the model is stored in package folder for models. If the the NVIDIA GPU and CUDA toolkit is installed it will use cuda for segmentation. It can also use MPS backend on Apple Sillicon M1-M3 chips. The CrackenPy has basically two ways of use. Either on single image, where a specimen on dark background is placed, or on set of images. 

Single image
=============================
The basic usage is t osegment the whole image, and all metrics and masks will represent the whole image. This can be done, if the whole image is either filled with specimen, or if on the image is only the 1 specimen. In both cases all metrics connected to the matrix of binder, cracks and pores are assessed as 1 body / specimen.

If there is multiple speicmens on 1 image, then the Cracken should be used.

Multiple images
=============================
In this scenario on the image multiple specimens are present. In this case it is nessessary which specimen should be assessed. For this a Cracken class is designed. At first it masks the whole image, and then using skimage library segment it into regions. The specimen mask is created out of inverse background mask, therefore it allows to have cracks which are going straight thgough the specimens, and the speicmen is still recognized. Otherwise, the segmentation would returned two different specimens, even tho they would be one body. It will create a set of operations neded to crop and rotate each specimen by its longer axis. The current state of the library takes into account the fact that the photos are taken perpendicular to the base on which the bodies under consideration are placed. Each specimen is given an ID, and allows to retrive the mask and image in stabilized state.

The library is build to assess change in time of specimens, which are placed on fixed position in scene and are not moved throught the whole image acquistion period. The original aim is to monitorthe long term changes on multiple specimens. After the first registr of specimens, the segmentation is then always done on croped and stabilized version of image filled with the sample with adjustable frame around it. This saves time and allows to osberve also volumetric changes (expansion, shrinkage). 

```Python
from crackest.crack_analyzer import CrackAn

ca=CrackAn()
ca.input(file=r"Examples\Img\256_Image_29-01-2022 02-36-02.png")
ca.registr()
ca.preview()

```
![](https://github.com/Rievil/CrackPy/blob/main/Examples/Plots/Multiple_registr.png)

Both CrackPy and SubSpecies class inherit plotting methods from class CrackPlot, so it is possible to use same plotting functions for both cases.

```Python
spec=ca.get_spec(specid=0,frame=50)
spec.overlay()
```
![](https://github.com/Rievil/CrackPy/blob/main/Examples/Plots/Subspec.png)


The same apply for the CrackAnalyzer, which is generating metrics out of image and its segmented mask. By this a development of metrics can be generated if the folder is given to CrackAn instead of single file. The segmentation in registr method is currently using the same model for recognizing how many specimens is on the image, as well for the segmnetation itself, so for the experiments it is recomended to keep the speicmens still on same spot. Otherwise it would need to segment whole image every time.

Future plans
=============================
The current library is designed for laboratory enviroment, where every possible of binder can be placed (cement, geopolymers, alkali activated systems) and it should be able to segment cracks, matrix, pores and background. This is intented use, however the crack variety and vast brightness and texture allow to design in general accurate and reliable deep learning aplication for crack 
segmentation in different type of mediums.

- Segmentation using photogrammetry (basicly running CrackenPy on UV textures of generated 3D models)
- Segmentation on videos (using less power demanding models to process bigger volumes of data)
- Connecting the image with BIM metadata for structure surveing (for infrastructre isnepctions)


Acknowledgment
=============================
This package was written under the Czech Science Foundation, grant number 22-02098S, project title „Experimental analysis of the shrinkage, creep and cracking mechanism of the materials based on the alkali-activated slag“.

Please cite our work
========================
```latex
@misc {richard_dvorak_2024,
	author       = { {Richard Dvorak} },
	title        = { crackenpy (Revision 04ed02c) },
	year         = 2024,
	url          = { https://huggingface.co/rievil/crackenpy },
	doi          = { 10.57967/hf/3295 },
	publisher    = { Hugging Face }
}

@software {Dvorak_CrackenPy_Image_segmentation_2024,
	author       = {Dvorak, Richard and Bilek, Vlastimil and Krc, Rostislav and Kucharczykova, Barbara},
	doi          = {10.5281/zenodo.13969747},
	month        = oct,
	title        = {{CrackenPy: Image segmentation tool for semantic segmentation of building material surfaces using deep learning}},
	url          = {https://github.com/Rievil/CrackenPy},
	year         = {2024}
}

@misc {richard_dvorak_2024,
	author       = { {Richard Dvorak} },
	title        = { crackenpy_dataset (Revision ce5c857) },
	year         = 2024,
	url          = { https://huggingface.co/datasets/rievil/crackenpy_dataset },
	doi          = { 10.57967/hf/3496 },
	publisher    = { Hugging Face }
}
```
