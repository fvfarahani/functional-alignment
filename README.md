# Effect of connectivity hyperalignment (CHA) on brain network properties: from coarse-scale to fine-scale

## An overview of the analysis pipeline
(A) Searchlight connectivity hyperalignment is performed in three steps: first, a connectome is created for each participant (i.e., a similarity matrix made up of data and targets); second, for a given searchlight across participants, the local transformation per subject and the common model is computed (here, an example for two subjects is shown); third, a mapper (whole-cortex transformation) is obtained for each participant by aggregating their local transformations. (B) Fine-scale graph analysis. A connectome was created using both the original and hyperaligned data. For each type, the connectomes were used to compute networks and a series of commonly used graph measures. Group-level statistical analysis was performed on the extracted measures.

![alt text](https://github.com/fvfarahani/hyperaligned-brain-network/blob/main/Pipeline.png?raw=true)

### Helpful links: <br />
Searchlight Hyperalignment: https://github.com/PyMVPA/PyMVPA/blob/master/mvpa2/algorithms/searchlight_hyperalignment.py <br />
Hyperalignment Tutorial: https://github.com/Summer-MIND/mind_2017/blob/master/Tutorials/hyperalignment/hyperalignment_tutorial.ipynb <br />
Connectivity Hyperalignment: https://github.com/PyMVPA/PyMVPA/blob/master/mvpa2/algorithms/connectivity_hyperalignment.py <br />
BrainIAK Tutorial: https://brainiak.org/tutorials/ <br />
Hypertools (Visualization): https://hypertools.readthedocs.io/en/latest/tutorials/align.html#aligning-data-with-hyperalignment <br />
BALSA: https://balsa.wustl.edu/DLabel/allScenes/B4Pxk <br />
HCP MMP: https://github.com/MRtrix3/mrtrix3/blob/master/share/mrtrix3/labelconvert/hcpmmp1_original.txt <br />
Templates and atlases: https://brainhack-princeton.github.io/handbook/content_pages/04-02-templates.html <br />
