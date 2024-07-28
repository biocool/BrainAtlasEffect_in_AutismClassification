# AutismAtlasBrainFMRI

## Required dependencies for this project include
- python 3.7 or higher
- mne
- pandas
- stats
- sklearn
- nilearn
- numpy
- seaborn
- matplotlib
- xlrd
- kaleido 0.1.0.post1
- plotly
- pdfkit


## How to download processed datasets from ABIDE?
To download all processed datasets from ABIDE, you can run the following command:
`python <a href='BrainAtlasEffect_in_AutismClassification/blob/main/Download_from_abide.py'>Download_from_abide.py</a>` 

2- To extract feature using nilearn package use the Atlas_Extractor.py. Atlas must be saved in the  Nilearn subfolder of AtlasExtracted directory.

3- Build the model  
  

The Making_Model.py is the first step that you need to run the code. 
There are four classes in this file: Pure_Classifier, PreProcess, AutismClassifier, and ConnectivityMap.

In order to get the result, you need to make an object from the AutismClassifier. Here is the example:

autism = AutismClassifier()
autism.Classify_Atlas(‘filenameresult’,fromage,toage)
By setting the age parameters, you can specify the age range.

Then you must call the AutismClassifier method for the Combat result
autism. Classify_Atlas_With_Combat(“filenameresult”,fromage,toage)

Tthe site results will be returned by calling the following function :

autism.Classify_Atlas_For_each_Site()

You must call Atlas_Best.py to determine the best classifier and best Atlas after you have run this file. 

Results will be saved in the result directory, so make sure it exists in the current folder.

Lastly, to get ChordDiagram and important features, run the following code:

connectivity = ConnectivityMap()
connectivity.Connectivity()
The results will saved in the result directory. 

The Connectivty.py provide some useful function to draw Chordmap.

Atlas_Best is the python script file for find best classifier across all calssifier and atlas 

Atlas_Download is the another python script for download atlas from ABIDE website. In this script there are some option to set which type of image shoukd be dowbload. For example you can set sex, site, modality and so on . . .









