from nilearn import datasets
import nilearn as nib
from nilearn.datasets import fetch_abide_pcp
from nilearn.input_data import NiftiMapsMasker
from nilearn import image
import nibabel
from nilearn.connectome import ConnectivityMeasure
import numpy as np
from nilearn import plotting
import glob
import os


def Save_To_File(filename, corrmatrix, out_dir):
    basename = os.path.basename(filename)
    newname = basename[0:len(basename) - 7]
    np.save("{}{}.npy".format(out_dir, newname), corrmatrix)


if __name__ == '__main__':

    # path of FMRI files
    fmri_path = "dat/Outputs/cpac/filt_global/func_preproc/*.gz"
    atlas_dir = "Atlas/"

    out_dir = "Atlas_Extractor_test/NilearnNew_test/"

    os.makedirs(out_dir, exist_ok=True)

    # Load atlases directly from local files
    craddock = nibabel.load(os.path.join(atlas_dir, 'craddock_2012/scorr05_mean_all.nii.gz'))
    smith_bm70 = nibabel.load(os.path.join(atlas_dir, 'smith_2009/bm70.nii.gz'))
    # smith_rsn70 = nibabel.load(os.path.join(atlas_dir, 'smith_2009/rsn70.nii.gz'))
    msdl = nibabel.load(os.path.join(atlas_dir, 'msdl_atlas/MSDL_rois/msdl_rois.nii'))
    hoa21 = nibabel.load(os.path.join(atlas_dir, 'fsl/data/atlases/HarvardOxford/HarvardOxford-sub-prob-1mm.nii.gz'))
    difumo_128 = nibabel.load(os.path.join(atlas_dir, 'difumo_atlases/128/3mm/maps.nii.gz'))
    difumo_256 = nibabel.load(os.path.join(atlas_dir, 'difumo_atlases/256/3mm/maps.nii.gz'))
    difumo_512 = nibabel.load(os.path.join(atlas_dir, 'difumo_atlases/512/3mm/maps.nii.gz'))

    # craddock = datasets.fetch_atlas_craddock_2012(atlas_dir)
    # smith = datasets.fetch_atlas_smith_2009(atlas_dir)
    # msdl = datasets.fetch_atlas_msdl(atlas_dir)
    # hard = datasets.fetch_atlas_harvard_oxford('sub-prob-1mm', atlas_dir)
    # mutli = datasets.fetch_atlas_basc_multiscale_2015("sym", atlas_dir)
    # difumo = datasets.fetch_atlas_difumo(dimension=512, resolution_mm=3, data_dir=atlas_dir)

    # image = image.load_img("ROI/cc400_roi_atlas.nii/cc400.nii")

    # craddockmasker = NiftiMapsMasker(maps_img=craddock.scorr_mean, standardize=True, memory='nilearn_cache', verbose=0)
    # smithmasker70 = NiftiMapsMasker(maps_img=smith.bm70, standardize=True, memory='nilearn_cache', verbose=0)
    # smithmaskerr70 = NiftiMapsMasker(maps_img=smith.rsn70, standardize=True, memory='nilearn_cache', verbose=0)
    # msdlmasker = NiftiMapsMasker(maps_img=msdl.maps, standardize=True, memory='nilearn_cache', verbose=0)

    # hardmasker = NiftiMapsMasker(maps_img=hard.maps, standardize=True, memory='nilearn_cache', verbose=0)
    #difumomasker = NiftiMapsMasker(maps_img=difumo.maps, standardize=True, memory='nilearn_cache', memory_level=1,
    #                               verbose=0)

    # Create the NiftiMapsMasker objects
    craddockmasker = NiftiMapsMasker(maps_img=craddock, standardize=True, memory='nilearn_cache', verbose=0)
    smithmasker70 = NiftiMapsMasker(maps_img=smith_bm70, standardize=True, memory='nilearn_cache', verbose=0)
    # smithmaskerr70 = NiftiMapsMasker(maps_img=smith_rsn70, standardize=True, memory='nilearn_cache', verbose=0)
    hardmasker = NiftiMapsMasker(maps_img=hoa21, standardize=True, memory='nilearn_cache', verbose=0)
    difumomasker_128 = NiftiMapsMasker(maps_img=difumo_128, standardize=True, memory='nilearn_cache', memory_level=1,
                                       verbose=0)
    difumomasker_256 = NiftiMapsMasker(maps_img=difumo_256, standardize=True, memory='nilearn_cache', memory_level=1,
                                       verbose=0)
    difumomasker_512 = NiftiMapsMasker(maps_img=difumo_512, standardize=True, memory='nilearn_cache', memory_level=1,
                                       verbose=0)
    msdlmasker = NiftiMapsMasker(maps_img=msdl, standardize=True, memory='nilearn_cache', verbose=0)

    atlas_dict = {'difumo128': difumomasker_128, 'difumo256': difumomasker_256, 'difumo512': difumomasker_512,
                  'hard': hardmasker, 'msdl': msdlmasker, 'smith70': smithmasker70, 'craddock': craddockmasker}

    for atlas in atlas_dict.keys():
        for file in glob.glob(fmri_path):
            try:
                print(file)
                time_series = atlas_dict[atlas].fit_transform(file)
                print(time_series.shape)
                correlation_measure = ConnectivityMeasure(kind='correlation')
                correlation_matrix = correlation_measure.fit_transform([time_series])[0]

                os.makedirs(out_dir + atlas, exist_ok=True)

                Save_To_File(file, correlation_matrix, out_dir + atlas + "/")

            except:
                print("Error in {}".format(file))

    # web site: https://joaoloula.github.io/functional-atlas.html
