import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import plotting
from nilearn import image
import requests
import os
import shutil
import gzip
import numpy as np

if __name__ == '__main__':

    # Get the current working directory
    current_directory = os.getcwd()

    # Define the new directory path
    atlas_dir = os.path.join(current_directory, 'Atlas')

    dict_datasets = {}

    # download datasets
    datasets.fetch_coords_power_2011()
    dict_datasets['craddock_2012'] = datasets.fetch_atlas_craddock_2012(data_dir=atlas_dir)

    dict_datasets['AAL template for SPM 12'] = datasets.fetch_atlas_aal(data_dir=atlas_dir)
    atlas_filename = dict_datasets['AAL template for SPM 12'].maps

    plotting.plot_roi(atlas_filename, title="AAL template for SPM 12", view_type='contours')
    output_file = os.path.join(atlas_dir, 'aal_SPM12/aal_SPM12.pdf')
    plt.savefig(output_file)
    plt.close()  # Close the plot to free up memory

    # Harvard Oxford Atlas
    dict_datasets['harvard_oxford_cort-prob-2mm'] = \
        datasets.fetch_atlas_harvard_oxford('cort-prob-2mm', data_dir=atlas_dir)
    dict_datasets['harvard_oxford_sub-prob-2mm'] = \
        datasets.fetch_atlas_harvard_oxford('sub-prob-2mm', data_dir=atlas_dir)

    # Multi Subject Dictionary Learning Atlas
    dict_datasets['msdl'] = datasets.fetch_atlas_msdl(data_dir=atlas_dir)

    # Smith ICA Atlas and Brain Maps 2009
    dict_datasets['smith_2009'] = datasets.fetch_atlas_smith_2009(data_dir=atlas_dir)

    # ICBM tissue probability
    # datasets.fetch_icbm152_2009(data_dir=atlas_dir)

    # Allen RSN networks
    # datasets.fetch_atlas_allen_2011(data_dir=atlas_dir)

    # Pauli subcortical atlas
    datasets.fetch_atlas_pauli_2017(data_dir=atlas_dir)

    datasets.fetch_atlas_schaefer_2018(data_dir=atlas_dir)

    datasets.fetch_atlas_basc_multiscale_2015(data_dir=atlas_dir)

    # Dictionaries of Functional Modes (“DiFuMo”) atlas
    for dim in [128, 256, 512, 1024]:
        datasets.fetch_atlas_difumo(dimension=dim, resolution_mm=2, data_dir=atlas_dir)

    # Define URLs
    cc200_roi_atlas_url = \
        'https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/cc200_roi_atlas.nii.gz'
    cc200_roi_atlas_label_url = 'https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/CC200_ROI_labels.csv'

    cc400_roi_atlas_url = 'https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/cc400_roi_atlas.nii.gz'
    cc400_roi_atlas_label_url = 'https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/CC400_ROI_labels.csv'

    # Define the paths for the downloaded and uncompressed files
    cc200_roi_atlas_atlas_dir = os.path.join(atlas_dir, 'cc200_roi_atlas.nii')
    cc400_roi_atlas_atlas_dir = os.path.join(atlas_dir, 'cc400_roi_atlas.nii')

    # Create the directory for the uncompressed file
    os.makedirs(cc200_roi_atlas_atlas_dir, exist_ok=True)
    os.makedirs(cc400_roi_atlas_atlas_dir, exist_ok=True)

    # Define directories and file paths
    cc200_roi_atlas_compressed_file_path = os.path.join(atlas_dir, 'cc200_roi_atlas.nii.gz')
    cc200_roi_atlas_uncompressed_file_path = os.path.join(cc200_roi_atlas_atlas_dir, 'CC200.nii')
    cc200_roi_atlas_label_file_path = os.path.join(atlas_dir, 'CC200_ROI_labels.csv')

    cc400_roi_atlas_compressed_file_path = os.path.join(atlas_dir, 'cc400_roi_atlas.nii.gz')
    cc400_roi_atlas_uncompressed_file_path = os.path.join(cc400_roi_atlas_atlas_dir, 'CC400.nii')
    cc400_roi_atlas_label_file_path = os.path.join(atlas_dir, 'CC400_ROI_labels.csv')

    # Download the .nii.gz file
    cc200_roi_atlas_response = requests.get(cc200_roi_atlas_url, stream=True)
    if cc200_roi_atlas_response.status_code == 200:
        with open(cc200_roi_atlas_compressed_file_path, 'wb') as f:
            f.write(cc200_roi_atlas_response.content)
        print(f"Downloaded {cc200_roi_atlas_compressed_file_path}")
    else:
        print(f"Failed to download the .nii.gz file. Status code: {cc200_roi_atlas_response.status_code}")
        cc200_roi_atlas_response.raise_for_status()

    # Unzip the file
    with gzip.open(cc200_roi_atlas_compressed_file_path, 'rb') as f_in:
        with open(cc200_roi_atlas_uncompressed_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Uncompressed file saved as {cc200_roi_atlas_uncompressed_file_path}")

    # Download the CSV file
    cc200_roi_atlas_label_response = requests.get(cc200_roi_atlas_label_url, stream=True)
    if cc200_roi_atlas_label_response.status_code == 200:
        with open(cc200_roi_atlas_label_file_path, 'wb') as f:
            f.write(cc200_roi_atlas_label_response.content)
        print(f"Downloaded {cc200_roi_atlas_label_file_path}")
    else:
        print(f"Failed to download the CSV file. Status code: {cc200_roi_atlas_label_response.status_code}")
        cc200_roi_atlas_label_response.raise_for_status()

    # Download the .nii.gz file
    cc400_roi_atlas_response = requests.get(cc400_roi_atlas_url, stream=True)
    if cc400_roi_atlas_response.status_code == 200:
        with open(cc400_roi_atlas_compressed_file_path, 'wb') as f:
            f.write(cc400_roi_atlas_response.content)
        print(f"Downloaded {cc400_roi_atlas_compressed_file_path}")
    else:
        print(f"Failed to download the .nii.gz file. Status code: {cc400_roi_atlas_response.status_code}")
        cc400_roi_atlas_response.raise_for_status()

    # Unzip the file
    with gzip.open(cc400_roi_atlas_compressed_file_path, 'rb') as f_in:
        with open(cc400_roi_atlas_uncompressed_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Uncompressed file saved as {cc400_roi_atlas_uncompressed_file_path}")

    # Download the CSV file
    cc400_roi_atlas_label_response = requests.get(cc400_roi_atlas_label_url, stream=True)
    if cc400_roi_atlas_label_response.status_code == 200:
        with open(cc400_roi_atlas_label_file_path, 'wb') as f:
            f.write(cc400_roi_atlas_label_response.content)
        print(f"Downloaded {cc400_roi_atlas_label_file_path}")
    else:
        print(f"Failed to download the CSV file. Status code: {cc400_roi_atlas_label_response.status_code}")
        cc400_roi_atlas_label_response.raise_for_status()

    # dataset = datasets.fetch_atlas_difumo(1024,resolution_mm=3,data_dir=atlas_dir)
    # dataset = datasets.fetch_coords_power_2011()
    # maps = dataset.rois
    # plotting.plot_roi(atlas_filename, title="Harvard Oxford atlas",view_type='contours')
    # plotting.show()

    # dataset = datasets.fetch_atlas_craddock_2012(atlas_dir)
    # atlas_filename = dataset.scorr_mean
    # print(image.load_img(atlas_filename).shape)
    # first_rsn = image.index_img(atlas_filename, 1)
    # plotting.plot_roi(first_rsn, title="Cockdoc",view_type='contours')
    # plotting.show()
