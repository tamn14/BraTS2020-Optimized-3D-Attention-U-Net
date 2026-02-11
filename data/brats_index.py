
from pathlib import Path

def build_brats_index(data_path : Path) -> list:
    """
    Build an index of BraTS dataset files.

    Args:
        data_path (Path): The root directory of the BraTS dataset.

    Returns:
        List[dict] : A list of dictionaries, each containing paths to the modalities and segmentation for a patient.
    """
    image_keys = ["image_flair", "image_t1", "image_t1ce", "image_t2"]  # Define the keys for each MRI modality
    all_files = []  # List to hold the file dictionaries for each patient
    
    """ Check if the data path exists """
    if not data_path.exists():
        raise FileNotFoundError(f"The specified data path {data_path} does not exist.")
    
    
    """
        Scan through each patient directory in the dataset
        if the directory contains all required modalities and segmentation, add to index
        else, skip the patient and print a warning
    
    """
    for patient_dir in sorted(data_path.iterdir()):
        if not patient_dir.is_dir():
            continue
        
        scans = {}  # It like draft to hold modality file paths
        valid = True 
        
        
        """ Load each modality """
        for key in image_keys : 
            modal = key.replace("image_", "")
            matches = sorted(patient_dir.glob(f"*_{modal}.nii*"))
            
            if len(matches) == 0:
                print(f"Warning: No files found for modality {modal} in patient directory {patient_dir}. Skipping this patient.")
                valid = False
                break
            
            """
                matches will return a list of files matching the pattern
                we expect only one file per modality, so we take the first match
                But in some cases, we have more than one match, for example :
                matches =[
                            Path("..._flair.nii.gz"),
                            Path("..._flair_backup.nii.gz")
                        ] , so we use matches[0] to make sure we get the correct file at index 0
                
            """
            scans[key] = matches[0]  # Take the first match for the modality
            
        """ Load segmentation file """
        seg_matches = sorted(patient_dir.glob("*_seg.nii*"))
        if len(seg_matches) == 0:
            print(f"Warning: No segmentation file found in patient directory {patient_dir}. Skipping this patient.")
            valid = False
        
        if not valid:
            continue
        
        file_dict = {
            **scans,
            "label": seg_matches[0],
            "case_id": patient_dir.name
            
            }
        all_files.append(file_dict)
        
    return all_files
    
    
    