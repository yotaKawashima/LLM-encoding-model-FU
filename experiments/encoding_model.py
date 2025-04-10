import os 
import numpy as np
from tqdm import tqdm 
import config
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.base import clone
from scipy.stats import pearsonr
#from utils.roi_mask import get_roi_mask 


def fit_and_predict(fmri_train_path, 
                    fmri_test_path,
                    model_representation_train_path,
                    model_representation_test_path,
                    regression=RidgeCV(alphas=config.alphas, cv=config.num_folds)):
    """
    Fits a regression model to fMRI data and predicts the test data for a given subject.
    Assumes that the fMRI data is stored in voxel x trial format.

    Args:
        fmri_train_path (str): Path to the training fMRI data.
        fmri_test_path (str): Path to the test fMRI data.
        model_representation_train_path (str): Path to the training model representation data.
        model_representation_test_path (str): Path to the test model representation data.        
        regression (object): A regression model instance from scikit-learn. Default is RidgeCV().

    Returns:
        tuple: A tuple containing:
            - corrs (np.ndarray): Correlation coefficients.
            - coeff_determination (np.ndarray): Coefficient of determination (R^2) for the test data.
            - best_alpha (float): Best alpha value from the regression model.
            - best_score_train (np.ndarray): Best score for the training data.
            - fmri_test_pred (np.ndarray): Predicted fMRI data.
            - reg (object): Fitted regression model.
    """

    # Load fmri of the subject
    # fMRI data: voxel x trial -> trial x voxel 
    fmri_train = np.transpose(np.load(fmri_train_path))
    fmri_test = np.transpose(np.load(fmri_test_path))
    
    # Load model representation 
    model_representation_train = \
            np.load(model_representation_train_path)
    model_representation_test = \
            np.load(model_representation_test_path)

    # Make new instances of the linear reg class passed as input
    reg = clone(regression)

    print("shape of fmri_train: ", fmri_train.shape)
    print("shape of fmri_test: ", fmri_test.shape)
    print("shape of model_representation_train: ", model_representation_train.shape)
    print("shape of model_representation_test: ", model_representation_test.shape)

    # check if the number of trials in fmri and model representation are the same
    if fmri_train.shape[0] != model_representation_train.shape[0]:
        raise ValueError("The number of train trials in fMRI and model representation data do not match.")
    if fmri_test.shape[0] != model_representation_test.shape[0]:
        raise ValueError("The number of test trials in fMRI and model representation data do not match.")

    # Exclude voxels with Nan values in the training fMRI data as RidgeCV does not support NaN values.
    voxel_with_nan = np.isnan(fmri_train).any(axis=0)
    fmri_train_without_nan = fmri_train[:, ~voxel_with_nan]
    if voxel_with_nan.any():
        Warning("There are np.nan in the training fMRI data. ")
        print("The number of voxels with Nan value: ", np.sum(voxel_with_nan))

    # Fit linear regressions on the training data
    reg.fit(model_representation_train, fmri_train_without_nan)

    # Keep ridge regression data. 
    # By default, sklearn score averages coefficient of determination R^2 across all targets (i.e. voxels).
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html 
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
    best_alpha = reg.alpha_
    best_score_train = reg.best_score_

    # Use fitted linear regressions to predict the validation fMRI data
    coeff_determination = reg.score(model_representation_test, fmri_test[:, ~voxel_with_nan])
    
    # Predict the test data
    fmri_test_pred_without_nan = reg.predict(model_representation_test)
    fmri_test_pred = np.zeros(fmri_test.shape)
    fmri_test_pred[:, ~voxel_with_nan] = fmri_test_pred_without_nan
    fmri_test_pred[:, voxel_with_nan] = np.nan # set the voxels with nan to np.nan
    
    # Correlate predicted and ground-truth values
    # initialise test_corrs to store correlations for each fMRI vertex.
    num_voxels = fmri_test.shape[1]
    corrs = np.zeros(num_voxels)

    # Correlate each predicted fMRI vertex with the corresponding ground truth vertex.
    # Voxels with NaN are excluded from the correlation calculation.
    for v in range(num_voxels):
        if not voxel_with_nan[v]:
            corrs[v] = pearsonr(fmri_test_pred[:,v], fmri_test[:,v])[0]
        else:
            corrs[v] = np.nan

    return corrs, coeff_determination, best_alpha, best_score_train, reg

def save_prediction_in_single_file(save_data_path: str,
                                   **data):
    """ Save the predictions to a file.
    Args:
        save_data_path (str): Path to save the predictions.
        **data: Data to be saved.
    """
    # Save the predictions to a file
    np.savez(save_data_path, **data)
    print(f'Saved an encoding model result to {save_data_path}')
    return None                              

def fit_and_predict_all_subj(fmri_dir,
                             model_representation_dir, 
                             rois: list = config.rois, 
                             fmri_file_name: str = config.fmri_file_name,
                             fmri_file_extension: str = config.fmri_file_extension,
                             model_representation_file_name: str = config.model_representation_file_name,
                             model_representation_file_extension: str = config.model_representation_file_extension,
                             regression=RidgeCV(alphas=config.alphas, cv=config.num_folds)) -> pd.DataFrame: 
    """ Fits a linear regression model and predicts fMRI data for all subjects.

    Args:
        fmri_dir (str): Directory containing fMRI data for all subjects.
        model_representation_dir (str): Directory containing model representation data.
        rois (list): List of regions of interest (ROIs) to be analyzed.
        fmri_file_name (str): Base name of the fMRI data files.
        fmri_file_extension (str): File extension of the fMRI data files.
        model_representation_file_name (str): Base name of the model representation files.
        model_representation_file_extension (str): File extension of the model representation files.
        regression (object): A regression model instance from scikit-learn. Default is RidgeCV().

    Returns:
        pd.DataFrame: A DataFrame containing the mean correlation results 
            for each subject, hemisphere, and ROI. Columns include 
            'subject', 'mean_corr', 'hemisphere', and 'roi'.
    """    
    # initialise a list to store the results
    prediction_df = []

    # loop over all subjects
    num_subj = int(len(os.listdir(fmri_dir)) / 2) # 2 files per participant (train & test) 
    print('Fitting and predicting for each subject: Start')
    
    for subj in tqdm(range(1, 1+num_subj)):
        
        # get data path 
        fmri_train_file_name = f'subj{subj:02d}_' + fmri_file_name + '_train'+ fmri_file_extension
        fmri_test_file_name = f'subj{subj:02d}_' + fmri_file_name + '_test'+ fmri_file_extension
        fmri_train_path = os.path.join(fmri_dir, fmri_train_file_name)
        fmri_test_path = os.path.join(fmri_dir, fmri_test_file_name)
        
        model_representation_train_file_name = \
            f'subj{subj:02d}_' + model_representation_file_name + '_train' + model_representation_file_extension
        model_representation_test_file_name = \
            model_representation_file_name + '_test' + model_representation_file_extension
        model_representation_train_path = \
            os.path.join(model_representation_dir, model_representation_train_file_name)
        model_representation_test_path = \
            os.path.join(model_representation_dir, model_representation_test_file_name)
        
        # fit and predict for the current subject
        corrs, _, _, _ = fit_and_predict(fmri_train_path, 
                                         fmri_test_path, 
                                         model_representation_train_path,
                                         model_representation_test_path,
                                         regression=regression)
        
        prediction_df.append({
            'subject': f'subj-{subj:02d}',
            'corr': corrs,
        })
        
        # # summary for each ROI 
        # for roi in rois:
        #     roi_mask, _ = get_roi_mask(roi, 'left', subj_dir)

        #     prediction_df.append({
        #         'subject': f'subj-{subj:02d}',
        #         'roi': roi, 
        #         'mean_corr': np.mean(corrs[roi_mask!=0]),
        #     })

    print('Fitting and predicting for each subject: End')    

    # convert to padas dataframe
    prediction_df = pd.DataFrame(prediction_df)

    return prediction_df