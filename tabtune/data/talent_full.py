"""Complete TALENT dataset implementation with all 300+ datasets."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import requests
import zipfile
import json
import os
from urllib.parse import urljoin
import gdown
from abc import ABC, abstractmethod

from .base import BaseDataset

logger = logging.getLogger(__name__)


class TALENTFullDataset(BaseDataset):
    """Complete TALENT dataset with all 300+ datasets from the official repository."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.preprocess = config.get('preprocess', True)
        self.dataset_name = config.get('dataset_name')
        self.normalize = config.get('normalize', True)
        self.handle_missing = config.get('handle_missing', 'fill')
        self.data_path = Path(config.get('data_path', './talent_data'))
        self.data_path.mkdir(parents=True, exist_ok=True)
        # self.talent_file_url = 'https://drive.google.com/uc?id=1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z&export=download'

        self.talent_file_url = 'https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z'

        
        # Discovery/IO flags
        self.skip_download = bool(config.get('skip_download', True))
        # Root folder where individual TALENT datasets live (may be nested inside extracted zip)
        

        #self.talent_folder_url = 'https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z'
        
        # The dataset_root is the directory where the individual dataset folders will reside.
        # gdown will create a 'data' subfolder inside data_path.
        self.dataset_root = self.data_path / 'data'
        
        # All 300+ TALENT datasets from official repository
        self.all_datasets = [
            # A-C datasets
            '1000-Cameras-Dataset', '2dplanes', '3D_Estimation_using_RSSI_of_WLAN_dataset',
            '3D_Estimation_using_RSSI_of_WLAN_dataset_complete_1_target', 'ASP-POTASSCO-classification',
            'Abalone_reg', 'Ailerons', 'Amazon_employee_access', 'Another-Dataset-on-used-Fiat-500-(1538-rows)',
            'BLE_RSSI_dataset_for_Indoor_localization', 'BNG(breast-w)', 'BNG(cmc)', 'BNG(echoMonths)',
            'BNG(lowbwt)', 'BNG(mv)', 'BNG(stock)', 'BNG(tic-tac-toe)', 'Bank_Customer_Churn_Dataset',
            'Basketball_c', 'Bias_correction_r', 'Bias_correction_r_2', 'Brazilian_houses_reproduced',
            'CPMP-2015-regression', 'CPS1988', 'California-Housing-Classification', 'Cardiovascular-Disease-dataset',
            'Click_prediction_small', 'Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-10.0GHz(Urbinati)',
            
            # D-F datasets
            'Diabetes130US', 'Diabetes_binary_5050split_health_indicators_BRFSS2015',
            'Diabetes_binary_health_indicators_BRFSS2015', 'Dresses_Attribute_Sales',
            'Electrical_Grid_Stability_Simulated_Data', 'Employee_Attrition', 'Employee_Salaries',
            'Epileptic_Seizure_Recognition', 'FIFA_2018_Statistics', 'Fashion_MNIST_784',
            'Feature_Selection_for_Machine_Learning', 'Fertility', 'Financial_Distress',
            
            # G-I datasets
            'German_Credit_Data', 'Glass_Identification', 'Graduate_Admission_2', 'Grocery_Dataset',
            'Heart_Disease_Prediction', 'Heart_failure_clinical_records', 'House_Prices_Advanced_Regression_Techniques',
            'Housing_Boston', 'IBM_HR_Analytics_Employee_Attrition_Performance', 'ILPD_Indian_Liver_Patient_Dataset',
            'Insurance', 'Ionosphere',
            
            # J-L datasets
            'Jannis', 'Kc1', 'Kc2', 'Kc3', 'Kdd_internet_usage', 'Kidney_Disease',
            'LDPA', 'Laptop_price', 'Letter_Recognition', 'Liver_Disorders',
            
            # M-O datasets
            'MNIST_784', 'Machine_Learning_based_ZZAlPha_Ltd_Stock_Recommendations_2012_to_2014',
            'Medical_Cost_Personal_Datasets', 'Mice_Protein_Expression', 'Mobile_Price_Classification',
            'Mushroom_Classification', 'NASA_Airfoil_Self_Noise', 'Netflix_Movies_and_TV_Shows',
            'Numerai28.6', 'Obesity_among_adults_by_country_1975_2016', 'Online_Shoppers_Purchasing_Intention_Dataset',
            'Optical_Recognition_of_Handwritten_Digits', 'Otto_Group_Product_Classification_Challenge',
            
            # P-R datasets
            'Palmer_Penguins', 'Parkinsons', 'Parkinsons_Telemonitoring', 'Phishing_Websites',
            'Pima_Indians_Diabetes_Database', 'Pokemon', 'Predict_students_dropout_and_academic_success',
            'Protein_Localization_Sites', 'Quality_Prediction_in_a_Mining_Process', 'Red_Wine_Quality',
            'Retail_Rocket_ecommerce_dataset', 'Riiid_Answer_Correctness_Prediction',
            
            # S-T datasets
            'Santander_Customer_Satisfaction', 'Santander_Customer_Transaction_Prediction',
            'Seismic_Bumps', 'Sensorless_Drive_Diagnosis', 'Skin_Segmentation', 'Sonar',
            'Spambase', 'Steel_Plates_Faults', 'Student_Performance', 'Synthetic_Control_Chart_Time_Series',
            'Titanic_Machine_Learning_from_Disaster', 'Turkiye_Student_Evaluation',
            
            # U-Z datasets
            'UCI_Credit_Card', 'US_Accidents_Dec19', 'Used_Car_Dataset', 'Vehicle_Silhouettes',
            'Vertebral_Column', 'Video_Games_Sales', 'Wall_Following_Robot_Navigation_Data',
            'Water_Quality', 'White_Wine_Quality', 'Wine_Recognition', 'Yeast', 'Zoo',
            
            # Additional datasets (continuing the full list)
            'abalone', 'adult', 'agaricus-lepiota', 'allbp', 'allhyper', 'allhypo', 'allrep', 'analcatdata_aids',
            'analcatdata_asbestos', 'analcatdata_authorship', 'analcatdata_bankruptcy', 'analcatdata_boxing1',
            'analcatdata_boxing2', 'analcatdata_creditscore', 'analcatdata_cyyoung8092', 'analcatdata_cyyoung9302',
            'analcatdata_dmft', 'analcatdata_fraud', 'analcatdata_germangss', 'analcatdata_happiness',
            'analcatdata_japansolvent', 'analcatdata_lawsuit', 'analcatdata_neavote', 'analcatdata_supreme',
            'appendicitis', 'australian', 'auto', 'backache', 'balance-scale', 'balloons', 'banana',
            'bands', 'biomed', 'breast-cancer', 'breast-cancer-wisconsin', 'breast-w', 'bupa', 'car',
            'caravan', 'cars', 'chess', 'churn', 'clean1', 'clean2', 'cleve', 'cleveland',
            'cloud', 'cmc', 'coil2000', 'colic', 'collins', 'colorectal-histology', 'connbench',
            'contraceptive', 'corral', 'credit-approval', 'credit-g', 'cylinder-bands', 'dermatology',
            'diabetes', 'dis', 'dna', 'echoMonths', 'ecoli', 'electricity', 'eucalyptus', 'fahionmnist',
            'fallingobjects', 'fertility_Diagnosis', 'flare', 'german', 'glass', 'haberman', 'hayes-roth',
            'heart-c', 'heart-h', 'heart-statlog', 'hepatitis', 'hill-valley', 'horse-colic', 'house_16H',
            'house_8L', 'hungarian', 'hypothyroid', 'image-segmentation', 'ionosphere', 'iris', 'isolet',
            'kc1', 'kc2', 'kr-vs-kp', 'labor', 'led24', 'letter', 'libras', 'liver-disorders',
            'lupus', 'lymph', 'magic', 'mammographic', 'marketing', 'mfeat-factors', 'mfeat-fourier',
            'mfeat-karhunen', 'mfeat-morphological', 'mfeat-pixel', 'mfeat-zernike', 'molecular-biology_promoters',
            'monks-problems-1', 'monks-problems-2', 'monks-problems-3', 'mushroom', 'mux6', 'nursery',
            'optdigits', 'page-blocks', 'parkinsons', 'pendigits', 'phoneme', 'pima', 'poker',
            'postoperative', 'primary-tumor', 'prnn_crabs', 'prnn_fglass', 'prnn_synth', 'promoters',
            'ring', 'saheart', 'satellite', 'segment', 'shuttle', 'sick', 'sonar', 'soybean',
            'spambase', 'spect', 'spectf', 'splice', 'tae', 'texture', 'tic-tac-toe', 'titanic',
            'twonorm', 'vehicle', 'vertebral-column-2clases', 'vertebral-column-3clases', 'vote', 'vowel',
            'waveform-5000', 'wine', 'wine-quality-red', 'wine-quality-white', 'yeast', 'zoo'
        ]
        
        # Dataset metadata for common datasets
        self.dataset_metadata = {
            'adult': {'task_type': 'binclass', 'n_num_features': 6, 'n_cat_features': 8},
            'wine': {'task_type': 'multiclass', 'n_num_features': 13, 'n_cat_features': 0},
            'diabetes': {'task_type': 'binclass', 'n_num_features': 8, 'n_cat_features': 0},
            'iris': {'task_type': 'multiclass', 'n_num_features': 4, 'n_cat_features': 0},
            'credit-g': {'task_type': 'binclass', 'n_num_features': 7, 'n_cat_features': 13},
            'spambase': {'task_type': 'binclass', 'n_num_features': 57, 'n_cat_features': 0},
            'heart-statlog': {'task_type': 'binclass', 'n_num_features': 13, 'n_cat_features': 0},
            'sonar': {'task_type': 'binclass', 'n_num_features': 60, 'n_cat_features': 0},
            'ionosphere': {'task_type': 'binclass', 'n_num_features': 34, 'n_cat_features': 0},
            'glass': {'task_type': 'multiclass', 'n_num_features': 9, 'n_cat_features': 0},
            'vehicle': {'task_type': 'multiclass', 'n_num_features': 18, 'n_cat_features': 0},
            'abalone': {'task_type': 'multiclass', 'n_num_features': 7, 'n_cat_features': 1}
        }
    
    def download_talent_datasets(self) -> bool:
        """
        Download the TALENT zip file from Google Drive using the gdown library.
        """
        zip_path = self.data_path / 'benchmark_dataset.zip'
        # This is the working File ID from your successful bash script
        file_id = "1-dzY-BhMzcqjCM8vMTkVwa0hOYQ1598T" 
        
        # 1. Check if the data is already properly extracted
        self._resolve_dataset_root()
        if (self.dataset_root / 'adult').exists(): # Check for a known dataset
            logger.info(f"[TALENTLoader] TALENT datasets already extracted in {self.dataset_root}. Skipping download and extraction")
            return True

        # 2. Skip download if configured and file doesn't exist
        if self.skip_download and not zip_path.exists():
            logger.error(f"[TALENTLoader] Data not found at {self.data_path} and skip_download is True. Cannot proceed")
            return False

        try:
            # 3. Download the file if it doesn't exist
            if not zip_path.exists():
                logger.info(f"[TALENTLoader] Downloading TALENT zip file to {zip_path} using gdown...")
                # Use gdown to handle the download reliably
                gdown.download(id=file_id, output=str(zip_path), quiet=False)
                
                # Verify download was successful
                if not zip_path.exists() or zip_path.stat().st_size < 1000: # Check if file is tiny
                     raise IOError("Download failed, resulting file is missing or empty.")
                
                logger.info(f"[TALENTLoader] Download complete: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
            else:
                logger.info(f"[TALENTLoader] Zip file {zip_path} already exists. Skipping download")

            # 4. Extract the zip file
            logger.info(f"[TALENTLoader] Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_path)
        
            self._resolve_dataset_root()
            logger.info(f"[TALENTLoader] Successfully extracted datasets to {self.dataset_root}")
            return True
        
        except Exception as e:
            logger.error(f"[TALENTLoader] Failed during download or extraction: {e}", exc_info=True)
            # Clean up corrupted file on failure
            if zip_path.exists():
                try:
                    # Only remove if it's the bad, small file
                    if zip_path.stat().st_size < 1024 * 1024: # Less than 1MB
                        zip_path.unlink()
                        logger.info(f"[TALENTLoader] Removed corrupted/incomplete file: {zip_path}")
                except Exception as cleanup_e:
                     logger.error(f"[TALENTLoader] Error during file cleanup: {cleanup_e}")
            return False

    def _resolve_dataset_root(self) -> None:
        """Resolve the actual folder containing dataset directories after extraction."""
        try:
            # Prefer explicit 'data' subfolder if present (official zip structure)
            data_dir = self.data_path / 'data'
            if data_dir.exists() and any(d.is_dir() for d in data_dir.iterdir()):
                self.dataset_root = data_dir
                logger.info(f"[TALENTLoader] Resolved TALENT dataset root to nested folder: {self.dataset_root}")
                return

            # Fallback: if exactly one subdir exists, use it
            top_dirs = [d for d in self.data_path.iterdir() if d.is_dir() and d.name != '__pycache__']
            if len(top_dirs) == 1:
                candidate = top_dirs[0]
                if any(d.is_dir() for d in candidate.iterdir()):
                    self.dataset_root = candidate
                    logger.info(f"[TALENTLoader] Resolved TALENT dataset root to nested folder: {self.dataset_root}")
                    return

            # Default to data_path itself
            self.dataset_root = self.data_path
        except Exception as e:
            logger.warning(f"[TALENTLoader] Could not resolve dataset root, defaulting to {self.data_path}: {e}")
            self.dataset_root = self.data_path

    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Loads the specific TALENT dataset defined in the constructor config.
        This fulfills the abstract method requirement from BaseDataset.
        """
        if not self.dataset_name:
            raise ValueError("TALENTFullDataset was initialized without a 'dataset_name' in its config.")

        # Ensure the data is downloaded before trying to load
        if not self.download_talent_datasets():
            raise RuntimeError("Could not download or find TALENT datasets.")

        # Directly call the method to load a single dataset
        self.data, self.target = self._load_single_dataset(self.dataset_name)
        return self.data, self.target
    
    # def load_data(self, dataset_name: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    #     """
    #     Load TALENT dataset(s).
        
    #     Args:
    #         dataset_name: Specific dataset to load, or None to load all available datasets
            
    #     Returns:
    #         Tuple of (features, targets)
    #     """
    #     if not self.download_talent_datasets():
    #          raise RuntimeError("Could not download or find TALENT datasets.")

    #     if dataset_name is None:
    #         logger.info("Loading all available TALENT datasets (this is not recommended for single runs)")
    #         return self._load_all_datasets()
    #     else:
    #         # Set self.data and self.target as required by BaseDataset
    #         self.data, self.target = self._load_single_dataset(dataset_name)
    #         return self.data, self.target
    
    def _load_all_datasets(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load all available TALENT datasets."""
        # First try to download official datasets
        if not self.download_talent_datasets():
            raise RuntimeError("Could not download official TALENT datasets. No fallback allowed.")
        
        all_data = []
        all_targets = []
        successful_datasets = []
        
        # Load ALL available datasets from the extracted zip
        logger.info("[TALENTLoader] Loading ALL TALENT datasets from extracted zip file...")
        
        # Get all dataset directories from the extracted zip
        self._resolve_dataset_root()
        dataset_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir() and d.name != '__pycache__']
        logger.info(f"[TALENTLoader] Found {len(dataset_dirs)} dataset directories in extracted zip")
        
        for dataset_dir in dataset_dirs:
            name = dataset_dir.name
            try:
                X, y = self._load_single_dataset(name)
                all_data.append(X)
                all_targets.append(y)
                successful_datasets.append(name)
                logger.info(f"[TALENTLoader] Successfully loaded {name}: {X.shape}")
            except Exception as e:
                logger.warning(f"[TALENTLoader] Failed to load dataset {name}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No datasets could be loaded")
        
        logger.info(f"[TALENTLoader] Successfully loaded {len(successful_datasets)}/{len(self.all_datasets)} TALENT datasets: {successful_datasets}")
        
        # Combine all datasets
        X = pd.concat(all_data, ignore_index=True)
        y = np.concatenate(all_targets)
        
        return X, y
    
    def _load_single_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load a single TALENT dataset."""
        # Prefer official TALENT format only; do not fall back to other sources
        try:
            return self._load_talent_format(dataset_name)
        except Exception as e:
            logger.error(f"[TALENTLoader] TALENT format loading failed for {dataset_name}: {e}")
            raise RuntimeError(f"Failed to load {dataset_name} from official TALENT format. Ensure it exists under {self.data_path}.")
    
    def _load_talent_format(self, dataset_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load dataset in official TALENT format."""
        # Use resolved dataset root
        self._resolve_dataset_root()
        dataset_path = self.dataset_root / dataset_name
        
        logger.info(f"[TALENTLoader] Attempting to load {dataset_name} from TALENT format: {dataset_path}")
        
        if not dataset_path.exists():
            # List available directories for debugging
            available_dirs = [d.name for d in self.dataset_root.iterdir() if d.is_dir()]
            logger.error(f"[TALENTLoader] Dataset directory not found: {dataset_path}")
            logger.error(f"[TALENTLoader] Available directories: {available_dirs[:10]}...")
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        # Load TALENT format files
        info_path = dataset_path / 'info.json'
        if not info_path.exists():
            # List available files for debugging
            available_files = [f.name for f in dataset_path.iterdir()]
            logger.error(f"[TALENTLoader] info.json not found in {dataset_path}")
            logger.error(f"[TALENTLoader] Available files: {available_files}")
            raise FileNotFoundError(f"info.json not found in {dataset_path}")
        
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # Load features
        X_parts = []
        
        # Numerical features
        if info.get('n_num_features', 0) > 0:
            missing = [p.name for p in [dataset_path / 'N_train.npy', dataset_path / 'N_val.npy', dataset_path / 'N_test.npy'] if not p.exists()]
            if missing:
                logger.error(f"[TALENTLoader] Missing numeric feature files for {dataset_name}: {missing}. Expected: N_train.npy, N_val.npy, N_test.npy")
                logger.error("[TALENTLoader] How to fix: Ensure TALENT zip is fully extracted under talent_data/data/<dataset_name>/ and files exist")
                raise FileNotFoundError(f"Missing numeric feature files: {missing}")
            N_train = np.load(dataset_path / 'N_train.npy', allow_pickle=True)
            N_val = np.load(dataset_path / 'N_val.npy', allow_pickle=True)
            N_test = np.load(dataset_path / 'N_test.npy', allow_pickle=True)
            N_all = np.vstack([N_train, N_val, N_test])
            
            num_cols = [f'num_{i}' for i in range(N_all.shape[1])]
            X_num = pd.DataFrame(N_all, columns=num_cols)
            X_parts.append(X_num)
            print(X_num.head())
        # Categorical features
        if info.get('n_cat_features', 0) > 0:
            missing = [p.name for p in [dataset_path / 'C_train.npy', dataset_path / 'C_val.npy', dataset_path / 'C_test.npy'] if not p.exists()]
            if missing:
                logger.error(f"[TALENTLoader] Missing categorical feature files for {dataset_name}: {missing}. Expected: C_train.npy, C_val.npy, C_test.npy")
                logger.error("[TALENTLoader] How to fix: Ensure TALENT zip is fully extracted under talent_data/data/<dataset_name>/ and files exist")
                raise FileNotFoundError(f"Missing categorical feature files: {missing}")
            C_train = np.load(dataset_path / 'C_train.npy', allow_pickle=True)
            C_val = np.load(dataset_path / 'C_val.npy', allow_pickle=True)
            C_test = np.load(dataset_path / 'C_test.npy', allow_pickle=True)
            C_all = np.vstack([C_train, C_val, C_test])
            
            cat_cols = [f'cat_{i}' for i in range(C_all.shape[1])]
            X_cat = pd.DataFrame(C_all, columns=cat_cols)
            for col in cat_cols:
                X_cat[col] = X_cat[col].fillna('_missing_').astype(str)
            X_parts.append(X_cat)
        
        # Combine features
        if X_parts:
            X = pd.concat(X_parts, axis=1)
        else:
            raise ValueError(f"No features found for dataset {dataset_name}")
        
        # Load targets
        missing_y = [p.name for p in [dataset_path / 'y_train.npy', dataset_path / 'y_val.npy', dataset_path / 'y_test.npy'] if not p.exists()]
        if missing_y:
            logger.error(f"[TALENTLoader] Missing target files for {dataset_name}: {missing_y}. Expected: y_train.npy, y_val.npy, y_test.npy")
            logger.error("[TALENTLoader] How to fix: Ensure TALENT zip is fully extracted under talent_data/data/<dataset_name>/ and files exist")
            raise FileNotFoundError(f"Missing target files: {missing_y}")
        y_train = np.load(dataset_path / 'y_train.npy', allow_pickle=True)
        y_val = np.load(dataset_path / 'y_val.npy', allow_pickle=True)
        y_test = np.load(dataset_path / 'y_test.npy', allow_pickle=True)
        y = np.concatenate([y_train, y_val, y_test])

        print(X,y)
        
        logger.info(f"[TALENTLoader] Loaded TALENT dataset {dataset_name} in official format: {X.shape}")
        return X, y
    
    def _load_from_openml(self, dataset_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load dataset from OpenML."""
        try:
            import openml
            
            # Map dataset names to OpenML IDs
            openml_mapping = {
                'adult': 1590,
                'wine': 187,
                'diabetes': 37,
                'iris': 61,
                'credit-g': 31,
                'spambase': 44,
                'heart-statlog': 53,
                'sonar': 40,
                'ionosphere': 59,
                'glass': 41,
                'vehicle': 54,
                'abalone': 183,
                'mushroom': 24,
                'breast-cancer': 13,
                'hepatitis': 55,
                'german': 31,
                'australian': 143,
                'car': 21,
                'nursery': 26,
                'balance-scale': 11,
                'tic-tac-toe': 50,
                'chess': 4,
                'letter': 6,
                'optdigits': 28,
                'pendigits': 32,
                'satellite': 40900,
                'segment': 36,
                'shuttle': 40685,
                'splice': 46,
                'vowel': 307,
                'yeast': 181,
                'zoo': 62
            }
            
            if dataset_name not in openml_mapping:
                raise ValueError(f"No OpenML mapping for {dataset_name}")
            
            openml_id = openml_mapping[dataset_name]
            dataset = openml.datasets.get_dataset(openml_id)
            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
            
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            
            logger.info(f"[TALENTLoader] Loaded {dataset_name} from OpenML: {X.shape}")
            return X, y
            
        except Exception as e:
            logger.warning(f"[TALENTLoader] Failed to load {dataset_name} from OpenML: {e}")
            raise e
    
    def _load_from_sklearn(self, dataset_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load dataset from sklearn."""
        try:
            from sklearn import datasets
            
            sklearn_mapping = {
                'iris': datasets.load_iris,
                'wine': datasets.load_wine,
                'diabetes': datasets.load_diabetes,
                'breast-cancer': datasets.load_breast_cancer,
                'digits': datasets.load_digits
            }
            
            if dataset_name not in sklearn_mapping:
                raise ValueError(f"No sklearn mapping for {dataset_name}")
            
            data = sklearn_mapping[dataset_name]()
            X = pd.DataFrame(data.data, columns=data.feature_names if hasattr(data, 'feature_names') else [f'feature_{i}' for i in range(data.data.shape[1])])
            y = data.target
            
            logger.info(f"[TALENTLoader] Loaded {dataset_name} from sklearn: {X.shape}")
            return X, y
            
        except Exception as e:
            logger.warning(f"[TALENTLoader] Failed to load {dataset_name} from sklearn: {e}")
            raise e
    
    
    def get_available_datasets(self) -> List[str]:
        """
        Discovers available classification datasets by checking for 'info.json' files
        WITHOUT loading the actual data.
        """
        # 1. Ensure data is downloaded and extracted before trying to discover it.
        if not self.skip_download:
            if not self.download_talent_datasets():
                logger.error("[TALENTLoader] Cannot discover datasets because download/extraction failed")
                return []
        
        if not self.dataset_root.exists():
            logger.warning(f"[TALENTLoader] Dataset root for TALENT does not exist: {self.dataset_root}. Cannot discover datasets")
            return []

        # 2. Scan the directory for valid dataset folders
        discovered: List[str] = []
        for d in sorted([p for p in self.dataset_root.iterdir() if p.is_dir()]):
            info_path = d / 'info.json'
            if not info_path.exists():
                continue
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                # Filter for classification tasks only
                if 'class' in str(info.get('task_type', '')).lower():
                    discovered.append(d.name)
            except Exception:
                continue
        
        logger.info(f"[TALENTLoader] Discovered {len(discovered)} TALENT classification datasets at {self.dataset_root}")
        return discovered
    
    def get_dataset_info(self, dataset_name: str = None) -> Dict[str, Any]:
        """Get information about available datasets."""
        if dataset_name is not None:
            # Get information about a specific dataset
            if dataset_name not in self.all_datasets:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            return self.dataset_metadata.get(dataset_name, {
                'task_type': 'unknown',
                'n_num_features': 'unknown',
                'n_cat_features': 'unknown',
                'description': f'TALENT dataset: {dataset_name}'
            })
        else:
            # Get general information about all datasets
            return {
                'datasets': self.all_datasets,
                'total_datasets': len(self.all_datasets),
                'config': {
                    'preprocess': self.preprocess,
                    'normalize': self.normalize,
                    'handle_missing': self.handle_missing
                }
            }
    
    def list_priority_datasets(self) -> List[str]:
        """Get list of all available TALENT datasets (300+ datasets from official repository)."""
        # Return ALL available TALENT datasets from the official repository
        logger.info(f"[TALENTLoader] Returning ALL {len(self.all_datasets)} TALENT datasets from official repository")
        return self.all_datasets.copy()