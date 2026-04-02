import itertools
import multiprocessing
import os
import sys
import gc
import traceback
import json
import time
import glob
import numpy as np
from queue import Queue
from threading import Thread
from time import sleep
from typing import Tuple, Union, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import SimpleITK
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json, subfiles
from torch._dynamo import OptimizedModule
from tqdm import tqdm

import lesionlocator
from lesionlocator.preprocessing.resampling.default_resampling import compute_new_shape
from lesionlocator.configuration import default_num_processes
from lesionlocator.training.data_iterators import preprocessing_iterator_fromfiles
from lesionlocator.inference.export_prediction import export_prediction_from_logits
from lesionlocator.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from lesionlocator.utilities.file_path_utilities import check_workers_alive_and_busy
from lesionlocator.utilities.find_class_by_name import recursive_find_python_class
from lesionlocator.utilities.helpers import empty_cache, dummy_context
from lesionlocator.utilities.label_handling.label_handling import determine_num_input_channels
from lesionlocator.utilities.plans_handling.plans_handler import PlansManager
from lesionlocator.utilities.prompt_handling.prompt_handler import sparse_to_dense_prompt
from lesionlocator.utilities.surface_distance_based_measures import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient, compute_robust_hausdorff

from torch.cuda.amp import GradScaler
from torch.amp import autocast

from lesionlocator.training.data_wrapper import LesionTrackingDatasetWrapper
from lesionlocator.training.utils import training_collate_fn, tracking_collate_fn
import gc

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

def dice_loss(pred, target, epsilon=1e-6):
    """
    Dice loss function for segmentation training.
    
    Args:
        pred: Model predictions [B, C, H, W, D] (logits)
        target: Ground truth labels [B, H, W, D] (class indices)
        epsilon: Small value to avoid division by zero
        
    Returns:
        Dice loss value (1 - dice_coefficient)
    """
    pred_soft = torch.softmax(pred, dim=1)
    target_onehot = nn.functional.one_hot(target.long(), num_classes=pred.shape[1])
    target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float() if pred.dim() == 5 else target_onehot.permute(0, 3, 1, 2).float()
    
    dims = (0,) + tuple(range(2, pred.dim()))
    intersection = torch.sum(pred_soft * target_onehot, dims)
    union = torch.sum(pred_soft, dims) + torch.sum(target_onehot, dims)
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return 1 - dice.mean()

def unique_ids_to_indices(id_to_indices, unique_ids):
    indices = []
        
    #unique_ids_to_indices = {uid: [] for uid in unique_ids}
    for uni_id in unique_ids:
        indices.extend(id_to_indices.get(uni_id, []))
    return indices

# def create_cv_folds(input_files, prompt_files, output_files, n_folds=5, random_seed=42):
#     """
#     Create cross-validation folds for training data.
    
#     Args:
#         input_files: List of training input files
#         prompt_files: List of training prompt files  
#         output_files: List of training output files
#         n_folds: Number of folds for cross-validation
#         random_seed: Random seed for reproducibility
        
#     Returns:
#         List of fold dictionaries, each containing train and val splits
#     """
#     # Set random seed for reproducibility
#     np.random.seed(random_seed)
    
#     # Create indices for the files
#     indices = list(range(len(input_files)))
#     unique_ids = sorted(list(set([i.split('_')[-2] for i in input_files])))
    
#     id_to_indices = {}
#     for i, f in enumerate(input_files):
#         uid = f.split('_')[-2]
#         if uid not in id_to_indices:
#             id_to_indices[uid] = []
#         id_to_indices[uid].append(i)

#     # Create KFold splitter
#     kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
#     folds = []
#     for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(unique_ids)):            
#     #for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(indices)):
#         train_indices = unique_ids_to_indices(id_to_indices, [unique_ids[i] for i in train_indices])
#         val_indices = unique_ids_to_indices(id_to_indices, [unique_ids[i] for i in val_indices])

#         fold = {
#             'fold_idx': fold_idx,
#             'train': {
#                 'input_files': [input_files[i] for i in train_indices],
#                 'prompt_files': [prompt_files[i] for i in train_indices],
#                 'output_files': [output_files[i] for i in train_indices]
#             },
#             'val': {
#                 'input_files': [input_files[i] for i in val_indices],
#                 'prompt_files': [prompt_files[i] for i in val_indices], 
#                 'output_files': [output_files[i] for i in val_indices]
#             }
#         }
#         folds.append(fold)
#     return folds


# class LesionDatasetWrapper(IterableDataset):
#     """
#     PyTorch IterableDataset wrapper that preserves the existing multiprocessing pipeline
#     while providing PyTorch DataLoader compatibility for training.
    
#     This wrapper:
#     1. Preserves all existing multiprocessing data loading
#     2. Converts each lesion instance into a training sample
#     3. Provides PyTorch DataLoader compatibility
#     4. Maintains all preprocessing logic unchanged
    
#     Example usage:
#         # Create trainer instance
#         trainer = LesionLocatorSegmenter(device=torch.device('cuda'))
#         trainer.initialize_from_trained_model_folder(model_dir, track_dir, folds)
        
#         # Create datasets
#         train_dataset = trainer.create_training_dataset(
#             input_files=['img1.nii.gz', 'img2.nii.gz'],
#             prompt_files=['prompt1.nii.gz', 'prompt2.nii.gz'],
#             output_files=['out1', 'out2'],
#             prompt_type='box'
#         )
        
#         # Train
#         trainer.train(train_dataset, epochs=100, lr=1e-4)
#     """
#     def __init__(self, input_files, prompt_files, output_files, prompt_type, 
#                  plans_config, dataset_json, configuration_config, modality,
#                  num_processes=3, pin_memory=False, verbose=False, track=False):
#         self.input_files = input_files
#         self.prompt_files = prompt_files
#         self.output_files = output_files
#         self.prompt_type = prompt_type
#         self.plans_config = plans_config
#         self.dataset_json = dataset_json
#         self.configuration_config = configuration_config
#         self.modality = modality
#         self.num_processes = num_processes
#         self.pin_memory = pin_memory
#         self.verbose = verbose
#         self.track = track
        
#     def __len__(self):
#         """
#         Return an estimate of the dataset length for PyTorch DataLoader.
#         This is an approximation since the actual number of lesions per file varies.
#         """
#         # Estimate: assume average of 2-3 lesions per file
#         return len(self.input_files) 
        
#     def __iter__(self):
#         """
#         Create the multiprocessing data iterator and yield training samples.
#         This preserves the existing preprocessing pipeline completely.
#         """
#         data_iterator = preprocessing_iterator_fromfiles(
#             self.input_files, self.prompt_files, self.output_files,
#             self.prompt_type, self.plans_config, self.dataset_json,
#             self.configuration_config, self.modality, self.num_processes, self.pin_memory,
#             self.verbose, self.track
#         )

#         print('Data iterator created, yielding training samples...')

#         for preprocessed in data_iterator:
#             data = preprocessed['data']
#             prompt = preprocessed['prompt']
#             seg_mask = preprocessed['seg']
#             properties = preprocessed['data_properties']

#             # Convert each lesion instance into a training sample
#             for inst_id, p in enumerate(prompt):
#                 # print(f'Processing instance {inst_id}', flush=True)
#                 if len(p) == 0:
#                     continue

#                 # hard-coded for point prompts
#                 mask_id = inst_id + 1
#                 # mask_id = preprocessed['seg'][0, int(p[0]), int(p[1]), int(p[2])]
#                 gt_mask = (seg_mask == mask_id).astype(np.uint8)
#                 p_dense = sparse_to_dense_prompt(p, self.prompt_type, array=data)
                
#                 if p_dense is None:
#                     continue
                
#                 # Yield training sample - convert to torch tensors for consistent batching
#                 # Handle both numpy arrays and already converted tensors
#                 if isinstance(data, torch.Tensor):
#                     data_tensor = data.float()
#                 else:
#                     data_tensor = torch.from_numpy(data).float()
                
#                 if isinstance(p_dense, torch.Tensor):
#                     prompt_tensor = p_dense.float()
#                 else:
#                     prompt_tensor = torch.from_numpy(p_dense).float()
                
#                 if isinstance(gt_mask[0], torch.Tensor):
#                     target_tensor = gt_mask[0].long()
#                 else:
#                     target_tensor = torch.from_numpy(gt_mask[0]).long()
                
#                 yield {
#                     'data': data_tensor,                    # Input image [C, H, W, D]
#                     'prompt': prompt_tensor,               # Dense prompt [1, H, W, D]
#                     'target': target_tensor, 
#                     'properties': properties,               # Metadata
#                     'lesion_id': mask_id,                  # Lesion instance ID
#                     'filename': preprocessed['ofile']      # Original filename
#                 }


# def training_collate_fn(batch):
#     """
#     Efficiently collate a list of dicts into a dict of stacked tensors/lists.
#     Assumes all dicts have the same keys.
#     """
#     if len(batch) == 1:
#         return batch[0]

#     # Stack tensors for each key; keep lists for non-tensor values
#     collated = {}
#     for key in batch[0]:
#         values = [d[key] for d in batch]
#         # Stack if all values are tensors and have the same shape
#         if isinstance(values[0], torch.Tensor):
#             try:
#                 collated[key] = torch.stack(values, dim=0)
#             except Exception as e:
#                 print(f"Stacking failed for key '{key}': {e}")
#                 collated[key] = values  # fallback to list
#         else:
#             collated[key] = values
#     return collated


# def tracking_collate_fn(batch):
#     """
#     Custom collate function for tracking training samples.
#     Handles baseline data, follow-up data, baseline prompt, and target.
#     """
#     if len(batch) == 1:
#         return batch[0]
    
#     # Stack all batch items into proper tensors
#     batch_baseline = []
#     batch_followup = []
#     batch_prompts = []
#     batch_targets = []
#     batch_properties = []
#     batch_lesion_ids = []
#     batch_filenames = []

#     for item in batch:
#         batch_baseline.append(item['baseline_data'])
#         batch_followup.append(item['followup_data'])
#         batch_prompts.append(item['baseline_prompt'])
#         batch_targets.append(item['target'])
#         batch_properties.append(item['properties'])
#         batch_lesion_ids.append(item['lesion_id'])
#         batch_filenames.append(item['filename'])
    
#     # Stack tensors - all should have same dimensions due to preprocessing
#     try:
#         stacked_baseline = torch.stack(batch_baseline, dim=0)       # [B, C, H, W, D]
#         stacked_followup = torch.stack(batch_followup, dim=0)       # [B, C, H, W, D]
#         stacked_prompts = torch.stack(batch_prompts, dim=0)         # [B, 1, H, W, D]
#         stacked_targets = torch.stack(batch_targets, dim=0)         # [B, H, W, D]
        
#         return {
#             'baseline_data': stacked_baseline,
#             'followup_data': stacked_followup,
#             'baseline_prompt': stacked_prompts,
#             'target': stacked_targets,
#             'properties': batch_properties,
#             'lesion_id': batch_lesion_ids,
#             'filename': batch_filenames
#         }

#     except RuntimeError as e:
#         # Print shapes for debugging
#         print(f"Tracking batch stacking failed: {e}")
#         print(f"Baseline shapes: {[d.shape for d in batch_baseline]}")
#         print(f"Follow-up shapes: {[d.shape for d in batch_followup]}")
#         print(f"Prompt shapes: {[p.shape for p in batch_prompts]}")
#         print(f"Target shapes: {[t.shape for t in batch_targets]}")
#         # Fallback: process as batch_size=1
#         print(f"Warning: Could not stack tracking batch, falling back to single sample processing")
#         return batch[0]


# class LesionTrackingDatasetWrapper(IterableDataset):
#     """
#     PyTorch IterableDataset wrapper for tracking training that handles paired baseline and follow-up data.
    
#     This wrapper:
#     1. Loads baseline and follow-up image pairs
#     2. Handles baseline segmentation masks as tracking prompts
#     3. Provides PyTorch DataLoader compatibility
#     4. Maintains all preprocessing logic for tracking
    
#     Example usage:
#         trainer = LesionLocatorTrack(device=torch.device('cuda'))
#         trainer.initialize_from_trained_model_folder(model_dir, track_dir, folds)
        
#         train_dataset = trainer.create_tracking_dataset(
#             baseline_files=['bl1.nii.gz', 'bl2.nii.gz'],
#             followup_files=['fu1.nii.gz', 'fu2.nii.gz'],
#             baseline_seg_files=['seg1.nii.gz', 'seg2.nii.gz'],
#             followup_seg_files=['seg1_fu.nii.gz', 'seg2_fu.nii.gz'],
#             output_files=['out1', 'out2']
#         )
        
#         trainer.train_tracking(train_dataset, epochs=100, lr=1e-4)
#     """
#     def __init__(self, baseline_files, followup_files, baseline_seg_files, followup_seg_files, 
#                  output_files, plans_config, dataset_json, configuration_config, modality,
#                  num_processes=3, pin_memory=False, verbose=False):
#         self.baseline_files = baseline_files
#         self.followup_files = followup_files
#         self.baseline_seg_files = baseline_seg_files
#         self.followup_seg_files = followup_seg_files
#         self.output_files = output_files
#         self.plans_config = plans_config
#         self.dataset_json = dataset_json
#         self.configuration_config = configuration_config
#         self.modality = modality
#         self.num_processes = num_processes
#         self.pin_memory = pin_memory
#         self.verbose = verbose
        
#     def __len__(self):
#         """
#         Return an estimate of the dataset length for PyTorch DataLoader.
#         This is an approximation since the actual number of lesions per file varies.
#         """
#         # Estimate: assume average of 2-3 lesions per file pair
#         return len(self.baseline_files) 
        
#     def __iter__(self):
#         """
#         Create the tracking data iterator and yield training samples.
#         For tracking, we need baseline image, follow-up image, baseline segmentation as prompt,
#         and follow-up segmentation as target.
#         """
#         # For simplicity, we'll process pairs of baseline and follow-up data
#         # This creates separate iterators for baseline and follow-up data processing
        
#         print('Creating tracking data iterator...')
        
#         # Process each pair of files
#         for i in range(len(self.baseline_files)):
#             try:
#                 # Load and process baseline data with its segmentation
#                 baseline_iterator = preprocessing_iterator_fromfiles(
#                     [self.baseline_files[i]], [self.baseline_seg_files[i]], 
#                     [self.output_files[i] + '_baseline'], 'point',  # Use 'point' prompt type (centroid-based)
#                     self.plans_config, self.dataset_json, self.configuration_config, 
#                     self.modality, 1, self.pin_memory, self.verbose, track=False
#                 )
                
#                 # Load and process follow-up data with its segmentation  
#                 followup_iterator = preprocessing_iterator_fromfiles(
#                     [self.followup_files[i]], [self.followup_seg_files[i]],
#                     [self.output_files[i] + '_followup'], 'point',  # Use 'point' prompt type (centroid-based)
#                     self.plans_config, self.dataset_json, self.configuration_config,
#                     self.modality, 1, self.pin_memory, self.verbose, track=False
#                 )
                
#                 # Get preprocessed data from both iterators
#                 baseline_data_list = list(baseline_iterator)
#                 followup_data_list = list(followup_iterator)
                
#                 if len(baseline_data_list) == 0 or len(followup_data_list) == 0:
#                     print(f"Skipping pair {i} - no data loaded")
#                     continue
                
#                 baseline_preprocessed = baseline_data_list[0]  # Get first (and should be only) item
#                 followup_preprocessed = followup_data_list[0]
                
#                 baseline_data = baseline_preprocessed['data']           # [C, H, W, D] tensor
#                 baseline_seg = baseline_preprocessed['seg']             # [H, W, D] numpy array
#                 followup_data = followup_preprocessed['data']           # [C, H, W, D] tensor
#                 followup_seg = followup_preprocessed['seg']             # [H, W, D] numpy array
#                 properties = baseline_preprocessed['data_properties']   # Use baseline properties
                
#                 # Convert tensor to numpy if needed
#                 if isinstance(baseline_data, torch.Tensor):
#                     baseline_data = baseline_data.numpy()
#                 if isinstance(followup_data, torch.Tensor):
#                     followup_data = followup_data.numpy()
                
#                 # Process each lesion instance in the segmentation
#                 unique_ids = np.unique(baseline_seg)
#                 unique_ids = unique_ids[unique_ids > 0]  # Remove background
                
#                 for lesion_id in unique_ids:
#                     # Create binary mask for this specific lesion in baseline
#                     baseline_lesion_mask = (baseline_seg == lesion_id).astype(np.uint8)
#                     followup_lesion_mask = (followup_seg == lesion_id).astype(np.uint8)
                    
#                     # Skip if no corresponding lesion in follow-up
#                     if np.sum(followup_lesion_mask) == 0:
#                         # print(f"Skipping lesion {lesion_id} for {self.baseline_files[i]} - no corresponding lesion in follow-up")
#                         continue
                    
#                     # Convert to torch tensors
#                     baseline_tensor = torch.from_numpy(baseline_data).float()
#                     followup_tensor = torch.from_numpy(followup_data).float()
#                     baseline_prompt_tensor = torch.from_numpy(baseline_lesion_mask).float().unsqueeze(0)  # Add channel dim
#                     target_tensor = torch.from_numpy(followup_lesion_mask).long()
                    
#                     yield {
#                         'baseline_data': baseline_tensor,              # [C, H, W, D] - baseline image
#                         'followup_data': followup_tensor,             # [C, H, W, D] - follow-up image
#                         'baseline_prompt': baseline_prompt_tensor,     # [1, H, W, D] - baseline segmentation as prompt
#                         'target': target_tensor,                       # [H, W, D] - follow-up segmentation target
#                         'properties': properties,                      # Metadata
#                         'lesion_id': lesion_id,                       # Lesion instance ID
#                         'filename': f"{os.path.basename(self.baseline_files[i])}_to_{os.path.basename(self.followup_files[i])}_lesion_{lesion_id}"
#                     }
                    
#             except Exception as e:
#                 print(f"Error processing pair {i}: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 continue

class LesionLocatorTrack(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 visualize: bool = False,
                 adaptive_mode: bool = False):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        # Training-specific attributes
        self.optimizer = None
        self.loss_function = None
        self.scheduler = None
        self.scaler = GradScaler()

        self.training_mode = False

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device
        self.visualize = visualize
        self.adaptive_mode = adaptive_mode
        
        print('Adaptive mode: ', self.adaptive_mode)

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             model_track_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             modality: str = 'ct',
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        print("Loading tracking model")
        # print("Loading segmentation model.")
        if use_folds is None:
            use_folds = LesionLocatorSegmenter.auto_detect_available_folds(model_training_output_dir, checkpoint_name)
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        self.plans = plans
        plans_manager = PlansManager(plans)
        
        # Debug: Print plans structure for troubleshooting
        print("Plans structure debug:")
        print(f"  Plans type: {type(plans)}")
        print(f"  Plans keys: {list(plans.keys()) if isinstance(plans, dict) else 'Not a dict'}")
        if isinstance(plans, dict) and 'configurations' in plans:
            print(f"  Available configurations: {list(plans['configurations'].keys())}")
        else:
            print("  No 'configurations' key found in plans")

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                
                # Enhanced configuration mapping for better compatibility
                original_config = configuration_name
                if configuration_name == '3d_fullres_bs3':
                    configuration_name = '3d_fullres'
                    print(f"Warning: Mapped {original_config} to {configuration_name}")
                elif configuration_name not in plans_manager.plans.get('configurations', {}):
                    # Try common fallback mappings
                    fallback_mappings = {
                        '3d_fullres_bs2': '3d_fullres',
                        '3d_fullres_bs4': '3d_fullres', 
                        '3d_fullres_bs8': '3d_fullres',
                        '3d_lowres': '3d_fullres',
                        '2d_bs3': '2d',
                        '2d_bs2': '2d'
                    }
                    if configuration_name in fallback_mappings:
                        new_config = fallback_mappings[configuration_name]
                        if new_config in plans_manager.plans.get('configurations', {}):
                            configuration_name = new_config
                            print(f"Warning: Mapped {original_config} to {configuration_name}")
                        else:
                            print(f"Error: Neither {original_config} nor {new_config} found in plans")
                            print(f"Available configurations: {list(plans_manager.plans.get('configurations', {}).keys())}")
                    else:
                        print(f"Error: Configuration {configuration_name} not found in plans")
                        print(f"Available configurations: {list(plans_manager.plans.get('configurations', {}).keys())}")
                
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            # load previous fine-tuned model
            if os.path.exists(join(model_training_output_dir, f'fold_{f}', 'best_model.pth')):
                print(f'Loading fold {f} best model for segmentation')
                checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', 'best_model.pth'),
                            map_location=torch.device('cpu'), weights_only=False)

            parameters.append(checkpoint['network_weights'])

        self.configuration_name = configuration_name
        self.modality = modality
        configuration_manager = plans_manager.get_configuration(configuration_name, modality=modality)
        configuration_manager.set_preprocessor_name('TrainingPreprocessor')

        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(lesionlocator.__path__[0], "training", "LesionLocatorTrainer"),
                                                    trainer_name, 'lesionlocator.training.LesionLocatorTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in lesionlocator.training.LesionLocatorTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        
        # Store configuration name for checkpoint saving
        self.configuration_name = configuration_name

        network.load_state_dict(parameters[0])
        
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('LesionLocator_compile' in os.environ.keys()) and (os.environ['LesionLocator_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

        #Tracker network
        dataset_json_tracker = load_json(join(model_track_training_output_dir, 'dataset.json'))
        plans_tracker = load_json(join(model_track_training_output_dir, 'plans.json'))
        plans_manager_tracker = PlansManager(plans_tracker)

        parameters_tracker = []
        for i, f in enumerate(use_folds):
            print(f'Loading fold {f}')
            f = int(f) if f != 'all' else f
            checkpoint_tracker = torch.load(join(model_track_training_output_dir, f'fold_{f}', "checkpoint_final.pth"),
                                    map_location=torch.device('cpu'), weights_only=False)

            if i == 0:
                trainer_name_tracker = checkpoint_tracker['trainer_name']
                configuration_name_tracker = checkpoint_tracker['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint_tracker['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint_tracker.keys() else None

            # resume from a previous tracking model, 
            # TODO: update the final_checkpoint with the new tracking model
            if os.path.exists(join(model_track_training_output_dir, f'fold_{f}', 'best_tracking_model.pth')):
                print(f'Loading fold {f} best tracking model')
                checkpoint_tracker = torch.load(join(model_track_training_output_dir, f'fold_{f}', 'best_tracking_model.pth'),
                            map_location=torch.device('cpu'), weights_only=False)
            
            # load segmentation decoder for the tracker
            for key in checkpoint_tracker['network_weights'].keys():
                if 'unet.decoder' in key:
                    seg_key = key.replace('unet.', '')
                    if seg_key in checkpoint['network_weights'].keys():
                        checkpoint_tracker['network_weights'][key] = checkpoint['network_weights'][seg_key]
                    else:
                        print(f'Key {key} not in tracker network, skipping loading segmentation weights for it')

            parameters_tracker.append(checkpoint_tracker['network_weights'])

        configuration_manager_tracker = plans_manager_tracker.get_configuration(configuration_name_tracker)
        # set spacing
        configuration_manager.set_spacing([1.5, 1.5, 1.5])
        configuration_manager_tracker.set_spacing([1.5, 1.5, 1.5])
        # restore networks
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager_tracker, dataset_json_tracker)
        trainer_class = recursive_find_python_class(join(lesionlocator.__path__[0], "training", "LesionLocatorTrainer"),
                                                    trainer_name_tracker, 'lesionlocator.training.LesionLocatorTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name_tracker} in lesionlocator.training.LesionLocatorTrainer. '
                               f'Please place it there (in any .py file)!')
        network_tracker = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            configuration_manager.patch_size,
            enable_deep_supervision=False
        )
       
        self.plans_manager_tracker = plans_manager_tracker
        self.configuration_manager_tracker = configuration_manager_tracker
        self.list_of_parameters_tracker = parameters_tracker

        network_tracker.load_state_dict(parameters_tracker[0])

        self.network_tracker = network_tracker
        self.dataset_json_tracker = dataset_json_tracker
        self.trainer_name_tracker = trainer_name_tracker
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json_tracker)
        self.tile_step_size = 0.5
        self.use_gaussian = True
        self.use_mirroring = True
        # For LesionLocatorTrack, always use tracker spacing
        self.target_spacing = self.configuration_manager_tracker.spacing
        print('Using target spacing: ', self.target_spacing)
        print('Segmentation configuration: ', self.configuration_manager)
        print('Tracking configuration: ', self.configuration_manager_tracker)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds

    def predict_from_files(self,
                           source_folder_or_file: str,
                           output_folder_or_file: str,
                           prompt_folder_or_file: str,
                           prompt_type: str,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is the default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                      "So if there are 3 parts then valid part IDs are 0, 1, 2")
        if os.path.isdir(source_folder_or_file):
            assert os.path.isdir(output_folder_or_file) and os.path.isdir(prompt_folder_or_file), \
                "If '-i' is a folder then '-o' (output) and '-p' (prompt) must also be folders."
            # list and sort all the files
            input_files = subfiles(source_folder_or_file, suffix=self.dataset_json['file_ending'], join=True, sort=True)
            prompt_files_json = subfiles(prompt_folder_or_file, suffix='.json', join=True, sort=True)
            prompt_files_mask = subfiles(prompt_folder_or_file, suffix=self.dataset_json['file_ending'], join=True, sort=True)
            output_files = [join(output_folder_or_file, os.path.basename(i)) for i in input_files]
            
            # Assertions
            if len(input_files) == 0:
                print(f'No files found in {source_folder_or_file}')
                return
            assert len(prompt_files_json) == 0 or len(prompt_files_mask) == 0, \
                "Prompt folder must contain either json files or mask files, not both."
            assert len(input_files) == len(prompt_files_json) or len(input_files) == len(prompt_files_mask), \
                "Number of files in source folder and prompt folder must be the same."
            
            prompt_files = prompt_files_json if len(prompt_files_json) > 0 else prompt_files_mask
            
            # Check if the output folder exists
            if not os.path.isdir(output_folder_or_file):
                os.makedirs(output_folder_or_file)
            else:
                if not overwrite:
                    # Remove already predicted files from the lists
                    existing_files = [os.path.isfile(i) for i in output_files]
                    not_existing_indices = [i for i, j in enumerate(input_files) if j not in existing_files]
                    input_files = [input_files[i] for i in not_existing_indices]
                    prompt_files = [prompt_files[i] for i in not_existing_indices]
                    output_files = [output_files[i] for i in not_existing_indices]
        else:
            assert not os.path.isdir(prompt_folder_or_file), \
                "If '-i' is a file then '-p' (prompt) must also be files not folders."
            input_files = [source_folder_or_file]
            prompt_files = [prompt_folder_or_file]
            output_files = [join(output_folder_or_file, os.path.basename(source_folder_or_file))]

        # Truncate output files
        output_files = [i.replace(self.dataset_json['file_ending'], '') for i in output_files]
        data_iterator = preprocessing_iterator_fromfiles(input_files, prompt_files,
                                                output_files, prompt_type, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes_preprocessing, self.device.type == 'cuda',
                                                self.verbose_preprocessing, True)
       
        return self.predict_from_data_iterator(data_iterator, prompt_type, output_folder_or_file, num_processes_segmentation_export)


    def predict_from_data_iterator(self,
                                   data_iterator,
                                   prompt_type: str,
                                   output_folder_or_file: str,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        This function takes a data iterator and makes predictions and saves each instance (lesion) as a separate file.
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            error_all={'dice': {'mean':0, 'TP0':{'all':[], 'mean':0}, 'TP1': {'all':[], 'mean':0}, 'TP2': {'all':[], 'mean':0}}, 
               'nsd': {'mean':0, 'TP0':{'all':[], 'mean':0}, 'TP1': {'all':[], 'mean':0}, 'TP2': {'all':[], 'mean':0}},
               'hausdorff': {'mean':0, 'TP0':{'all':[], 'mean':0}, 'TP1': {'all':[], 'mean':0}, 'TP2': {'all':[], 'mean':0}},
               'lesion_found':{'all':0, 'mean':0, 'TP0':{'all':0, 'mean':0}, 'TP1':{'all':0, 'mean':0}, 'TP2':{'all':0, 'mean':0}},
               'lesion_all': {'all':0, 'TP0':{'all':0}, 'TP1':{'all':0}, 'TP2':{'all':0}}}
            dice_score_all = []
            hausdorff_score_all = []
            nsd_score_all = []
            metrics = {
                'dice': 0.0,
                'hausdorff': 0.0,
                'nsd': 0.0
            }
            for preprocessed in data_iterator:
                data = preprocessed['data']
                #baseline data, None for TP0 scans
                bl_data = preprocessed['bl_data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)
                ofile = preprocessed['ofile']
                print(f'\n === Predicting {os.path.basename(ofile)} === ')
                patient_tp = os.path.basename(ofile)
                timepoint = os.path.basename(ofile).split('_')[0]
                properties = preprocessed['data_properties']
                prompt = preprocessed['prompt']
                seg_mask = preprocessed['seg']
                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                if len(prompt) == 0:
                    print(f" No prompt found for {os.path.basename(ofile)}")
                else:
                    for inst_id, p in enumerate(prompt):
                        inst_id += 1
                        gt_mask = ((seg_mask == inst_id).astype(np.uint8))
                        if len(p) == 0:
                            print(f"--- No prompt found for Lesion ID {inst_id} ---")
                            continue                 
                        print(f'\n Lesion ID {inst_id}: ')
                        for k in error_all.keys():
                            if k == 'lesion_all' or k == 'lesion_found':
                                continue
                            if os.path.basename(ofile) not in error_all[k].keys():
                                error_all[k][patient_tp]={'mean':0, 'per_lesion':[]}
                        p_sparse = p
                        p = sparse_to_dense_prompt(p, prompt_type, array=data)
                        if p is None:
                            print(f" Invalid prompt found for {os.path.basename(ofile)}")
                            continue
                        #Check if there is an existing segmentation mask for the baseline image
                        tp_order = ['TP2', 'TP1', 'TP0']
                        current_tp = None
                        for tp in tp_order:
                            if tp in os.path.basename(ofile):
                                current_tp = tp
                                break

                        prev_tp = None
                        use_prev_tp = False
                        low_score = False
                        if current_tp == 'TP2':
                            # Try TP1 first, then TP0
                            for candidate in ['TP1', 'TP0']:
                                prev_tp_candidate = os.path.basename(ofile).replace('TP2', candidate)
                                if os.path.exists(os.path.join(output_folder_or_file, prev_tp_candidate + '_lesion_' + str(inst_id) + '.nii.gz')):
                                    prev_tp = prev_tp_candidate
                                    break
                        elif current_tp == 'TP1':
                            prev_tp_candidate = os.path.basename(ofile).replace('TP1', 'TP0')
                            if os.path.exists(os.path.join(output_folder_or_file, prev_tp_candidate + '_lesion_' + str(inst_id) + '.nii.gz')):
                                prev_tp = prev_tp_candidate
                        
                        if prev_tp is not None:
                            use_prev_tp = True
                            prev_seg_sitk = SimpleITK.ReadImage(os.path.join(output_folder_or_file, prev_tp+'_lesion_'+str(inst_id)+'.nii.gz'))
                            original_spacing = prev_seg_sitk.GetSpacing()[::-1]
                            print('Reading segmentation mask with spacing: ', original_spacing, ', target spacing is: ', self.target_spacing)
                            # Convert to numpy and compute new shape
                            prev_seg_np = SimpleITK.GetArrayFromImage(prev_seg_sitk)
                            new_shape = compute_new_shape(prev_seg_np.shape, original_spacing, self.target_spacing)
                            bl_spacing = (prev_seg_np.shape[0]* original_spacing[0] /  bl_data.shape[1],
                                        prev_seg_np.shape[1] * original_spacing[1] /  bl_data.shape[2],
                                        prev_seg_np.shape[2] * original_spacing[2] /  bl_data.shape[3])
                            #print('BL SPACING: ', bl_spacing)
                            #print('New shape for resampling: ', new_shape)
                            #print('BL DATA SHAPE FOR RESAMPLING: ', bl_data.shape)
                            prev_seg_resampled = self.configuration_manager.resampling_fn_seg(
                                prev_seg_np[None], 
                                bl_data.shape[1:], 
                                original_spacing, 
                                bl_spacing
                            )[0]
                            print('Use previous timepoint prediction as prompt: ', prev_tp+'_lesion_'+str(inst_id))
                            prompt_bl = torch.from_numpy(prev_seg_resampled).unsqueeze(0).to(self.device).half()
                            print('Resampled prompt shape: ', prompt_bl.shape)
                            # Predict the logits using the preprocessed data and the prompt
                            prediction = self.track_single_lesion(torch.from_numpy(bl_data[np.newaxis,:]).to(self.device), data.unsqueeze(0).to(self.device), prompt_bl.unsqueeze(0)).cpu()
                            seg = torch.softmax(prediction, 0).argmax(0)
                            pred = seg.detach().cpu().numpy().astype(np.uint8)
                            print('Prediction shape: ', pred.shape)
                            print('Ground truth shape: ',  gt_mask[0].shape)
                            dice_score = compute_dice_coefficient(gt_mask[0], pred)
                            if dice_score < 0.1:
                                print(f'Low Dice score {dice_score:.2f} for lesion {inst_id} at timepoint {timepoint}. Disabling tracking...')
                                low_score = True
                            
                        if (prev_tp is None) or (low_score and self.adaptive_mode):
                            use_prev_tp = False
                            print('Use current timepoint ground truth as prompt: ', p.shape)
                            # Predict the logits using the preprocessed data and the prompt
                            prediction = self.predict_logits_from_preprocessed_data(data, p).cpu()
                            seg = torch.softmax(prediction, 0).argmax(0)
                            pred = seg.detach().cpu().numpy().astype(np.uint8)
                            print('Prediction shape: ', pred.shape)
                            print('Ground truth shape: ', gt_mask[0].shape)
                            dice_score = compute_dice_coefficient(gt_mask[0], pred)
                        
                        error_all['lesion_all']['all']+=1
                        error_all['lesion_all'][timepoint]['all']+=1
                        if dice_score >= 0.1:
                            error_all['lesion_found']['all']+=1
                            error_all['lesion_found'][timepoint]['all']+=1
                        print('Dice Score: ', dice_score)
                        surface_distances = compute_surface_distances(gt_mask[0], pred, self.target_spacing)
                        hausdorff_score = compute_robust_hausdorff(surface_distances, 95)
                        nsd_score = compute_surface_dice_at_tolerance(surface_distances, 2)
                        
                        dice_score_all.append(dice_score)
                        hausdorff_score_all.append(hausdorff_score)
                        nsd_score_all.append(nsd_score)
                        metrics = {
                            'dice': dice_score,
                            'hausdorff': hausdorff_score,
                            'nsd': nsd_score
                        }
                        # Update all metrics in a loop
                        for metric_name, score in metrics.items():
                            error_all[metric_name][timepoint]['all'].append(score)
                            error_all[metric_name][patient_tp]['per_lesion'].append(score)
                        print('Avg Mean Dice: ', np.mean(dice_score_all))
                        print('Avg Mean Hausdorff: ',  np.mean(hausdorff_score_all))
                        print('Avg Mean NSD: ', np.mean(nsd_score_all))
                        print('Avg Lesion Detection Score: {:.2f}%'.format((error_all['lesion_found']['all'] / error_all['lesion_all']['all']) * 100))
                        with open(os.path.join(output_folder_or_file, 'error_dict.json'), 'w') as fjson:
                            json.dump(error_all, fjson)
                        print('----------')
                        out_file = ofile + f'_lesion_{inst_id}'
                        # Visualize the prediction
                        if self.visualize:
                            subplot_count = 3
                            if use_prev_tp:
                                subplot_count = 4
                            
                            # Find axial slice with most lesion pixels
                            mask_ones_gt_axial = np.where(gt_mask[0] == 1)
                            if len(mask_ones_gt_axial[0]) > 0:  # Check if mask is not empty
                                # Find the z-slice with most mask voxels for axial view
                                largest_mask_slice_id_axial = np.bincount(mask_ones_gt_axial[0]).argmax()
                            
                            # Find coronal slice with most lesion pixels
                            mask_ones_gt_coronal = np.where(gt_mask[0] == 1)
                            if len(mask_ones_gt_coronal[1]) > 0:  # Check if mask is not empty
                                # Find the y-slice with most mask voxels for coronal view
                                largest_mask_slice_id_coronal = np.bincount(mask_ones_gt_coronal[1]).argmax()

                            # Create subplot with 2 rows
                            fig, axs = plt.subplots(2, subplot_count, figsize=(subplot_count * 4, 8))
                            
                            # First row - Axial View
                            # Original img
                            axs[0,0].imshow(data[0][largest_mask_slice_id_axial, :, :].detach().cpu().numpy(), cmap='gray')
                            axs[0,0].set_title('Image (Axial)') 
                            axs[0,0].axis('off')
                            # Ground truth
                            axs[0,1].imshow(data[0][largest_mask_slice_id_axial, :, :].detach().cpu().numpy(), cmap='gray')
                            axs[0,1].imshow(gt_mask[0][largest_mask_slice_id_axial, :, :]*255, alpha=0.5)
                            axs[0,1].set_title('Ground truth') 
                            axs[0,1].axis('off')
                            # Predictions
                            axs[0,2].imshow(data[0][largest_mask_slice_id_axial, :, :].detach().cpu().numpy(), cmap='gray')
                            axs[0,2].imshow(pred[largest_mask_slice_id_axial, :, :], alpha=0.5)
                            axs[0,2].set_title('Prediction') 
                            axs[0,2].axis('off')

                            # Second row - Coronal View
                            # Original img
                            axs[1,0].imshow(data[0][:, largest_mask_slice_id_coronal, :].detach().cpu().numpy(), cmap='gray', origin='lower')
                            axs[1,0].set_title('Image (Coronal)') 
                            axs[1,0].axis('off')
                            # Ground truth
                            axs[1,1].imshow(data[0][:, largest_mask_slice_id_coronal, :].detach().cpu().numpy(), cmap='gray', origin='lower')
                            axs[1,1].imshow(gt_mask[0][:, largest_mask_slice_id_coronal, :]*255, alpha=0.5, origin='lower')
                            axs[1,1].set_title('Ground truth') 
                            axs[1,1].axis('off')
                            # Predictions
                            axs[1,2].imshow(data[0][:, largest_mask_slice_id_coronal, :].detach().cpu().numpy(), cmap='gray', origin='lower')
                            axs[1,2].imshow(pred[:, largest_mask_slice_id_coronal, :], alpha=0.5, origin='lower')
                            axs[1,2].set_title('Prediction') 
                            axs[1,2].axis('off')
                            if use_prev_tp:
                                prompt_bl = prompt_bl[0].detach().cpu().numpy()
                                try:
                                    # Axial view for baseline
                                    mask_ones_gt_axial_bl = np.where(prev_seg_resampled == 1)
                                    largest_mask_slice_id_axial_bl = np.bincount(mask_ones_gt_axial_bl[0]).argmax()
                                    axs[0,3].imshow(bl_data[0][largest_mask_slice_id_axial_bl, :, :], cmap='gray')
                                    axs[0,3].imshow(prev_seg_resampled[largest_mask_slice_id_axial_bl, :, :], alpha=0.5)
                                    axs[0,3].set_title('Baseline prompt') 
                                    axs[0,3].axis('off')

                                    # Coronal view for baseline
                                    mask_ones_gt_coronal_bl = np.where(prev_seg_resampled == 1)
                                    largest_mask_slice_id_coronal_bl = np.bincount(mask_ones_gt_coronal_bl[1]).argmax()
                                    axs[1,3].imshow(bl_data[0][:, largest_mask_slice_id_coronal_bl, :], cmap='gray', origin='lower')
                                    axs[1,3].imshow(prev_seg_resampled[:, largest_mask_slice_id_coronal_bl, :], alpha=0.5, origin='lower')
                                    axs[1,3].set_title('Baseline prompt') 
                                    axs[1,3].axis('off')
                                except Exception as e:
                                    print(f'Error visualizing baseline prompt: {e}')

                            fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.05, hspace=0.15)
                            plt.savefig(os.path.join(output_folder_or_file, f'{out_file}_dice_{dice_score:.2f}.png'), bbox_inches='tight')
                            plt.close()

                        
                        r.append(
                            export_pool.starmap_async(
                                export_prediction_from_logits,
                                ((prediction, properties, self.configuration_manager, self.plans_manager,
                                    self.dataset_json, out_file, False),)
                            )
                        )

                        # no multiprocessing
                        # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                        #     self.dataset_json, out_file, False)
                    for metric_name in metrics.keys():
                        error_all[metric_name][patient_tp]['mean'] = np.mean(error_all[metric_name][patient_tp]['per_lesion'])
                print(f'done with {os.path.basename(ofile)}')
            error_all['dice']['mean']= np.mean(dice_score_all)
            error_all['hausdorff']['mean'] = np.mean(hausdorff_score_all)
            error_all['nsd']['mean'] = np.mean(nsd_score_all)
            error_all['lesion_found']['mean'] = (error_all['lesion_found']['all']/error_all['lesion_all']['all'])*100
            for tp in ['TP0','TP1','TP2']:
                error_all['dice'][tp]['mean']=np.mean(error_all['dice'][tp]['all'])
                error_all['hausdorff'][tp]['mean']=np.mean(error_all['hausdorff'][tp]['all'])
                error_all['nsd'][tp]['mean']=np.mean(error_all['nsd'][tp]['all'])
                error_all['lesion_found'][tp]['mean'] = (error_all['lesion_found'][tp]['all']/error_all['lesion_all'][tp]['all'])*100
            with open(os.path.join(output_folder_or_file, 'error_dict.json'), 'w') as fjson:
                json.dump(error_all, fjson)
            
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret


    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor, dense_prompt: torch.Tensor) -> torch.Tensor:
        """
        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None

        # Add the dense prompt to the data
        data = torch.cat([data, dense_prompt], dim=0)

        for params in self.list_of_parameters:

            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)
        
            # why not leave prediction on device if perform_everything_on_device? Because this may cause the
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
            # this actually saves computation time
            if prediction is None:
                prediction = self.predict_sliding_window_return_logits(data, dense_prompt).to('cpu')
            else:
                prediction += self.predict_sliding_window_return_logits(data, dense_prompt).to('cpu')

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        if self.verbose: print('Prediction done')
        torch.set_num_threads(n_threads)
        return prediction

    def mirror_and_predict(self, x0, x1, prompt):
        output = self.network_tracker(x0, x1, prompt, is_inference=True)
        prediction = output[0] if isinstance(output, tuple) else output
        reg_loss = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        
        total_reg_loss = reg_loss.all_loss.item() if reg_loss is not None else 0
        num_predictions = 1  # Count original prediction
        
        if reg_loss is not None:
            print('Registration Loss:', reg_loss.all_loss.item())

        if self.use_mirroring:
            mirror_axes = [2, 3, 4]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]

            for axes in axes_combinations:
                mirror_output = self.network_tracker(torch.flip(x0, axes), torch.flip(x1, axes), torch.flip(prompt, axes), is_inference=True)
                mirror_pred = mirror_output[0] if isinstance(mirror_output, tuple) else mirror_output
                mirror_reg_loss = mirror_output[1] if isinstance(mirror_output, tuple) and len(mirror_output) > 1 else None
                
                if mirror_reg_loss is not None:
                    mirror_loss = mirror_reg_loss.all_loss
                    total_reg_loss += mirror_loss
                    num_predictions += 1
                
                prediction += torch.flip(mirror_pred, axes)
            
            prediction /= (len(axes_combinations) + 1)
            
            # Print average registration loss
            if num_predictions > 0:
                print('Average Registration Loss: {:.4f}'.format(total_reg_loss / num_predictions))
        
        prediction = prediction[0]
        return prediction

    @torch.inference_mode()
    def track_single_lesion(self, bl: torch.Tensor, fu: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=True) if self.device.type == 'cuda' else dummy_context():
            prediction = None
            for params in self.list_of_parameters_tracker: # fold iteration
                self.network_tracker.load_state_dict(params)
                self.network_tracker = self.network_tracker.to(self.device)
                self.network_tracker.eval()
                print('BL shape', bl.shape, bl.dtype)
                print('FU shape', fu.shape, fu.dtype)
                print('PROMPT shape',prompt.shape, prompt.dtype)
                if prediction is None:
                    prediction = self.mirror_and_predict(bl, fu, prompt).to('cpu')
                else:
                    prediction += self.mirror_and_predict(bl, fu, prompt).to('cpu')

            if len(self.list_of_parameters) > 1:
                prediction /= len(self.list_of_parameters)
            return prediction

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...], dense_prompt: torch.Tensor = None) -> List:
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            raise NotImplementedError('This predictor only supports 3D images')
        else:
            # No bbox will yield all slices
            if dense_prompt is None:
                steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                        self.tile_step_size)
                if self.verbose: print(
                    f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                    f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
                for sx in steps[0]:
                    for sy in steps[1]:
                        for sz in steps[2]:
                            slicers.append(
                                tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                    zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
            else:
                prompt_coords = torch.where(dense_prompt[0] > 0)
                prompt_coords = [int(prompt_coords[i].min().item()) for i in range(3)] + [int(prompt_coords[i].max().item()) for i in range(3)]
                prompt_coords = torch.tensor(prompt_coords)
                # Bbox focused
                if all(prompt_coords[3:] - prompt_coords[:3] < torch.tensor(self.configuration_manager.patch_size)):
                    # Return a slicer that covers the bbox in the middle of the patch
                    slicer = [slice(None)]
                    for i in range(3):
                        start = int((prompt_coords[i] + prompt_coords[i + 3] - self.configuration_manager.patch_size[i]) / 2)
                        end = start + self.configuration_manager.patch_size[i]
                        if start < 0:
                            start = 0
                            end = self.configuration_manager.patch_size[i]
                        elif end > image_size[i]:
                            end = image_size[i]
                            start = end - self.configuration_manager.patch_size[i]
                        slicer.append(slice(start, end))
                    slicers.append(slicer)
                else: # Non bbox focused, return all slices which overlap with the bbox
                    steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                            self.tile_step_size)
                    if self.verbose: print(
                        f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                        f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
                    for sx in steps[0]:
                        for sy in steps[1]:
                            for sz in steps[2]:
                                slc = tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                        zip((sx, sy, sz), self.configuration_manager.patch_size)]])
                                # Make sure the slicer has some overlap with the bbox
                                if (prompt_coords[0] < slc[1].stop and prompt_coords[3] > slc[1].start) and \
                                        (prompt_coords[1] < slc[2].stop and prompt_coords[4] > slc[2].start) and \
                                        (prompt_coords[2] < slc[3].stop and prompt_coords[5] > slc[3].start):
                                    slicers.append(slc)
        return slicers

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        print('NETWORK INPUT SHAPE: ', x.shape)
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                    
                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            # predicted_logits /= n_predictions
            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor, dense_prompt: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
       
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                       'constant', {'value': 0}, True,
                                                       None)

            # Make sure we get only the patches we need to predict, i.e. overlab with the prompt
            slicers = self._internal_get_sliding_window_slicers(data.shape[1:], dense_prompt)

            if self.perform_everything_on_device and self.device != 'cpu':
                # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                try:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                    empty_cache(self.device)
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
                predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                       self.perform_everything_on_device)

            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits


    def setup_tracking_training(self, learning_rate=1e-4, weight_decay=1e-5, use_scheduler=True, finetune_mode='all'):
        """
        Setup training components for tracking network: optimizer, loss function, and scheduler.
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            use_scheduler: Whether to use learning rate scheduler
            finetune_mode: Which part to finetune ('reg_net', 'unet', 'all')
        """
        if self.network_tracker is None:
            raise RuntimeError("Tracking network not initialized. Call initialize_from_trained_model_folder first.")
        
        # Set tracking network to training mode
        self.network_tracker.train()
        self.training_mode = True
        
        # Freeze/unfreeze parameters based on finetune_mode for tracking network
        self._configure_tracking_trainable_parameters(finetune_mode)
        
        # Get trainable parameters for optimizer
        trainable_params = [p for p in self.network_tracker.parameters() if p.requires_grad]
        
        # Setup optimizer with only trainable parameters
        self.optimizer = optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss function for tracking (segmentation + registration loss)
        self.loss_function = self._tracking_combined_loss
        
        # Setup learning rate scheduler
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            self.scheduler = None
        
        # Count and print trainable parameters
        total_params = sum(p.numel() for p in self.network_tracker.parameters())
        trainable_params_count = sum(p.numel() for p in self.network_tracker.parameters() if p.requires_grad)
        frozen_params_count = total_params - trainable_params_count
        
        print(f"Tracking training setup complete. Mode: {finetune_mode}, LR: {learning_rate}, Device: {self.device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params_count:,} ({trainable_params_count/1e6:.2f}M)")
        print(f"Frozen parameters: {frozen_params_count:,} ({frozen_params_count/1e6:.2f}M)")
        print(f"Trainable ratio: {100*trainable_params_count/total_params:.1f}%")

    def setup_training(self, learning_rate=1e-4, weight_decay=1e-5, use_scheduler=True, finetune_mode='all'):
        """
        Setup training components: optimizer, loss function, and scheduler.
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            use_scheduler: Whether to use learning rate scheduler
            finetune_mode: Which part to finetune ('encoder', 'decoder', 'all')
        """
        if self.network is None:
            raise RuntimeError("Network not initialized. Call initialize_from_trained_model_folder first.")
        
        # Set network to training mode
        self.network.train()
        self.training_mode = True
        
        # Freeze/unfreeze parameters based on finetune_mode
        self._configure_trainable_parameters(finetune_mode)
        
        # Get trainable parameters for optimizer
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        
        # Setup optimizer with only trainable parameters
        self.optimizer = optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss function (CrossEntropy + Dice loss)
        self.loss_function = self._combined_loss
        
        # Setup learning rate scheduler
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            self.scheduler = None
        
        # Count and print trainable parameters
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params_count = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        frozen_params_count = total_params - trainable_params_count
        
        print(f"Training setup complete. Mode: {finetune_mode}, LR: {learning_rate}, Device: {self.device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params_count:,} ({trainable_params_count/1e6:.2f}M)")
        print(f"Frozen parameters: {frozen_params_count:,} ({frozen_params_count/1e6:.2f}M)")
        print(f"Trainable ratio: {100*trainable_params_count/total_params:.1f}%")

    def _configure_trainable_parameters(self, finetune_mode='all'):
        """
        Configure which parameters are trainable based on finetune mode.
        
        Args:
            finetune_mode: 'encoder', 'decoder', or 'all'
        """
        print(f"Configuring trainable parameters for mode: {finetune_mode}")
        
        # Print network architecture for inspection
        print("\n" + "="*80)
        print("NETWORK ARCHITECTURE:")
        print("="*80)
        for name, param in self.network.named_parameters():
            print(f"{name}: {param.shape}")
        print("="*80 + "\n")
        
        if finetune_mode == 'all':
            # Enable gradients for all parameters
            for param in self.network.parameters():
                param.requires_grad = True
            print("All parameters enabled for training")
            
        elif finetune_mode == 'encoder':
            # Freeze all parameters first
            for param in self.network.parameters():
                param.requires_grad = False
            
            # Enable encoder parameters (encoder.stem.* and encoder.stages.*)
            enabled_count = 0
            
            for name, param in self.network.named_parameters():
                # Check if parameter belongs to encoder
                if name.startswith('encoder.'):
                    param.requires_grad = True
                    enabled_count += 1
                    print(f"  Enabled: {name}")
            
            print(f"Encoder mode: {enabled_count} parameter groups enabled")
            
        elif finetune_mode == 'decoder':
            # Freeze all parameters first
            for param in self.network.parameters():
                param.requires_grad = False
            
            # Enable decoder parameters (decoder.stages.*, decoder.transpconvs.*, decoder.seg_layers.*)
            enabled_count = 0
            
            for name, param in self.network.named_parameters():
                # Check if parameter belongs to decoder
                if name.startswith('decoder.'):
                    param.requires_grad = True
                    enabled_count += 1
                    print(f"  Enabled: {name}")
            
            print(f"Decoder mode: {enabled_count} parameter groups enabled")
            
        else:
            raise ValueError(f"Unknown finetune_mode: {finetune_mode}. Use 'encoder', 'decoder', or 'all'")
        
        # Print summary of enabled/disabled parameters
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    def _combined_loss(self, predictions, targets):
        """Combined CrossEntropy + Dice loss for segmentation training."""
        # CrossEntropy loss
        ce_loss = nn.CrossEntropyLoss()(predictions, targets.long())
        
        # Dice loss (use external function)
        dice_loss_val = dice_loss(predictions, targets)
        
        return ce_loss + dice_loss_val
    
    def _tracking_combined_loss(self, seg_output, reg_loss, targets, reg_loss_weight=1.0, scale_factor=1.0):
        """Combined segmentation and registration loss for tracking training."""
        # Segmentation loss (CrossEntropy + Dice)
        seg_loss = self._combined_loss(seg_output, targets)
        
        # Registration loss (if available)
        total_reg_loss = 0.0
        if reg_loss is not None:
            total_reg_loss = reg_loss.all_loss if hasattr(reg_loss, 'all_loss') else reg_loss
        
        # Combined loss
        total_loss = seg_loss + reg_loss_weight * total_reg_loss
        
        # Apply scaling for gradient accumulation
        # if scale_factor != 1.0:
        #     total_loss = total_loss * scale_factor
        
        return total_loss, seg_loss, total_reg_loss
    
    def _configure_tracking_trainable_parameters(self, finetune_mode='all'):
        """
        Configure which parameters are trainable for tracking network.
        
        Args:
            finetune_mode: 'reg_net', 'unet', or 'all'
        """
        print(f"Configuring tracking trainable parameters for mode: {finetune_mode}")
        
        if finetune_mode == 'all':
            # Enable gradients for all parameters
            for param in self.network_tracker.parameters():
                param.requires_grad = True
            print("All tracking parameters enabled for training")
            
        elif finetune_mode == 'reg_net':
            # Freeze all parameters first
            for param in self.network_tracker.parameters():
                param.requires_grad = False
            
            # Enable registration network parameters only
            enabled_count = 0
            for name, param in self.network_tracker.named_parameters():
                if name.startswith('reg_net.'):
                    param.requires_grad = True
                    enabled_count += 1
                    print(f"  Enabled: {name}")
            
            print(f"Registration network mode: {enabled_count} parameter groups enabled")
            
        elif finetune_mode == 'unet':
            # Freeze all parameters first
            for param in self.network_tracker.parameters():
                param.requires_grad = False
            
            # Enable UNet parameters only
            enabled_count = 0
            for name, param in self.network_tracker.named_parameters():
                if name.startswith('unet.'):
                    param.requires_grad = True
                    enabled_count += 1
                    print(f"  Enabled: {name}")
            
            print(f"UNet mode: {enabled_count} parameter groups enabled")
            
        else:
            raise ValueError(f"Unknown finetune_mode: {finetune_mode}. Use 'reg_net', 'unet', or 'all'")
        
        # Print summary of enabled/disabled parameters
        trainable_params = sum(p.numel() for p in self.network_tracker.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.network_tracker.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    def _visualize_validation_sample(self, data, target, prediction, filename, output_folder, epoch):
        """
        Visualize validation sample with overlaid masks in both axial and coronal views.
        Shows the cropped data that the network actually sees during training.
        
        Args:
            data: Input image [C, H, W, D] (cropped data seen by network)
            target: Ground truth mask [H, W, D] (cropped data seen by network)
            prediction: Predicted mask [H, W, D] (cropped data seen by network)
            filename: Sample filename
            output_folder: Output folder path
            epoch: Current epoch number
        """
        # Create epoch folder
        epoch_folder = os.path.join(output_folder, f'epoch_{epoch}')
        os.makedirs(epoch_folder, exist_ok=True)
        
        # Convert to numpy arrays
        if isinstance(data, torch.Tensor):
            data_np = data[0].cpu().numpy() if len(data.shape) > 3 else data.cpu().numpy()
        else:
            data_np = data[0] if len(data.shape) > 3 else data
            
        if isinstance(target, torch.Tensor):
            target_np = target.cpu().numpy()
        else:
            target_np = target
            
        if isinstance(prediction, torch.Tensor):
            pred_np = prediction.cpu().numpy()
        else:
            pred_np = prediction
        
        # Find the axial slice with the most target pixels (first dimension)
        target_sums_axial = np.sum(target_np, axis=(1, 2))  # Sum over each axial slice
        max_axial_slice = np.argmax(target_sums_axial) if np.max(target_sums_axial) > 0 else target_np.shape[0] // 2
        
        # Find the coronal slice with the most target pixels (second dimension)
        target_sums_coronal = np.sum(target_np, axis=(0, 2))  # Sum over each coronal slice
        max_coronal_slice = np.argmax(target_sums_coronal) if np.max(target_sums_coronal) > 0 else target_np.shape[1] // 2
        
        # Create visualization with 2 rows and 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # First row - Axial view
        # Original image
        axes[0, 0].imshow(data_np[max_axial_slice, :, :], cmap='gray')
        axes[0, 0].set_title('Original Image (Axial)')
        axes[0, 0].axis('off')
        
        # Ground truth overlay
        axes[0, 1].imshow(data_np[max_axial_slice, :, :], cmap='gray')
        axes[0, 1].imshow(target_np[max_axial_slice, :, :], alpha=0.5, cmap='Reds')
        axes[0, 1].set_title('Ground Truth (Axial)')
        axes[0, 1].axis('off')
        
        # Prediction overlay
        axes[0, 2].imshow(data_np[max_axial_slice, :, :], cmap='gray')
        axes[0, 2].imshow(pred_np[max_axial_slice, :, :], alpha=0.5, cmap='Blues')
        axes[0, 2].set_title('Prediction (Axial)')
        axes[0, 2].axis('off')
        
        # Second row - Coronal view
        # Original image
        axes[1, 0].imshow(data_np[:, max_coronal_slice, :], cmap='gray', origin='lower')
        axes[1, 0].set_title('Original Image (Coronal)')
        axes[1, 0].axis('off')
        
        # Ground truth overlay
        axes[1, 1].imshow(data_np[:, max_coronal_slice, :], cmap='gray', origin='lower')
        axes[1, 1].imshow(target_np[:, max_coronal_slice, :], alpha=0.5, cmap='Reds', origin='lower')
        axes[1, 1].set_title('Ground Truth (Coronal)')
        axes[1, 1].axis('off')
        
        # Prediction overlay
        axes[1, 2].imshow(data_np[:, max_coronal_slice, :], cmap='gray', origin='lower')
        axes[1, 2].imshow(pred_np[:, max_coronal_slice, :], alpha=0.5, cmap='Blues', origin='lower')
        axes[1, 2].set_title('Prediction (Coronal)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(epoch_folder, f'{filename}_axial_{max_axial_slice}_coronal_{max_coronal_slice}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _save_checkpoint(self, output_folder, filename, epoch, fold_idx=None, ckpt_path=None, prompt_type='point', best_val_loss=None):
        """Save model checkpoint and optionally save inference-compatible checkpoint."""
        os.makedirs(output_folder, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'network_weights': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'trainer_name': self.trainer_name,
        }
        if self.scheduler:
            checkpoint['scheduler_state'] = self.scheduler.state_dict()
        if best_val_loss is not None:
            checkpoint['best_val_loss'] = best_val_loss
            
        checkpoint_path = os.path.join(output_folder, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save inference-compatible checkpoint if ckpt_path is provided and this is a best model
        if ckpt_path and filename == 'best_model.pth' and fold_idx is not None:
            # Create inference-compatible directory structure
            optimized_folder = "point_optimized" if prompt_type == 'point' else "bbox_optimized"
            inference_dir = os.path.join(ckpt_path, 'LesionLocatorSeg', optimized_folder, f'fold_{fold_idx}')
            os.makedirs(inference_dir, exist_ok=True)
            
            # Get configuration name - this should match what's expected in the loading code
            # The loading code expects checkpoint['init_args']['configuration']
            config_name = getattr(self, 'configuration_name', None)
            if config_name is None:
                # Try to get it from the configuration manager or use a default
                config_name = 'default'  # This should be set to the actual configuration name used
                print(f"Warning: configuration_name not found, using default: {config_name}")
            
            # Create inference-compatible checkpoint with the same structure as the loaded checkpoints
            inference_checkpoint = {
                'network_weights': self.network.state_dict(),
                'trainer_name': self.trainer_name,
                'init_args': {
                    'configuration': config_name,
                },
                'inference_allowed_mirroring_axes': getattr(self, 'allowed_mirroring_axes', None),
            }
            
            inference_checkpoint_path = os.path.join(inference_dir, 'checkpoint_final.pth')
            torch.save(inference_checkpoint, inference_checkpoint_path)
            print(f"Inference-compatible checkpoint saved: {inference_checkpoint_path}")
            print(f"  - trainer_name: {self.trainer_name}")
            print(f"  - configuration: {config_name}")
            print(f"  - allowed_mirroring_axes: {getattr(self, 'allowed_mirroring_axes', None)}")
    
    def create_tracking_dataset(self, baseline_files, followup_files, baseline_seg_files, 
                               followup_seg_files, output_files, num_processes=3, verbose=False):
        """
        Create a tracking training dataset using paired baseline and follow-up data.
        
        Args:
            baseline_files: List of baseline image files
            followup_files: List of follow-up image files 
            baseline_seg_files: List of baseline segmentation files (as prompts)
            followup_seg_files: List of follow-up segmentation files (as targets)
            output_files: List of output file paths
            num_processes: Number of preprocessing workers
            verbose: Verbose output
            
        Returns:
            LesionTrackingDatasetWrapper instance
        """
        return LesionTrackingDatasetWrapper(
            baseline_files=baseline_files,
            followup_files=followup_files,
            baseline_seg_files=baseline_seg_files,
            followup_seg_files=followup_seg_files,
            output_files=output_files,
            plans_config=self.plans_manager_tracker.plans,
            dataset_json=self.dataset_json_tracker,
            configuration_config=self.configuration_name,
            modality=self.modality,
            num_processes=num_processes,
            pin_memory=self.device.type == 'cuda',
            verbose=verbose
        )
    
    def train_tracking(self, train_dataset, val_dataset=None, test_dataset=None, epochs=10, batch_size=1, lr=1e-4, 
                      device=None, output_folder=None, num_workers=0, finetune_mode='all', gradient_accumulation_steps=1):
        """
        Training function for tracking model using paired baseline and follow-up data.
        
        Args:
            train_dataset: LesionTrackingDatasetWrapper for training data
            val_dataset: LesionTrackingDatasetWrapper for validation data (optional)
            test_dataset: Dataset for test evaluation (optional)
            epochs: Number of training epochs
            batch_size: Batch size (typically 1 for tracking due to memory constraints)
            lr: Learning rate
            device: Training device (uses self.device if None)
            output_folder: Folder to save checkpoints
            num_workers: Number of preprocessing workers
            finetune_mode: Which part to finetune ('reg_net', 'unet', 'all')
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating
        """
        if device is None:
            device = self.device
            
        # Setup tracking training components
        self.setup_tracking_training(learning_rate=lr, finetune_mode=finetune_mode)
        
        # Move tracking network to device
        self.network_tracker.to(device)
        
        # Create DataLoaders with tracking collate function
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=tracking_collate_fn,
            num_workers=0  # Use 0 since we handle multiprocessing internally
        )
        
        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=1, # can only be 1 for validation
                collate_fn=tracking_collate_fn,
                num_workers=0
            )
        

        test_dataloader = None
        if test_dataset is not None:
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=1,
                collate_fn=tracking_collate_fn,
                num_workers=0
            )
        
        # Training history
        train_losses = []
        train_seg_losses = []
        train_reg_losses = []
        val_losses = []
        val_dice_scores = []
        test_dice_scores = []
        best_val_loss = float('inf')
        
        print(f"Starting tracking training for {epochs} epochs...")
        print(f"Device: {device}, Learning rate: {lr}")
        print(f"Batch size: {batch_size}, Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"Training samples: {len(train_dataset.baseline_files)}")
        if val_dataset is not None:
            print(f"Validation samples: {len(val_dataset.baseline_files)}")
        if test_dataset is not None:
            print(f"Test samples: {len(test_dataset.baseline_files)}")
        
        # Monitor OOM events
        oom_count = 0
        max_oom_retries = 6  # Allow more retries
        
        # Clear any existing GPU memory
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Print initial memory status
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Suggest memory-efficient settings if memory is limited
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_memory < 12:  # Less than 12GB
            print("WARNING: Limited GPU memory detected. Consider:")
            print("  - Using batch_size=1 and gradient_accumulation_steps=2-4")
            print("  - Setting finetune_mode='reg_net' to freeze UNet")
            print("  - Reducing input patch size in data preprocessing")
            
        # Memory optimization settings
        torch.backends.cudnn.benchmark = False  # Disable for consistent memory usage
        torch.backends.cudnn.deterministic = True
        
        for epoch in range(epochs):
            # # Training phase
            self.network_tracker.train()
            epoch_train_loss = 0.0
            epoch_seg_loss = 0.0
            epoch_reg_loss = 0.0
            num_train_batches = 0
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("Training...")
            
            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    # Extract tracking data from batch
                    baseline_data = batch['baseline_data'].to(device)      # [B, C, H, W, D]
                    followup_data = batch['followup_data'].to(device)      # [B, C, H, W, D]
                    baseline_prompt = batch['baseline_prompt'].to(device)  # [B, 1, H, W, D]
                    target = batch['target'].to(device)                    # [B, H, W, D]
                    
                    # Remove channel dimension from prompt - tracknet expects [B, H, W, D]
                    baseline_prompt = baseline_prompt.squeeze(1)  # [B, H, W, D]
                    
                    print(f"  Batch {batch_idx}: baseline_data shape: {baseline_data.shape}, followup_data shape: {followup_data.shape}, baseline_prompt shape: {baseline_prompt.shape}")

                    # Handle both batched and single sample data
                    if baseline_data.dim() == 4:  # Single sample [C, H, W, D]
                        baseline_data = baseline_data.unsqueeze(0)      # [1, C, H, W, D]
                        followup_data = followup_data.unsqueeze(0)      # [1, C, H, W, D]
                        baseline_prompt = baseline_prompt.unsqueeze(0)  # [1, H, W, D]
                        target = target.unsqueeze(0)                    # [1, H, W, D]
                    
                    # Clear gradients at start of accumulation cycle
                    #if batch_idx % gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()
                    
                    with autocast('cuda'):
                        # Training mode with proper x1_mask parameter
                        try:
                            seg_output, reg_loss, x1_mask_cropped = self.network_tracker(baseline_data, followup_data, baseline_prompt, 
                                                                                         is_inference=False, x1_mask=target)
                            import sys
                            sys.exit(0)

                            x1_mask_cropped = x1_mask_cropped.squeeze(1)  # [B, H, W, D]
                            # Calculate combined loss using the cropped target
                            # Calculate combined loss with scaling for gradient accumulation
                            total_loss, seg_loss, reg_loss_val = self._tracking_combined_loss(seg_output, reg_loss, x1_mask_cropped, scale_factor=1.0/gradient_accumulation_steps)
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                oom_count += 1
                                print(f"CUDA OOM in forward pass at batch {batch_idx} (OOM #{oom_count}). Clearing cache and skipping batch.")
                                print(f"Current GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
                                
                                if oom_count >= max_oom_retries:
                                    print(f"Too many OOM errors ({oom_count}). Consider reducing batch size or gradient accumulation steps.")
                                    print("Suggested fixes:")
                                    print("  1. Use --batch_size 1 --gradient_accumulation_steps 1")
                                    print("  2. Use --finetune reg_net to only train registration network")
                                    print("  3. Check if your GPU has enough memory for this model")
                                    raise RuntimeError(f"Too many OOM errors ({oom_count}). Training stopped.")
                                
                                # Aggressive cleanup
                                torch.cuda.empty_cache()
                                # Clear intermediate variables
                                del baseline_data, followup_data, baseline_prompt, target
                                if 'seg_output' in locals():
                                    del seg_output
                                if 'reg_loss' in locals():
                                    del reg_loss
                                if 'x1_mask_cropped' in locals():
                                    del x1_mask_cropped
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                continue
                            else:
                                raise e
                    
                    # Backward pass
                    try:
                        self.scaler.scale(total_loss).backward()
                        
                        # Only update weights every gradient_accumulation_steps
                        #if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            oom_count += 1
                            print(f"CUDA OOM in backward pass at batch {batch_idx} (OOM #{oom_count}). Clearing cache and skipping batch.")
                            print(f"Current GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
                            
                            if oom_count >= max_oom_retries:
                                print(f"Too many OOM errors ({oom_count}). Consider reducing batch size or gradient accumulation steps.")
                                print("Suggested fixes:")
                                print("  1. Use --batch_size 1 --gradient_accumulation_steps 1") 
                                print("  2. Use --finetune reg_net to only train registration network")
                                print("  3. Check if your GPU has enough memory for this model")
                                raise RuntimeError(f"Too many OOM errors ({oom_count}). Training stopped.")
                            
                            # Aggressive cleanup
                            torch.cuda.empty_cache()
                            # Clear all tensors
                            del baseline_data, followup_data, baseline_prompt, target
                            del seg_output, reg_loss, x1_mask_cropped, total_loss, seg_loss
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                            continue
                        else:
                            raise e
                    
                    # Update metrics (scale back the loss for display/tracking)
                    epoch_train_loss += total_loss.item() # * gradient_accumulation_steps  # Unscale for correct average
                    epoch_seg_loss += seg_loss.item()
                    epoch_reg_loss += reg_loss_val if isinstance(reg_loss_val, (int, float)) else reg_loss_val.item()
                    num_train_batches += 1
                    
                    # Clear intermediate tensors to free memory
                    del baseline_data, followup_data, baseline_prompt, target
                    del seg_output, reg_loss, x1_mask_cropped, total_loss, seg_loss
                    
                    # Periodic memory cleanup - more aggressive for tracking
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    
                    if batch_idx % 10 == 0:  # Less frequent reporting
                        print(f"  Batch {batch_idx}")
                        print(f"    Total Loss: {epoch_train_loss/num_train_batches:.4f}")
                        print(f"    Seg Loss: {epoch_seg_loss/num_train_batches:.4f}")
                        print(f"    Reg Loss: {epoch_reg_loss/num_train_batches:.4f}")
                        print(f"    GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                        print(f"    GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                        print(f"    Max GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
                        
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Compute average losses
            avg_train_loss = epoch_train_loss / max(num_train_batches, 1)
            avg_seg_loss = epoch_seg_loss / max(num_train_batches, 1)
            avg_reg_loss = epoch_reg_loss / max(num_train_batches, 1)
            
            train_losses.append(avg_train_loss)
            train_seg_losses.append(avg_seg_loss)
            train_reg_losses.append(avg_reg_loss)
            
            print(f"Training - Total Loss: {avg_train_loss:.4f}, Seg Loss: {avg_seg_loss:.4f}, Reg Loss: {avg_reg_loss:.4f}")
            
            # Validation phase
            if val_dataset is not None:
                self.network_tracker.eval()
                epoch_val_loss = 0.0
                epoch_val_dice_scores = []
                num_val_batches = 0
                
                print("Validating...")
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_dataloader):
                        try:
                            # Extract validation data from batch
                            baseline_data = batch['baseline_data'].to(device)
                            followup_data = batch['followup_data'].to(device)
                            baseline_prompt = batch['baseline_prompt'].to(device)
                            target = batch['target'].to(device)
                            filenames = batch['filename']
                            
                            # Remove channel dimension from prompt - tracknet expects [B, H, W, D]
                            baseline_prompt = baseline_prompt.squeeze(1)  # [B, H, W, D]
                            
                            # Handle both batched and single sample data
                            if baseline_data.dim() == 4:
                                baseline_data = baseline_data.unsqueeze(0)
                                followup_data = followup_data.unsqueeze(0)
                                baseline_prompt = baseline_prompt.unsqueeze(0)
                                target = target.unsqueeze(0)
                                filenames = [filenames]
                            
                            with autocast('cuda'):
                                # Training mode with proper x1_mask parameter
                                seg_output, reg_loss, x1_mask_cropped = self.network_tracker(baseline_data, followup_data, baseline_prompt, is_inference=False, x1_mask=target)
                                x1_mask_cropped = x1_mask_cropped.squeeze(1)  # [B, H, W, D]
                                total_loss, seg_loss, reg_loss_val = self._tracking_combined_loss(seg_output, reg_loss, x1_mask_cropped)
                            
                            epoch_val_loss += total_loss.item()
                            num_val_batches += 1
                            
                            # Compute dice scores for each sample in the batch
                            for i in range(baseline_data.shape[0]):
                                # Get predictions (convert to class predictions)
                                output_single = seg_output[i:i+1]
                                pred_probs = torch.softmax(output_single, dim=1)
                                pred_classes = torch.argmax(pred_probs, dim=1).squeeze(0)
                                
                                # Get target
                                target_single = target[i]
                                
                                # Convert to numpy for dice computation
                                pred_np = pred_classes.cpu().numpy()
                                target_np = target_single.cpu().numpy()
                                
                                # Compute Dice score
                                dice_score = compute_dice_coefficient(target_np, pred_np)
                                epoch_val_dice_scores.append(dice_score)
                            
                        except Exception as e:
                            print(f"Error in validation batch {batch_idx}: {e}")
                            traceback.print_exc()
                            continue
                
                # Compute averages
                avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
                avg_val_dice = np.mean(epoch_val_dice_scores) if epoch_val_dice_scores else 0.0
                
                val_losses.append(avg_val_loss)
                val_dice_scores.append(avg_val_dice)
                print(f"Validation Loss: {avg_val_loss:.4f}")
                print(f"Validation Dice Score: {avg_val_dice:.4f}")
                
                # Update learning rate scheduler
                if self.scheduler:
                    self.scheduler.step(avg_val_loss)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if output_folder:
                        self._save_tracking_checkpoint(output_folder, 'best_tracking_model.pth', epoch)
            
            # Test evaluation phase
            test_dice_score = 0.0
            if test_dataset is not None:
                print("Testing...")
                test_dice_score = self._evaluate_test_dataset(test_dataset, test_dataloader, device, epoch, output_folder)
                test_dice_scores.append(test_dice_score)
                print(f"Test Dice Score: {test_dice_score:.4f}")
            
            # Save periodic checkpoint
            if output_folder and (epoch + 1) % 10 == 0:
                self._save_tracking_checkpoint(output_folder, f'tracking_checkpoint_epoch_{epoch+1}.pth', epoch)
        
        # Save final checkpoint
        if output_folder:
            self._save_tracking_checkpoint(output_folder, 'final_tracking_checkpoint.pth', epochs-1)
            
        print("Tracking training completed!")
        return {
            'train_losses': train_losses,
            'train_seg_losses': train_seg_losses,
            'train_reg_losses': train_reg_losses,
            'val_losses': val_losses,
            'val_dice_scores': val_dice_scores,
            'test_dice_scores': test_dice_scores,
            'best_val_loss': best_val_loss
        }
    
    
    def _save_tracking_checkpoint(self, output_folder, filename, epoch):
        """Save tracking model checkpoint."""
        os.makedirs(output_folder, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'network_weights': self.network_tracker.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'trainer_name': self.trainer_name_tracker,
        }
        if self.scheduler:
            checkpoint['scheduler_state'] = self.scheduler.state_dict()
            
        checkpoint_path = os.path.join(output_folder, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"Tracking checkpoint saved: {checkpoint_path}")

    def _evaluate_test_dataset(self, test_dataset, test_dataloader, device, epoch, output_folder=None):
        """
        Evaluate the tracking model on test dataset and return average dice score.
        
        Args:
            test_dataset: Test dataset
            test_dataloader: Test data loader  
            device: Device for evaluation
            epoch: Current epoch number (for visualization folder naming)
            output_folder: Output folder for visualizations (optional)
            
        Returns:
            Average dice score across test samples
        """
        self.network_tracker.eval()
        test_dice_scores = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                try:
                    # Extract data based on dataset type
                    if hasattr(test_dataset, 'baseline_files'):  # LesionTrackingDatasetWrapper
                        # This is tracking test data                        
                        baseline_data = batch['baseline_data'].to(device)
                        followup_data = batch['followup_data'].to(device)  
                        baseline_prompt = batch['baseline_prompt'].to(device)
                        target = batch['target'].to(device)
                        filenames = batch['filename']
                        
                        # Remove channel dimension from prompt
                        baseline_prompt = baseline_prompt.squeeze(1)
                        
                        # Handle single sample case
                        if baseline_data.dim() == 4:
                            baseline_data = baseline_data.unsqueeze(0)
                            followup_data = followup_data.unsqueeze(0)
                            baseline_prompt = baseline_prompt.unsqueeze(0)
                            target = target.unsqueeze(0)
                            filenames = [filenames]
                        
                        # Forward pass
                        with autocast('cuda'):
                            # For inference, network_tracker returns only (seg_output, reg_loss)
                            network_output = self.network_tracker(
                                baseline_data, followup_data, baseline_prompt, is_inference=True
                            )
                            if len(network_output) == 3:
                                seg_output, reg_loss, x1_mask_cropped = network_output
                            else:
                                seg_output, reg_loss = network_output
                                x1_mask_cropped = None
                    
                    else:  # Regular LesionDatasetWrapper
                        # This is segmentation test data
                        data = batch['data'].to(device)
                        prompt = batch['prompt'].to(device)
                        target = batch['target'].to(device)
                        filenames = batch['filename']
                        
                        # Handle single sample case
                        if data.dim() == 4:
                            data = data.unsqueeze(0)
                            prompt = prompt.unsqueeze(0)
                            target = target.unsqueeze(0)
                            filenames = [filenames]
                        
                        # Combine input and prompt for segmentation network
                        combined_input = torch.cat([data, prompt], dim=1)
                        
                        # Forward pass through segmentation network
                        with autocast('cuda'):
                            seg_output = self.network(combined_input)
                    
                    # Compute dice scores for each sample in batch
                    for i in range(seg_output.shape[0]):
                        # Get predictions
                        output_single = seg_output[i:i+1] 
                        pred_probs = torch.softmax(output_single, dim=1)
                        pred_classes = torch.argmax(pred_probs, dim=1).squeeze(0)
                        
                        # Get target
                        target_single = target[i]
                        
                        # Convert to numpy for dice computation
                        pred_np = pred_classes.cpu().numpy()
                        target_np = target_single.cpu().numpy()
                        
                        # Compute Dice score
                        dice_score = compute_dice_coefficient(target_np, pred_np)
                        test_dice_scores.append(dice_score)
                        
                        # Visualize some test samples (every 10th sample and first 3 samples)
                        if output_folder and (len(test_dice_scores) <= 3 or len(test_dice_scores) % 10 == 0):
                            if hasattr(test_dataset, 'baseline_files'):  # Tracking data
                                self._visualize_tracking_test_sample(
                                    baseline_data[i], followup_data[i], baseline_prompt[i], 
                                    target_single, pred_classes, filenames[i], 
                                    output_folder, epoch, dice_score
                                )
                            else:  # Segmentation data
                                self._visualize_test_sample(
                                    data[i], target_single, pred_classes, filenames[i],
                                    output_folder, epoch, dice_score
                                )
                        
                except Exception as e:
                    print(f"Error in test evaluation batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Return average dice score
        return np.mean(test_dice_scores) if test_dice_scores else 0.0

    def _visualize_tracking_test_sample(self, baseline_data, followup_data, baseline_prompt, target, prediction, 
                                       filename, output_folder, epoch, dice_score):
        """
        Visualize tracking test sample with baseline, follow-up, prompt, target and prediction.
        """
        # Create test visualization folder
        test_vis_folder = os.path.join(output_folder, f'test_vis_epoch_{epoch}')
        os.makedirs(test_vis_folder, exist_ok=True)
        
        # Convert to numpy arrays
        baseline_np = baseline_data[0].cpu().numpy() if baseline_data.dim() > 3 else baseline_data.cpu().numpy()
        followup_np = followup_data[0].cpu().numpy() if followup_data.dim() > 3 else followup_data.cpu().numpy()
        prompt_np = baseline_prompt.cpu().numpy()
        target_np = target.cpu().numpy()
        pred_np = prediction.cpu().numpy()
        
        # Find the axial slice with the most target pixels
        target_sums = np.sum(target_np[0], axis=(1, 2))
        max_slice = np.argmax(target_sums) if np.max(target_sums) > 0 else target_np.shape[0] // 2

        ratio = max_slice / target_np.shape[1] 
        #max_slice = int(ratio * baseline_np.shape[0])
        
        # Ensure max_slice is within bounds for all arrays
        # min_depth = min(baseline_np.shape[0], prompt_np.shape[0], followup_np.shape[0], target_np.shape[0], pred_np.shape[0])
        max_slice = max(max_slice, 0)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # First row - baseline data
        bl_max_slice = int(ratio * baseline_np.shape[1])
        axes[0, 0].imshow(baseline_np[0, bl_max_slice, :, :], cmap='gray')
        axes[0, 0].set_title('Baseline Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(baseline_np[0, bl_max_slice, :, :], cmap='gray')  
        axes[0, 1].imshow(prompt_np[0, bl_max_slice, :, :], alpha=0.5, cmap='Greens')
        axes[0, 1].set_title('Baseline + Prompt')
        axes[0, 1].axis('off')
        
        fu_max_slice = int(ratio * followup_np.shape[1])
        axes[0, 2].imshow(followup_np[0, fu_max_slice, :, :], cmap='gray')
        axes[0, 2].set_title('Follow-up Image')
        axes[0, 2].axis('off')
        
        # Second row - targets and predictions
        axes[1, 0].imshow(followup_np[0, fu_max_slice, :, :], cmap='gray')
        axes[1, 0].imshow(target_np[0, max_slice, :, :], alpha=0.5, cmap='Reds') 
        axes[1, 0].set_title('Ground Truth')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(followup_np[0, fu_max_slice, :, :], cmap='gray')
        axes[1, 1].imshow(pred_np[0, max_slice, :, :], alpha=0.5, cmap='Blues')
        axes[1, 1].set_title('Prediction')
        axes[1, 1].axis('off')
        
        # Overlay comparison
        axes[1, 2].imshow(followup_np[0, fu_max_slice, :, :], cmap='gray')
        axes[1, 2].imshow(target_np[0, max_slice, :, :], alpha=0.3, cmap='Reds')
        axes[1, 2].imshow(pred_np[0, max_slice, :, :], alpha=0.3, cmap='Blues') 
        axes[1, 2].set_title('GT (Red) + Pred (Blue)')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Tracking Test Sample - Dice: {dice_score:.3f}')
        plt.tight_layout()
        
        # Save with descriptive filename
        safe_filename = filename.replace('/', '_').replace('\\', '_')
        save_path = os.path.join(test_vis_folder, f'{safe_filename}_dice_{dice_score:.3f}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_test_sample(self, data, target, prediction, filename, output_folder, epoch, dice_score):
        """
        Visualize segmentation test sample with image, target and prediction.
        """
        # Create test visualization folder  
        test_vis_folder = os.path.join(output_folder, f'test_vis_epoch_{epoch}')
        os.makedirs(test_vis_folder, exist_ok=True)
        
        # Convert to numpy arrays
        data_np = data[0].cpu().numpy() if data.dim() > 3 else data.cpu().numpy()
        target_np = target.cpu().numpy()
        pred_np = prediction.cpu().numpy()
        
        # Find the axial slice with the most target pixels
        target_sums = np.sum(target_np, axis=(1, 2))
        max_slice = np.argmax(target_sums) if np.max(target_sums) > 0 else target_np.shape[0] // 2
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(data_np[max_slice, :, :], cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(data_np[max_slice, :, :], cmap='gray')
        axes[1].imshow(target_np[max_slice, :, :], alpha=0.5, cmap='Reds')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(data_np[max_slice, :, :], cmap='gray')
        axes[2].imshow(pred_np[max_slice, :, :], alpha=0.5, cmap='Blues')
        axes[2].set_title('Prediction') 
        axes[2].axis('off')
        
        # Overlay comparison
        axes[3].imshow(data_np[max_slice, :, :], cmap='gray')
        axes[3].imshow(target_np[max_slice, :, :], alpha=0.3, cmap='Reds')
        axes[3].imshow(pred_np[max_slice, :, :], alpha=0.3, cmap='Blues')
        axes[3].set_title('GT (Red) + Pred (Blue)')
        axes[3].axis('off')
        
        plt.suptitle(f'Segmentation Test Sample - Dice: {dice_score:.3f}')
        plt.tight_layout()
        
        # Save with descriptive filename
        safe_filename = filename.replace('/', '_').replace('\\', '_')
        save_path = os.path.join(test_vis_folder, f'{safe_filename}_dice_{dice_score:.3f}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
        # def create_training_dataset(self, input_files, prompt_files, output_files, prompt_type,
    #                            num_processes=3, verbose=False, track=False):
    #     """
    #     Create a training dataset using the existing multiprocessing pipeline.
        
    #     Args:
    #         input_files: List of input image files
    #         prompt_files: List of prompt files (JSON or segmentation masks)
    #         output_files: List of output file paths
    #         prompt_type: Type of prompt ('point', 'box', etc.)
    #         num_processes: Number of preprocessing workers
    #         verbose: Verbose output
    #         track: Whether to include tracking data
            
    #     Returns:
    #         LesionDatasetWrapper instance
    #     """
    #     return LesionDatasetWrapper(
    #         input_files=input_files,
    #         prompt_files=prompt_files,
    #         output_files=output_files,
    #         prompt_type=prompt_type,
    #         plans_config = self.plans,
    #         # plans_manager=self.plans_manager,
    #         dataset_json=self.dataset_json,
    #         configuration_config = self.configuration_name,
    #         modality = self.modality,
    #         # configuration_manager=self.configuration_manager,
    #         num_processes=num_processes,
    #         pin_memory=self.device.type == 'cuda',
    #         verbose=verbose,
    #         track=track
    #     )

        # def train_cross_validation(self, train_input_files, train_prompt_files, train_output_files,
    #                           test_dataset=None, epochs=10, batch_size=2, lr=1e-4, 
    #                           device=None, output_folder=None, n_folds=5, num_workers=4, prompt_type='box',
    #                           ckpt_path=None, finetune_mode='all', train_fold=None):
    #     """
    #     Perform 5-fold cross-validation training.
        
    #     Args:
    #         train_input_files: Training input files
    #         train_prompt_files: Training prompt files  
    #         train_output_files: Training output files
    #         test_dataset: Test dataset for evaluation and visualization
    #         epochs: Number of epochs per fold
    #         batch_size: Batch size
    #         lr: Learning rate
    #         device: Training device
    #         output_folder: Output folder for checkpoints
    #         n_folds: Number of cross-validation folds
    #         num_workers: Number of preprocessing workers
    #         prompt_type: Type of prompt ('box', 'point', etc.)
    #     """
    #     if device is None:
    #         device = self.device
        
    #     print(f"Starting {n_folds}-fold cross-validation...")
    #     print(f"Total training samples: {len(train_input_files)}")
    #     print(f"Total prompt samples: {len(train_prompt_files)}")


    #     # Create cross-validation folds
    #     folds = create_cv_folds(train_input_files, train_prompt_files, train_output_files, n_folds)
        
    #     # Store results from all folds
    #     all_fold_results = []
        
    #     for fold_idx, fold_data in enumerate(folds):
    #         if fold_idx!=train_fold and train_fold is not None:
    #             continue
    #         print(f"\n{'='*50}")
    #         print(f"Starting: {fold_idx + 1}/{n_folds}")
    #         print(f"{'='*50}")
            
    #         # Reset network weights for each fold (reload from checkpoint)
    #         self.network.load_state_dict(self.list_of_parameters[0])
            
    #         # Train this fold
    #         fold_results = self.train_cv_fold(
    #             fold_data=fold_data,
    #             test_dataset=test_dataset,
    #             epochs=epochs,
    #             batch_size=batch_size,
    #             lr=lr,
    #             device=device,
    #             output_folder=output_folder,
    #             fold_idx=fold_idx,
    #             num_workers=num_workers,
    #             prompt_type=prompt_type,
    #             ckpt_path=ckpt_path,
    #             finetune_mode=finetune_mode
    #         )
            
    #         all_fold_results.append(fold_results)
    #         gc.collect()
    #         print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    #         print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    #         print(f"Fold {fold_idx+1} completed!")
    #         print(f"Best validation loss: {fold_results['best_val_loss']:.4f}")
    #         if fold_results['test_dice_scores']:
    #             print(f"Final test dice: {fold_results['test_dice_scores'][-1]:.4f}")

    #         torch.cuda.empty_cache()
        
    #     # Compute cross-validation statistics
    #     final_val_losses = [fold['val_losses'][-1] for fold in all_fold_results]
    #     best_val_losses = [fold['best_val_loss'] for fold in all_fold_results]
        
    #     if all_fold_results[0]['test_dice_scores']:
    #         final_test_dice = [fold['test_dice_scores'][-1] for fold in all_fold_results]
    #         print(f"\nCross-Validation Results:")
    #         print(f"Mean validation loss: {np.mean(final_val_losses):.4f} ± {np.std(final_val_losses):.4f}")
    #         print(f"Mean best validation loss: {np.mean(best_val_losses):.4f} ± {np.std(best_val_losses):.4f}")
    #         print(f"Mean test dice: {np.mean(final_test_dice):.4f} ± {np.std(final_test_dice):.4f}")
        
    #     # Save cross-validation summary
    #     if output_folder:
    #         cv_summary = {
    #             'n_folds': n_folds,
    #             'fold_results': all_fold_results,
    #             'mean_val_loss': np.mean(final_val_losses),
    #             'std_val_loss': np.std(final_val_losses),
    #             'mean_best_val_loss': np.mean(best_val_losses),
    #             'std_best_val_loss': np.std(best_val_losses)
    #         }
            
    #         if all_fold_results[0]['test_dice_scores']:
    #             cv_summary['mean_test_dice'] = np.mean(final_test_dice)
    #             cv_summary['std_test_dice'] = np.std(final_test_dice)
            
    #         with open(os.path.join(output_folder, 'cv_summary.json'), 'w') as f:
    #             json.dump(cv_summary, f, indent=2)
        
    #     return all_fold_results

    # def train_cv_fold(self, fold_data, test_dataset=None, epochs=10, batch_size=2, lr=1e-4, 
    #                   device=None, output_folder=None, fold_idx=None, num_workers=4, prompt_type='box',
    #                   ckpt_path=None, finetune_mode='all'):
    #     """
    #     Training function for a single cross-validation fold.
        
    #     Args:
    #         fold_data: Dictionary containing train and val data for this fold
    #         test_dataset: Test dataset for dice score computation and visualization
    #         epochs: Number of training epochs
    #         batch_size: Batch size
    #         lr: Learning rate
    #         device: Training device
    #         output_folder: Folder to save checkpoints
    #         fold_idx: Current fold index
    #         num_workers: Number of preprocessing workers
    #         prompt_type: Type of prompt ('box', 'point', etc.)
    #     """
    #     if device is None:
    #         device = self.device
            
    #     print(f"\n=== Training Fold {fold_idx} ===")
        
    #     # Create datasets for this fold
    #     train_dataset = self.create_training_dataset(
    #         input_files=fold_data['train']['input_files'],
    #         prompt_files=fold_data['train']['prompt_files'],
    #         output_files=fold_data['train']['output_files'],
    #         prompt_type=prompt_type,
    #         num_processes=num_workers,
    #         verbose=False,
    #         track=False
    #     )
        
    #     val_dataset = self.create_training_dataset(
    #         input_files=fold_data['val']['input_files'],
    #         prompt_files=fold_data['val']['prompt_files'],
    #         output_files=fold_data['val']['output_files'],
    #         prompt_type=prompt_type,
    #         num_processes=num_workers,
    #         verbose=False,
    #         track=False
    #     )
        
    #     # Setup training components
    #     self.setup_training(learning_rate=lr, finetune_mode=finetune_mode)
    #     self.network.to(device)
        
    #     # Create DataLoaders
    #     train_dataloader = DataLoader(
    #         train_dataset,
    #         batch_size=batch_size,
    #         collate_fn=training_collate_fn,
    #         num_workers=0,
    #     )
        
    #     val_dataloader = DataLoader(
    #         val_dataset,
    #         batch_size=batch_size,
    #         collate_fn=training_collate_fn,
    #         num_workers=0,
    #     )
        
    #     test_dataloader = None
    #     if test_dataset is not None:
    #         test_dataloader = DataLoader(
    #             test_dataset,
    #             batch_size=batch_size,
    #             collate_fn=training_collate_fn,
    #             num_workers=0,
    #         )
        
    #     # Training history for this fold
    #     fold_train_losses = []
    #     fold_val_losses = []
    #     fold_test_dice_scores = []
    #     best_val_loss = float('inf')

    #     print(f"Fold {fold_idx} - Training samples: {len(fold_data['train']['input_files'])}, Prompt samples: {len(fold_data['train']['prompt_files'])}")
    #     print(f"Fold {fold_idx} - Validation samples: {len(fold_data['val']['input_files'])}, Prompt samples: {len(fold_data['val']['prompt_files'])}")
    #     if test_dataset is not None:
    #         print(f"Test samples: {len(test_dataset.input_files)}")
        
    #     # Check for existing checkpoint to resume training
    #     start_epoch = 0
    #     if output_folder and fold_idx is not None:
    #         fold_folder = os.path.join(output_folder, f'fold_{fold_idx}')
    #         checkpoint_path = os.path.join(fold_folder, 'best_model.pth')
    #         if os.path.exists(checkpoint_path):
    #             print(f"Found existing checkpoint: {checkpoint_path}")
    #             try:
    #                 checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    #                 self.network.load_state_dict(checkpoint['network_weights'])
    #                 self.optimizer.load_state_dict(checkpoint['optimizer_state'])
    #                 if 'scheduler_state' in checkpoint and self.scheduler is not None:
    #                     self.scheduler.load_state_dict(checkpoint['scheduler_state'])
    #                 start_epoch = checkpoint['epoch'] + 1
    #                 best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    #                 print(f"Resuming training from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    #             except Exception as e:
    #                 print(f"Error loading checkpoint: {e}")
    #                 print("Starting fresh training...")
    #                 start_epoch = 0
    #         else:
    #             print("No existing checkpoint found, starting fresh training...")

    #     print(f"number of batches approximately: {len(train_dataloader)}")
    #     for epoch in range(start_epoch, epochs):
    #         # Training phase
    #         self.network.train()
    #         epoch_train_loss = 0.0
    #         num_train_batches = 0 
            
    #         print(f"\nFold {fold_idx}, Epoch {epoch+1}/{epochs}")
    #         print("Training...")
    #         file_names = set()

    #         for batch_idx, batch in enumerate(train_dataloader):
    #             try:
    #                 data = batch['data'].to(device)
    #                 prompt = batch['prompt'].to(device)
    #                 target = batch['target'].to(device)
                    
    #                 if data.dim() == 4:
    #                     data = data.unsqueeze(0)
    #                     prompt = prompt.unsqueeze(0)
    #                     target = target.unsqueeze(0)
                    
    #                 combined_input = torch.cat([data, prompt], dim=1)  
                    
    #                 self.optimizer.zero_grad()

    #                 with autocast('cuda'):
    #                     outputs = self.network(combined_input)
    #                     loss = self.loss_function(outputs, target)

    #                 # with autocast():
    #                 #     outputs = self.network(combined_input)
    #                 #     loss = self.loss_function(outputs, target)
                    
    #                 self.scaler.scale(loss).backward()
    #                 self.scaler.step(self.optimizer)
    #                 self.scaler.update()

    #                 # self.optimizer.zero_grad()
    #                 # outputs = self.network(combined_input)
    #                 # loss = self.loss_function(outputs, target)
    #                 # loss.backward()
    #                 # self.optimizer.step()
                                        
    #                 epoch_train_loss += loss.item()
    #                 num_train_batches += 1
                    
    #                 torch.cuda.empty_cache()
    #                 #if batch_idx % 10 == 0:
    #                 print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
    #                 print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    #                 print(f"  GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                        
    #             except Exception as e:
    #                 import sys
    #                 print(f"Error in training batch {batch_idx}: {e}")
    #                 sys.exit(1)

    #         avg_train_loss = epoch_train_loss / max(num_train_batches, 1)
    #         fold_train_losses.append(avg_train_loss)
    #         print(f"Training Loss: {avg_train_loss:.4f}")
            
    #         # Validation phase (for loss computation only)
    #         self.network.eval()
    #         epoch_val_loss = 0.0
    #         num_val_batches = 0
            
    #         print("Validating (loss computation on CV fold)...")
    #         with torch.no_grad():
    #             for batch_idx, batch in enumerate(val_dataloader):
    #                 try:
    #                     data = batch['data'].to(device)
    #                     prompt = batch['prompt'].to(device)
    #                     target = batch['target'].to(device)
                        
    #                     if data.dim() == 4:
    #                         data = data.unsqueeze(0)
    #                         prompt = prompt.unsqueeze(0)
    #                         target = target.unsqueeze(0)
                        
    #                     combined_input = torch.cat([data, prompt], dim=1)

    #                     with autocast('cuda'):
    #                         outputs = self.network(combined_input)
    #                         loss = self.loss_function(outputs, target)
                        
    #                     epoch_val_loss += loss.item()
    #                     num_val_batches += 1
                        
    #                     torch.cuda.empty_cache()
                    
    #                 except Exception as e:
    #                     import sys
    #                     print(f"Error in validation batch {batch_idx}: {e}")
    #                     sys.exit(1)

    #         avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
    #         fold_val_losses.append(avg_val_loss)
    #         print(f"Validation Loss: {avg_val_loss:.4f}")
            
    #         # Test phase (for dice computation and visualization)
    #         if test_dataset is not None:
    #             print("Testing (dice computation and visualization on test data)...")
    #             epoch_test_dice_scores = []
                
    #             with torch.no_grad():
    #                 for batch_idx, batch in enumerate(test_dataloader):
    #                     try:
    #                         data = batch['data'].to(device)
    #                         prompt = batch['prompt'].to(device)
    #                         target = batch['target'].to(device)
    #                         filenames = batch['filename']
                            
    #                         if data.dim() == 4:
    #                             data = data.unsqueeze(0)
    #                             prompt = prompt.unsqueeze(0)
    #                             target = target.unsqueeze(0)
    #                             filenames = [filenames]
                            
    #                         combined_input = torch.cat([data, prompt], dim=1)
    
    #                         with autocast('cuda'):
    #                             outputs = self.network(combined_input)
                            
    #                         # Process each sample for dice computation and visualization
    #                         for i in range(data.shape[0]):
    #                             filename = os.path.basename(filenames[i]).replace('.nii.gz', '')
                                
    #                             output_single = outputs[i:i+1]
    #                             pred_probs = torch.softmax(output_single, dim=1)
    #                             pred_classes = torch.argmax(pred_probs, dim=1).squeeze(0)
                                
    #                             data_single = data[i]
    #                             target_single = target[i]
                                
    #                             pred_cropped = pred_classes.cpu().numpy()
    #                             target_cropped = target_single.cpu().numpy()
                                
    #                             # Compute Dice score on test data
    #                             dice_score = compute_dice_coefficient(target_cropped, pred_cropped)
    #                             epoch_test_dice_scores.append(dice_score)
                                
    #                             if self.visualize:
    #                                 # Visualize test samples (first few batches only)
    #                                 if batch_idx < 1 and output_folder:
    #                                     test_viz_folder = os.path.join(output_folder, f'fold_{fold_idx}', 'test_visualizations')
    #                                     self._visualize_validation_sample(
    #                                         data_single, target_cropped, pred_cropped,
    #                                         f'{filename}_fold_{fold_idx}_epoch_{epoch}_batch_{batch_idx}_sample_{i}',
    #                                         test_viz_folder, epoch
    #                                     )
    #                         torch.cuda.empty_cache()
    #                     except Exception as e:
    #                         print(f"Error in test batch {batch_idx}: {e}")
    #                         sys.exit(1)

    #             avg_test_dice = np.mean(epoch_test_dice_scores) if epoch_test_dice_scores else 0.0
    #             fold_test_dice_scores.append(avg_test_dice)
    #             print(f"Test Dice Score: {avg_test_dice:.4f}")
            
    #         # Update learning rate scheduler
    #         if self.scheduler:
    #             self.scheduler.step(avg_val_loss)
            
    #         # Save best model for this fold (based on validation loss)
    #         if avg_val_loss < best_val_loss:
    #             best_val_loss = avg_val_loss
    #             if output_folder:
    #                 fold_folder = os.path.join(output_folder, f'fold_{fold_idx}')
    #                 self._save_checkpoint(fold_folder, 'best_model.pth', epoch, fold_idx=fold_idx, 
    #                                     ckpt_path=ckpt_path, prompt_type=prompt_type, best_val_loss=best_val_loss)
    #                 print(f"New best model saved for fold {fold_idx} (val_loss: {avg_val_loss:.4f})")
            
    #         # Save periodic checkpoint
    #         if output_folder and (epoch + 1) % 10 == 0:
    #             fold_folder = os.path.join(output_folder, f'fold_{fold_idx}')
    #             self._save_checkpoint(fold_folder, f'checkpoint_epoch_{epoch+1}.pth', epoch, fold_idx=fold_idx)
        
    #     # Save final checkpoint for this fold
    #     if output_folder:
    #         fold_folder = os.path.join(output_folder, f'fold_{fold_idx}')
    #         self._save_checkpoint(fold_folder, 'final_checkpoint.pth', epochs-1, fold_idx=fold_idx)
        
    #     return {
    #         'fold_idx': fold_idx,
    #         'train_losses': fold_train_losses,
    #         'val_losses': fold_val_losses,
    #         'test_dice_scores': fold_test_dice_scores,
    #         'best_val_loss': best_val_loss
    #     }


def train_from_prompt():
    import argparse
    parser = argparse.ArgumentParser(description='This function handles the LesionLocator single timepoint segmentation'
                                     'training using a point or 3D box prompt. Prompts can be the coordinates of a '
                                     'point or a 3D box as .json files or also (ground truth) instance segmentation maps.')
    # Tracking training arguments
    parser.add_argument('-bl', type=str, nargs='+', required=True,
                        help='Baseline image files or folder containing baseline images. Supports wildcards like TP0*. File endings should be .nii.gz')
    parser.add_argument('-fu', type=str, nargs='+', required=True,
                        help='Follow-up image files or folder containing follow-up images. Supports wildcards like TP1*. File endings should be .nii.gz')
    parser.add_argument('-pbl', type=str, nargs='+', required=True,
                        help='Baseline prompt/label files or folder with baseline segmentation masks. Supports wildcards like TP0*. File endings should be .nii.gz')
    parser.add_argument('-pfu', type=str, nargs='+', required=True,
                        help='Follow-up prompt/label files or folder with follow-up segmentation masks. Supports wildcards like TP1*. File endings should be .nii.gz')
    
    # Test/validation arguments for tracking (optional)
    parser.add_argument('-tbl', type=str, nargs='*', required=False,
                        help='Test baseline image files or folder containing test baseline images. Supports wildcards like TP0*. File endings should be .nii.gz')
    parser.add_argument('-tfu', type=str, nargs='*', required=False,
                        help='Test follow-up image files or folder containing test follow-up images. Supports wildcards like TP1*. File endings should be .nii.gz')
    parser.add_argument('-tpbl', type=str, nargs='*', required=False,
                        help='Test baseline prompt/label files or folder with test baseline segmentation masks. Supports wildcards like TP0*. File endings should be .nii.gz')
    parser.add_argument('-tpfu', type=str, nargs='*', required=False,
                        help='Test follow-up prompt/label files or folder with test follow-up segmentation masks. Supports wildcards like TP1*. File endings should be .nii.gz')
    
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If the folder does not exist it will be created. Training results and checkpoints'
                             'will be saved here.')
    parser.add_argument('-t', type=str, required=True, choices=['point', 'box'], default='box',
                        help='Specify the type of prompt. Options are "point" or "box". Default: box')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder of the LesionLocator model called "LesionLocatorCheckpoint"')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2).")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')
    parser.add_argument('--visualize', action='store_true', required=False, default=False,
                        help='Set this flag to visualize the prediction. This will open a napari viewer. You may need to '
                        ' run python -m pip install "napari[all]" first to use this feature.')
    # parser.add_argument('--track', action='store_true', required=False, default=False,
    #                     help='Set this flag to enable tracking. This will use the LesionLocatorTrack model to track lesions.')
    parser.add_argument('--modality', type=str, required=True, choices=['ct', 'pet'], default='ct', help="Use this to set the modality")
    # parser.add_argument('--adaptive_mode', action='store_true', help='Enable selection between segmentation and tracking based on Dice/NSD scores.')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, required=False, default=1,
                        help='Number of training epochs. Default: 1')
    parser.add_argument('--lr', type=float, required=False, default=1e-4,
                        help='Learning rate for training. Default: 1e-4')
    parser.add_argument('--batch_size', type=int, required=False, default=3,
                        help='Batch size for training. Default: 3')
    parser.add_argument('--gradient_accumulation_steps', type=int, required=False, default=1,
                        help='Number of steps to accumulate gradients before updating. Effective batch size = batch_size * gradient_accumulation_steps. Default: 1')
    parser.add_argument('--num_workers', type=int, required=False, default=4,
                        help='Number of workers for data loading. Default: 4')
    parser.add_argument('--ckpt_path', type=str, required=False, default=None,
                        help='Path to save inference-compatible checkpoints. Will create LesionLocatorSeg/point_optimized/fold_X structure. Default: None (no inference checkpoints saved)')
    parser.add_argument('--finetune', type=str, required=False, default='all', choices=['reg_net', 'unet', 'all'],
                        help='Which part of the tracking model to finetune. Options: reg_net (registration network only), unet (segmentation network only), all (both networks). Default: all')
    parser.add_argument('--train_fold', type=int, required=False, default=None,
                        help='Which fold configuration to use for training. Default: 0')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using LesionLocator:\n"
        "Rokuss, M., Kirchhoff, Y., Akbal, S., Kovacs, B., Roy, S., Ulrich, C., ... & Maier-Hein, K. (2025).\n"
        "LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging. "
        "CVPR.\n#######################################################################\n")

    args = parser.parse_args()
    
    # Complete the tracking training function
    if 'bl' in args:
        # This is tracking training mode - complete the function
        print("Starting tracking training mode...")
        
        if not isdir(args.o):
            maybe_mkdir_p(args.o)

        assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Got: {args.device}.'
        
        if args.device == 'cpu':
            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
            device = torch.device('cpu')
        elif args.device == 'cuda':
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            device = torch.device('cuda')
        else:
            device = torch.device('mps')

        # Initialize tracking trainer
        trainer = LesionLocatorTrack(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=device,
            verbose=args.verbose,
            allow_tqdm=True,
            verbose_preprocessing=args.verbose,
            visualize=args.visualize,
            adaptive_mode=False
        )
        
        # Load model checkpoints
        checkpoint_folder = join(args.m, 'LesionLocatorSeg', 'point_optimized')  # Use point optimized for tracking
        checkpoint_folder_track = join(args.m, 'LesionLocatorTrack')
        trainer.initialize_from_trained_model_folder(checkpoint_folder, checkpoint_folder_track, args.f, args.modality, "checkpoint_final.pth")
        
        # Helper function to process file arguments (handles folders, individual files, and wildcard expansions)
        # enable choose files from .txt
        def process_file_args(file_args, suffix, file_list_txt=None):
            all_files = []
            if file_list_txt is not None:
                print(f"Filtering files based on list in: {file_list_txt}")
                with open(file_list_txt, 'r') as f:
                    valid_files = set(line.strip() for line in f)
                
            for arg in file_args:
                if os.path.isdir(arg):
                    # If it's a directory, get all files with the specified suffix
                    from lesionlocator.utilities.file_path_utilities import subfiles
                    all_files.extend(subfiles(arg, suffix=suffix, join=True, sort=True))
                    if file_list_txt is not None:
                        all_files = [f for f in all_files if os.path.basename(f) in valid_files]
                else:
                    # If it's a file (or expanded from wildcard), add it directly
                    all_files.append(arg)
            return sorted(all_files)
        
        # Get training files
        baseline_files = process_file_args(args.bl, trainer.dataset_json['file_ending'])
        followup_files = process_file_args(args.fu, trainer.dataset_json['file_ending'])
        baseline_seg_files = process_file_args(args.pbl, trainer.dataset_json['file_ending'])
        followup_seg_files = process_file_args(args.pfu, trainer.dataset_json['file_ending'])
        
        # Match baseline and follow-up files based on patient ID (extract from filename)
        def extract_patient_id(filename):
            """Extract patient ID from filename (assumes pattern like PatientID_TP0_xxx.nii.gz or PatientID_TP1_xxx.nii.gz)"""
            basename = os.path.basename(filename)
            # Remove file extension and split by underscore
            parts = basename.replace(trainer.dataset_json['file_ending'], '').split('_')
            # Return the first part as patient ID (before _TP0 or _TP1)
            return parts[1] if parts else basename
        
        # Create dictionaries to match files by patient ID
        baseline_dict = {extract_patient_id(f): f for f in baseline_files}
        followup_dict = {extract_patient_id(f): f for f in followup_files}
        baseline_seg_dict = {extract_patient_id(f): f for f in baseline_seg_files}
        followup_seg_dict = {extract_patient_id(f): f for f in followup_seg_files}
        
        # Find common patient IDs that have all required files
        common_patients = set(baseline_dict.keys()) & set(followup_dict.keys()) & set(baseline_seg_dict.keys()) & set(followup_seg_dict.keys())
        
        print(f"Total baseline files: {len(baseline_files)}")
        print(f"Total follow-up files: {len(followup_files)}")
        print(f"Total baseline segmentation files: {len(baseline_seg_files)}")
        print(f"Total follow-up segmentation files: {len(followup_seg_files)}")
        print(f"Patients with complete data (baseline + follow-up + both segmentations): {len(common_patients)}")

        if len(common_patients) == 0:
            raise ValueError("No patients found with complete baseline and follow-up data. Check that your file naming convention allows matching.")
        
        # Create matched file lists
        matched_baseline_files = [baseline_dict[pid] for pid in sorted(common_patients)]
        matched_followup_files = [followup_dict[pid] for pid in sorted(common_patients)]
        matched_baseline_seg_files = [baseline_seg_dict[pid] for pid in sorted(common_patients)]
        matched_followup_seg_files = [followup_seg_dict[pid] for pid in sorted(common_patients)]
        
        # Update variables to use matched files
        baseline_files = matched_baseline_files
        followup_files = matched_followup_files
        baseline_seg_files = matched_baseline_seg_files
        followup_seg_files = matched_followup_seg_files

        # create cross fold validation dataset if specified
        if args.train_fold is not None:
            assert args.train_fold in [0, 1, 2, 3, 4], "train_fold must be one of [0, 1, 2, 3, 4]"
            # Split dataset into training and validation based on fold
            total_samples = len(baseline_files)
            fold_size = total_samples // 5
            val_start = args.train_fold * fold_size
            val_end = (args.train_fold + 1) * fold_size if args.train_fold < 4 else total_samples
            
            import random
            # use seed for reproducibility
            random.seed(42)
            total_indices = list(range(total_samples))
            random.shuffle(total_indices)
            val_indices = total_indices[val_start:val_end]
            train_indices = total_indices[:val_start] + total_indices[val_end:]
        else:
            # Use all data for training if no fold specified
            train_indices = list(range(len(baseline_files)))
            val_indices = []
        # Create output file names
        output_files = [join(args.o, 'train_' + os.path.basename(i).replace(trainer.dataset_json['file_ending'], '')) for i in baseline_files]
        
        # Create training dataset
        train_dataset = trainer.create_tracking_dataset(
            baseline_files=[baseline_files[i] for i in train_indices],
            followup_files=[followup_files[i] for i in train_indices],
            baseline_seg_files=[baseline_seg_files[i] for i in train_indices],
            followup_seg_files=[followup_seg_files[i] for i in train_indices],
            output_files=[output_files[i] for i in train_indices],
            num_processes=args.npp,
            verbose=args.verbose
        )
        
        val_dataset = trainer.create_tracking_dataset(
            baseline_files=[baseline_files[i] for i in val_indices],
            followup_files=[followup_files[i] for i in val_indices],
            baseline_seg_files=[baseline_seg_files[i] for i in val_indices],
            followup_seg_files=[followup_seg_files[i] for i in val_indices],
            output_files=[join(args.o, 'val_' + os.path.basename(baseline_files[i]).replace(trainer.dataset_json['file_ending'], '')) for i in val_indices],
            num_processes=args.npp,
            verbose=args.verbose
        ) if val_indices else None
                    
        # # Create validation dataset (none for tracking mode)
        # val_dataset = None

        test_dataset = None
        if args.tbl and args.tfu and args.tpbl and args.tpfu:
            test_baseline_files = process_file_args(args.tbl, trainer.dataset_json['file_ending'])
            test_followup_files = process_file_args(args.tfu, trainer.dataset_json['file_ending'])
            test_baseline_seg_files = process_file_args(args.tpbl, trainer.dataset_json['file_ending'])
            test_followup_seg_files = process_file_args(args.tpfu, trainer.dataset_json['file_ending'])
            
            # Match test files by patient ID
            test_baseline_dict = {extract_patient_id(f): f for f in test_baseline_files}
            test_followup_dict = {extract_patient_id(f): f for f in test_followup_files}
            test_baseline_seg_dict = {extract_patient_id(f): f for f in test_baseline_seg_files}
            test_followup_seg_dict = {extract_patient_id(f): f for f in test_followup_seg_files}
            
            common_test_patients = set(test_baseline_dict.keys()) & set(test_followup_dict.keys()) & set(test_baseline_seg_dict.keys()) & set(test_followup_seg_dict.keys())
            
            if len(common_test_patients) == 0:
                raise ValueError("No patients found with complete test baseline and follow-up data. Check that your file naming convention allows matching.")
            
            matched_test_baseline_files = [test_baseline_dict[pid] for pid in sorted(common_test_patients)]
            matched_test_followup_files = [test_followup_dict[pid] for pid in sorted(common_test_patients)]
            matched_test_baseline_seg_files = [test_baseline_seg_dict[pid] for pid in sorted(common_test_patients)]
            matched_test_followup_seg_files = [test_followup_seg_dict[pid] for pid in sorted(common_test_patients)]
            
            test_output_files = [join(args.o, 'test_' + os.path.basename(i).replace(trainer.dataset_json['file_ending'], '')) for i in matched_test_baseline_files]
            
            test_dataset = trainer.create_tracking_dataset(
                baseline_files=matched_test_baseline_files,
                followup_files=matched_test_followup_files,
                baseline_seg_files=matched_test_baseline_seg_files,
                followup_seg_files=matched_test_followup_seg_files,
                output_files=test_output_files,
                num_processes=args.npp,
                verbose=args.verbose
            )
            print(f"Test dataset created with {len(matched_test_baseline_files)} samples")
        
        print(f"Training dataset created with {len(baseline_files)} samples")
        
        # Start tracking training
        training_results = trainer.train_tracking(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            output_folder=args.o,
            num_workers=0,  # Set to 0 for tracking
            finetune_mode=args.finetune,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        
        print("Tracking training completed!")
        print(f"Final training loss: {training_results['train_losses'][-1]:.4f}")
        if training_results['val_losses']:
            print(f"Final validation loss: {training_results['val_losses'][-1]:.4f}")
            print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        if training_results['test_dice_scores']:
            print(f"Final test dice score: {training_results['test_dice_scores'][-1]:.4f}")
            print(f"Best test dice score: {max(training_results['test_dice_scores']):.4f}")
        
        return  # Exit after tracking training
        
    # Continue with segmentation training if not tracking mode
    args.f = [i if i == 'all' else int(i) for i in args.f]

    if not isdir(args.o):
        print(args.o)
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = LesionLocatorTrack(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                allow_tqdm=not args.disable_progress_bar,
                                verbose_preprocessing=args.verbose,
                                visualize=args.visualize,
                                track=True,
                                adaptive_mode=False)
    optimized_ckpt = "bbox_optimized" if args.t == 'box' else "point_optimized"
    checkpoint_folder = join(args.m, 'LesionLocatorSeg', optimized_ckpt)
    checkpoint_folder_track = join(args.m, 'LesionLocatorTrack')
    predictor.initialize_from_trained_model_folder(checkpoint_folder, checkpoint_folder_track, args.f, args.modality, "checkpoint_final.pth")
    
    # Helper function to process file arguments (handles folders, individual files, and wildcard expansions)
    # Training mode
    print("Starting training mode...")
    
    # Print checkpoint path information if provided
    if args.ckpt_path:
        optimized_folder = "point_optimized" if args.t == 'point' else "bbox_optimized"
        print(f"Inference-compatible checkpoints will be saved to:")
        print(f"  {args.ckpt_path}/LesionLocatorSeg/{optimized_folder}/fold_X/checkpoint_final.pth")
        print(f"  This structure is compatible with the inference code.")
    else:
        print("No inference checkpoint path specified (--ckpt_path). Only training checkpoints will be saved.")
    
    # Print fine-tuning mode information
    print(f"Fine-tuning mode: {args.finetune}")
    if args.finetune == 'reg_net':
        print("  Only registration network parameters will be trained (U-Net frozen)")
    elif args.finetune == 'unet':
        print("  Only U-Net segmentation parameters will be trained (registration network frozen)")
    elif args.finetune == 'all':
        print("  All tracking model parameters will be trained")
    
    # Get tracking training files
    baseline_files = process_file_args(args.bl, predictor.dataset_json['file_ending'])
    followup_files = process_file_args(args.fu, predictor.dataset_json['file_ending'])
    baseline_prompt_files = process_file_args(args.pbl, predictor.dataset_json['file_ending'])
    followup_prompt_files = process_file_args(args.pfu, predictor.dataset_json['file_ending'])
    
    # Match baseline and follow-up files based on patient ID (extract from filename)
    def extract_patient_id(filename):
        """Extract patient ID from filename (assumes pattern like PatientID_TP0_xxx.nii.gz or PatientID_TP1_xxx.nii.gz)"""
        basename = os.path.basename(filename)
        # Remove file extension and split by underscore
        parts = basename.replace(predictor.dataset_json['file_ending'], '').split('_')
        # Return the first part as patient ID (before _TP0 or _TP1)
        return parts[0] if parts else basename
    
    # Create dictionaries to match files by patient ID
    baseline_dict = {extract_patient_id(f): f for f in baseline_files}
    followup_dict = {extract_patient_id(f): f for f in followup_files}
    baseline_prompt_dict = {extract_patient_id(f): f for f in baseline_prompt_files}
    followup_prompt_dict = {extract_patient_id(f): f for f in followup_prompt_files}
    
    # Find common patient IDs that have all required files
    common_patients = set(baseline_dict.keys()) & set(followup_dict.keys()) & set(baseline_prompt_dict.keys()) & set(followup_prompt_dict.keys())
    
    print(f"Total baseline files: {len(baseline_files)}")
    print(f"Total follow-up files: {len(followup_files)}")
    print(f"Total baseline prompt files: {len(baseline_prompt_files)}")
    print(f"Total follow-up prompt files: {len(followup_prompt_files)}")
    print(f"Patients with complete data (baseline + follow-up + both prompts): {len(common_patients)}")
    
    if len(common_patients) == 0:
        raise ValueError("No patients found with complete baseline and follow-up data. Check that your file naming convention allows matching.")
    
    # Create matched file lists
    matched_baseline_files = [baseline_dict[pid] for pid in sorted(common_patients)]
    matched_followup_files = [followup_dict[pid] for pid in sorted(common_patients)]
    matched_baseline_prompt_files = [baseline_prompt_dict[pid] for pid in sorted(common_patients)]
    matched_followup_prompt_files = [followup_prompt_dict[pid] for pid in sorted(common_patients)]
    
    # Update variables to use matched files
    baseline_files = matched_baseline_files
    followup_files = matched_followup_files
    baseline_prompt_files = matched_baseline_prompt_files
    followup_prompt_files = matched_followup_prompt_files
        
    # Create output file names for training
    train_output_files = [join(args.o, 'train_' + os.path.basename(i).replace(predictor.dataset_json['file_ending'], '')) for i in baseline_files]
    
    # Get TEST files for evaluation during training
    test_baseline_files = None
    test_followup_files = None
    test_baseline_prompt_files = None
    test_followup_prompt_files = None
    test_dataset = None
    
    if args.tbl and args.tfu and args.tpbl and args.tpfu:
        # Test files
        test_baseline_files = process_file_args(args.tbl, predictor.dataset_json['file_ending'])
        test_followup_files = process_file_args(args.tfu, predictor.dataset_json['file_ending'])
        test_baseline_prompt_files = process_file_args(args.tpbl, predictor.dataset_json['file_ending'])
        test_followup_prompt_files = process_file_args(args.tpfu, predictor.dataset_json['file_ending'])
        
        # Match test files by patient ID
        test_baseline_dict = {extract_patient_id(f): f for f in test_baseline_files}
        test_followup_dict = {extract_patient_id(f): f for f in test_followup_files}
        test_baseline_prompt_dict = {extract_patient_id(f): f for f in test_baseline_prompt_files}
        test_followup_prompt_dict = {extract_patient_id(f): f for f in test_followup_prompt_files}
        
        # Find common patient IDs for test data
        test_common_patients = set(test_baseline_dict.keys()) & set(test_followup_dict.keys()) & set(test_baseline_prompt_dict.keys()) & set(test_followup_prompt_dict.keys())
        
        print(f"Test patients with complete data: {len(test_common_patients)}")
        
        if len(test_common_patients) > 0:
            # Create matched test file lists
            test_baseline_files = [test_baseline_dict[pid] for pid in sorted(test_common_patients)]
            test_followup_files = [test_followup_dict[pid] for pid in sorted(test_common_patients)]
            test_baseline_prompt_files = [test_baseline_prompt_dict[pid] for pid in sorted(test_common_patients)]
            test_followup_prompt_files = [test_followup_prompt_dict[pid] for pid in sorted(test_common_patients)]
        else:
            print("Warning: No test patients with complete data found. Skipping test dataset.")
            test_baseline_files = test_followup_files = test_baseline_prompt_files = test_followup_prompt_files = []
        
        # Create combined test files for evaluation
        if len(test_common_patients) > 0:
            test_input_files = test_baseline_files + test_followup_files
            test_prompt_files = test_baseline_prompt_files + test_followup_prompt_files
            test_output_files = [join(args.o, 'test_' + os.path.basename(i).replace(predictor.dataset_json['file_ending'], '')) for i in test_input_files]
            
            # Create test dataset for dice computation and visualization
            test_dataset = predictor.create_training_dataset(
                input_files=test_input_files,
                prompt_files=test_prompt_files,
                output_files=test_output_files,
                prompt_type=args.t,
                num_processes=args.npp,
                verbose=args.verbose,
                track=args.track
            )
            print(f"Test dataset created with {len(test_input_files)} samples")
        else:
            test_dataset = None
            print("No test dataset created (no complete test data available)")
    
    # Create combined training files from baseline and follow-up data
    train_input_files = baseline_files + followup_files
    train_prompt_files = baseline_prompt_files + followup_prompt_files
    
    print(f"Training dataset created with {len(train_input_files)} samples")
    
    # Start cross-validation training
    all_fold_results = predictor.train_cross_validation(
        train_input_files=train_input_files,
        train_prompt_files=train_prompt_files,
        train_output_files=train_output_files,
        test_dataset=test_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        output_folder=args.o,
        n_folds=5,
        num_workers=args.num_workers,
        prompt_type=args.t,
        ckpt_path=args.ckpt_path,
        finetune_mode=args.finetune,
        train_fold=args.train_fold,
    )
    
    print("Cross-validation training completed!")
    
    # Print summary statistics from all folds
    if all_fold_results:
        # Get final metrics from each fold
        final_train_losses = [fold['train_losses'][-1] for fold in all_fold_results]
        final_val_losses = [fold['val_losses'][-1] for fold in all_fold_results]
        best_val_losses = [fold['best_val_loss'] for fold in all_fold_results]
        
        print(f"Mean final training loss across folds: {np.mean(final_train_losses):.4f} ± {np.std(final_train_losses):.4f}")
        print(f"Mean final validation loss across folds: {np.mean(final_val_losses):.4f} ± {np.std(final_val_losses):.4f}")
        print(f"Mean best validation loss across folds: {np.mean(best_val_losses):.4f} ± {np.std(best_val_losses):.4f}")
        
        if all_fold_results[0]['test_dice_scores']:
            final_test_dice = [fold['test_dice_scores'][-1] for fold in all_fold_results]
            print(f"Mean final test dice across folds: {np.mean(final_test_dice):.4f} ± {np.std(final_test_dice):.4f}")


# if __name__ == "__main__":
#     import sys
    
#     # Check command line arguments to determine which mode to use
#     if len(sys.argv) > 1 and ('-ib' in sys.argv or '--baseline_images' in sys.argv):
#         # Tracking training mode
#         train_tracking_from_data()
#     else:
#         # Standard segmentation training mode
#         train_from_prompt()
