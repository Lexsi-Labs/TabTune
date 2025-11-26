import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from functools import partial
import numpy as np
import pandas as pd
import logging
import os

def ensure_device_consistency(model, device):
    """Ensure all model parameters and buffers are on the same device"""
    model.to(device)
    for param in model.parameters():
        param.data = param.data.to(device)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(device)
    return model


from ..models.tabpfn.classifier import TabPFNClassifier
from ..models.tabpfn.utils import meta_dataset_collator
from ..models.tabicl.sklearn.classifier import TabICLClassifier
from ..models.tabicl.sklearn.preprocessing import TabICLMetaDataset
from ..models.orion_msp.sklearn.classifier import OrionMSPClassifier
from ..models.orion_msp.sklearn.preprocessing import OrionMSPMetaDataset
from ..models.contexttab.contexttab import ConTextTabClassifier
from ..models.mitra.tab2d import Tab2D
from torch.utils.data import TensorDataset
from ..models.orion_bix.sklearn.classifier import OrionBixClassifier
from ..models.tabdpt.classifier import TabDPTClassifier
from ..models.tabdpt.utils import pad_x
from ..models.tabdpt.model import TabDPTModel

from .peft_utils import apply_tabular_lora

logger = logging.getLogger(__name__)




class TuningManager:
    """
    Handles the model adaptation process
    """
    def tune(self, model, X_train, y_train, strategy='inference', params=None, processor=None):
        
        params_copy = dict(params) if isinstance(params, dict) else {}
        finetune_mode = params_copy.pop('finetune_mode', 'meta-learning')
        save_checkpoint_path = params_copy.pop('save_checkpoint_path', None)
        if save_checkpoint_path is None:
            default_dir = params_copy.get("checkpoint_dir", "./checkpoints")
            if not os.path.exists(default_dir):
                os.makedirs(default_dir)
            save_checkpoint_path = os.path.join(default_dir, f"{type(model).__name__}_latest.pt")

        # Strategy selection: accept either explicit 'peft' strategy or finetune_method='peft'
        finetune_method = params_copy.pop('finetune_method', None)
        peft_config = params_copy.pop('peft_config', None)
        selected_strategy = strategy
        if strategy == 'finetune' and finetune_method == 'peft':
            selected_strategy = 'peft'
        elif strategy == 'finetune':
            selected_strategy = 'base-ft'

        is_finetuned = False
        original_is_tab2d = isinstance(model, Tab2D)


        if (isinstance(model, Tab2D) or original_is_tab2d) and selected_strategy in ('finetune', 'base-ft', 'peft'):
            if finetune_mode == 'sft':
                logger.info("[TuningManager] Using Pure SFT for Mitra (task-optimized)")
                self._finetune_mitra_pure_sft(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            else:  # default: 'meta-learning'
                logger.info("[TuningManager] Using Episodic Meta-Learning for Mitra (default)")
                self._finetune_mitra(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            is_finetuned = True
        
        elif isinstance(model, TabPFNClassifier) and selected_strategy in ('finetune', 'base-ft', 'peft'):
            if finetune_mode == 'sft':
                logger.info("[TuningManager] Using Pure SFT for TabPFN (task-optimized)")
                self._finetune_tabpfn_pure_sft(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            else:  # default: 'meta-learning'
                logger.info("[TuningManager] Using Episodic Meta-Learning for TabPFN (default)")
                self._finetune_tabpfn(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            is_finetuned = True
        
        elif isinstance(model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier)) and selected_strategy in ('finetune', 'base-ft', 'peft'):
            if finetune_mode == 'meta-learning':
                logger.info("[TuningManager] Meta Learning based FT")
                self._finetune_tabicl(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            else:
                logger.info("[TuningManager] Performing SFT")
                self._finetune_tabicl_simple_sft(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            is_finetuned = True
        
        elif isinstance(model, ConTextTabClassifier) and selected_strategy in ('finetune', 'base-ft', 'peft'):
            self._full_finetune_model(model, X_train, y_train, params=params_copy, processor=processor, peft_config=peft_config)
            is_finetuned = True
        
        elif isinstance(model, TabDPTClassifier) and selected_strategy in ('finetune', 'base-ft', 'peft'):
            if finetune_mode == 'sft':
                logger.info("[TuningManager] Using Pure SFT for TabDPT (task-optimized)")
                self._finetune_tabdpt_pure_sft(model, X_train, y_train, params=params_copy, processor=processor, peft_config=peft_config)
            else:  # default: 'meta-learning'
                logger.info("[TuningManager] Using Episodic Meta-Learning for TabDPT (default)")
                self._finetune_tabdpt(model, X_train, y_train, params=params_copy, processor=processor, peft_config=peft_config)
            is_finetuned = True

        
        elif isinstance(model, (Tab2D)) and selected_strategy == 'inference':
            logger.info("[TuningManager] In-context learning model in inference mode. No training needed.")
            pass
        elif isinstance(model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier)) and selected_strategy == 'inference':
            logger.info("[TuningManager] Applying standard .fit() for TabICL setup (inference mode)")
            model.fit(X_train, y_train)
        else:
            logger.info("[TuningManager] Applying standard model fitting (.fit)")
            model.fit(X_train, y_train)


        if is_finetuned and save_checkpoint_path:
            self._save_checkpoint(model, save_checkpoint_path)
            logger.info(f"[TuningManager] Saved fine-tuned checkpoint to {save_checkpoint_path}")
            
            model = self.load_checkpoint(model, save_checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
            logger.info("[TuningManager] Reloaded fine-tuned weights into model for inference")
            

            if isinstance(model, torch.nn.Module):
                model.eval()
            elif hasattr(model, 'model'):
                model.model.eval()
            elif hasattr(model, 'model_'):
                model.model_.eval()
            
        
            logger.info("[TuningManager] Reloaded fine-tuned weights and set model to eval mode")
            

        return model
        


    def _maybe_save_epoch_ckpt(self, model, ckpt_dir, ckpt_epochs, epoch, prefix):
        if ckpt_dir and (epoch in ckpt_epochs):
            fname = f"{prefix}_epoch{epoch}.pt"
            path = os.path.join(ckpt_dir, fname)
            self._save_checkpoint(model, path)
            
    def _save_checkpoint(self, model, path: str):
        logger.info(f"[TuningManager] Saving model checkpoint to {path}")

        torch_model = None
        if hasattr(model, 'model_'):  # For TabPFN, TabICL, OrionMSP, OrionBix
            torch_model = model.model_
        elif hasattr(model, 'model'):  # For ContextTab, TabDPT
            torch_model = model.model
        elif isinstance(model, torch.nn.Module):  # For Mitra
            torch_model = model

        if torch_model:
            try:
            # Ensure path is a string here!
                if not isinstance(path, str):
                    raise ValueError("Checkpoint path must be a string")
                torch.save(torch_model.state_dict(), path)
                logger.info(f"[TuningManager] Checkpoint saved successfully to {path}")
            except Exception as e:
                logger.error(f"[TuningManager] Failed to save checkpoint: {e}")
        else:
            logger.warning(f"[TuningManager] No compatible torch model found to save checkpoint")



    def load_checkpoint(self, model, ckpt_path: str, map_location='cpu'):
        """Loads a checkpoint automatically to correct submodule."""
        if not os.path.exists(ckpt_path):
            logger.warning(f"[TuningManager] Checkpoint path {ckpt_path} not found")
            return model

        state = torch.load(ckpt_path, map_location=map_location)
        state_dict = state.get('model_state_dict', state)
        candidates = [getattr(model, 'model_', None), getattr(model, 'model', None), model]

        for candidate in candidates:
            if isinstance(candidate, torch.nn.Module):
                try:
                    candidate.load_state_dict(state_dict, strict=False)
                    logger.info(f"[TuningManager] Loaded checkpoint weights into {type(candidate).__name__}")
                    return model
                except Exception as e:
                    logger.warning(f"[TuningManager] Could not load into {type(candidate).__name__}: {e}")
        logger.error("[TuningManager] Failed to load weights into model")
        return model
        
            
    def _full_finetune_model(self, model, X_train, y_train, params=None, processor=None, peft_config=None):
        """
        Performs a standard full fine-tuning loop. This has been refactored to
        use the model's own tokenizer for batch preparation, ensuring correctness.
        """
        logger.info(f"[TuningManager] Starting full fine-tuning for {type(model).__name__}")
        
        config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "epochs": 5,
            "learning_rate": 1e-4,
            "batch_size": 128,
            "show_progress": True
        }
        if params:
            config.update(params)
        logger.debug(f"[TuningManager] Using fine-tuning config: {config}")
            
        is_contexttab = isinstance(model, ConTextTabClassifier)
        torch_model = model.model
        
        device = torch.device(config["device"])
        torch_model.to(device)
        torch_model.train()

        for param in torch_model.parameters():
            param.data = param.data.to(device)

        if is_contexttab:
            logger.info("[TuningManager] Fitting the ConTextTab wrapper to set its data context")
            model.fit(X_train, y_train)

        if peft_config:
            logger.warning("[TuningManager] WARNING: ConTextTab PEFT support is currently experimental and may cause prediction issues")
            logger.warning("[TuningManager] ConTextTab's complex embedding pipeline may conflict with LoRA adapters")
            logger.info("[TuningManager] RECOMMENDATION: Use 'base-ft' strategy for ConTextTab instead of 'peft'")
            logger.info("[TuningManager] FALLBACK: Proceeding with standard base fine-tuning")
            peft_config = None  # Disable PEFT for ConTextTab
        
        optimizer = Adam(torch_model.parameters(), lr=config["learning_rate"])
        loss_fn = torch.nn.CrossEntropyLoss()

        # Create a simple dataset of indices
        dataset = TensorDataset(torch.arange(len(X_train)))
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        for epoch in range(1, config["epochs"] + 1):
            iterable = dataloader
            if config["show_progress"]:
                iterable = tqdm(dataloader, desc=f"Finetuning Epoch {epoch}")
            
            for batch_indices in iterable:
                # Get the raw data for the current batch
                if hasattr(X_train, 'iloc'):  # DataFrame
                    X_batch_raw = X_train.iloc[batch_indices[0].numpy()]
                    y_batch_raw = y_train.iloc[batch_indices[0].numpy()]
                else:  # numpy array
                    X_batch_raw = X_train[batch_indices[0].numpy()]
                    y_batch_raw = y_train[batch_indices[0].numpy()]

                optimizer.zero_grad()
                
                if is_contexttab:
                    # Use the model's own tokenizer to prepare the batch
                    # This guarantees the correct format.
                    data_batch = model.get_tokenized_data(X_batch_raw, bagging_index=epoch)
                    
                    # Move tensors to the correct device
                    for k, v in data_batch.items():
                        if isinstance(v, torch.Tensor):
                            data_batch[k] = v.to(device)
                        elif isinstance(v, dict): # Handle nested dicts like ⁠ data['data'] ⁠
                             for k_inner, v_inner in v.items():
                                 if isinstance(v_inner, torch.Tensor):
                                     v[k_inner] = v_inner.to(device)
                    
                    y_batch = data_batch['data']['target']
                    # Ensure y_batch is Long type for cross-entropy loss (ContextTab may return Float)
                    if y_batch.dtype != torch.long:
                        y_batch = y_batch.long()
                    logits = torch_model(**data_batch)

                else: # Fallback for other potential models
                    X_batch_processed, y_batch_processed = processor.transform(X_batch_raw, y_batch_raw)
                    X_batch = torch.tensor(X_batch_processed, dtype=torch.float32).to(device)
                    y_batch = torch.tensor(y_batch_processed, dtype=torch.long).to(device)
                    logits = torch_model(X_batch)

                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                if config["show_progress"]:
                    iterable.set_postfix(loss=f"{loss.item():.4f}")
        
        logger.info("[TuningManager] Full fine-tuning complete")

    def _finetune_tabpfn(self, model: TabPFNClassifier, X_train_processed: pd.DataFrame, y_train_processed: pd.Series, params: dict | None = None, peft_config=None):
        logger.info("[TuningManager] Starting advanced TabPFN fine-tuning")
        
        config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "epochs": 3, "learning_rate": 1e-5, "batch_size": 256, "show_progress": True 
        }
        if params:
            config.update(params)
        logger.debug(f"[TuningManager] Using fine-tuning config: {config}")

        device = torch.device(config["device"])
        model.model_.to(device)

        for param in model.model_.parameters():
            param.data = param.data.to(device)

        if peft_config:
            logger.warning("[TuningManager] WARNING: TabPFN PEFT support is currently experimental and unstable")
            logger.warning("[TuningManager] TabPFN's batched inference engine conflicts with LoRA adapter state")
            logger.info("[TuningManager] RECOMMENDATION: Use 'base-ft' strategy for TabPFN instead of 'peft'")
            logger.info("[TuningManager] FALLBACK: Proceeding with standard base fine-tuning")
            peft_config = None  # Disable PEFT for TabPFN

        optimizer = Adam(model.model_.parameters(), lr=config["learning_rate"])
        loss_function = torch.nn.CrossEntropyLoss()

        def stratified_splitter(X, y):
            """
            A robust splitter that attempts to stratify and falls back gracefully.
            """
            # Check if the target is multiclass and has at least 2 samples per class
            y_series = pd.Series(y)
            if y_series.nunique() > 1 and y_series.value_counts().min() > 1:
                # If stratification is possible, use it.
                return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
            else:
                # Otherwise, use a standard random split.
                return train_test_split(X, y, test_size=0.3, random_state=42)

        # Use our new, robust splitter function directly.
        splitter = stratified_splitter

        #splitter = partial(train_test_split, test_size=0.3, stratify=None)
        training_datasets = model.get_preprocessed_datasets(
            X_train_processed, y_train_processed, splitter, config["batch_size"]
        )
        finetuning_dataloader = DataLoader(
            training_datasets, batch_size=1, collate_fn=meta_dataset_collator
        )

        for epoch in range(1, config["epochs"] + 1):
            iterable = finetuning_dataloader
            if config["show_progress"]:
                iterable = tqdm(finetuning_dataloader, desc=f"Finetuning Epoch {epoch}")

            def _move_to_device(item, target_device: torch.device):
                if isinstance(item, torch.Tensor):
                    return item.to(target_device)
                if isinstance(item, list):
                    return [_move_to_device(x, target_device) for x in item]
                if isinstance(item, tuple):
                    return tuple(_move_to_device(x, target_device) for x in item)
                if isinstance(item, dict):
                    return {k: _move_to_device(v, target_device) for k, v in item.items()}
                return item
            
            for (X_train_batch, X_test_batch, y_train_batch, y_test_batch, cat_ixs, confs) in iterable:
                if len(np.unique(y_train_batch)) != len(np.unique(y_test_batch)):
                    logger.debug("[TuningManager] Skipping batch with inconsistent number of classes between train and test splits")
                    continue

                X_train_batch = _move_to_device(X_train_batch, device)
                y_train_batch = _move_to_device(y_train_batch, device)
                X_test_batch = _move_to_device(X_test_batch, device)
                y_test_batch = _move_to_device(y_test_batch, device)


                optimizer.zero_grad()
                model.fit_from_preprocessed(X_train_batch, y_train_batch, cat_ixs, confs)
                predictions = model.forward(X_test_batch, return_logits=True)
                if isinstance(predictions, torch.Tensor) and predictions.device != device:
                    predictions = predictions.to(device)
                # y_test_batch has already been moved above; in rare cases where it is a list
                # choose the first element (batch_size == 1 in our collator)
                if isinstance(y_test_batch, list) and len(y_test_batch) > 0 and isinstance(y_test_batch[0], torch.Tensor):
                    target = y_test_batch[0]
                else:
                    target = y_test_batch
                loss = loss_function(predictions, target)
                loss.backward()
                optimizer.step()
                if config["show_progress"]:
                    iterable.set_postfix(loss=f"{loss.item():.4f}")

        model.batched = False
        logger.info("[TuningManager] Fine-tuning complete")
        logger.debug("[TuningManager] Setting fine-tuned model context for inference...")
        #model.fit(X_train_processed, y_train_processed)




    def _finetune_tabpfn_pure_sft(self, model: TabPFNClassifier, X_train_processed: pd.DataFrame, y_train_processed: pd.Series, params: dict | None = None, peft_config=None):
        """
        Performs SFT-style finetuning.
        
        This is different from the meta-learning loop by:
        1. Using the *entire* dataset to create ONE single, large (Support, Query) episode.
        2. Training repeatedly over this single episode for multiple epochs.
        
        This forces the model to specialize on the single task derived from the 
        full dataset, giving the "SFT sense".
        """
        import torch
        import numpy as np
        import pandas as pd
        from torch.optim import Adam
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # This collator is required by the TabPFN API
        try:
            from ..models.tabpfn.utils import meta_dataset_collator
        except ImportError:
            logger.error("[TuningManager] FATAL: meta_dataset_collator not found. Please fix the import path")
            # Define a minimal fallback if import fails
            def meta_dataset_collator(batch): return batch[0]
            logger.warning("[TuningManager] Using a placeholder meta_dataset_collator. This may fail")
            
        # Helper to move tensors
        def _move_to_device(item, target_device: torch.device):
            if isinstance(item, torch.Tensor):
                return item.to(target_device)
            if isinstance(item, list):
                return [_move_to_device(x, target_device) for x in item]
            if isinstance(item, tuple):
                return tuple(_move_to_device(x, target_device) for x in item)
            if isinstance(item, dict):
                return {k: _move_to_device(v, target_device) for k, v in item.items()}
            return item

        
        logger.info("[TuningManager] Starting TabPFN SFT fine-tuning")

        config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "epochs": 25,  # More epochs needed as we only have one "batch"
            "learning_rate": 1e-5,
            "show_progress": True,
            "max_episode_size": len(X_train_processed),
            "query_set_ratio": 0.3,
            "weight_decay": 1e-4
        }
        if params:
            # Allow user to override SFT defaults
            config.update(params)
            # Ensure max_episode_size isn't accidentally overridden by 'batch_size'
            if 'batch_size' in params:
                logger.warning("[TuningManager] Ignoring 'batch_size' param, using 'max_episode_size' for SFT")
                config.pop('batch_size', None)
            
        logger.debug(f"[TuningManager] Using SFT-style config: {config}")

        device = torch.device(config["device"])
        model.model_.to(device)
        model.model_.train() # Set to train mode

        for param in model.model_.parameters():
            param.data = param.data.to(device)

        if peft_config:
            logger.warning("[TuningManager] TabPFN PEFT not supported, falling back to base fine-tuning")
            peft_config = None

        optimizer = Adam(model.model_.parameters(), 
                         lr=config["learning_rate"], 
                         weight_decay=config["weight_decay"])
        loss_function = torch.nn.CrossEntropyLoss()
        
        # --- Data & Label Preprocessing ---
        # (This section is the same as the meta-learning function)
        if isinstance(X_train_processed, pd.DataFrame):
            X_train_processed_np = X_train_processed.to_numpy()
        else:
            X_train_processed_np = X_train_processed
            
        if isinstance(y_train_processed, (pd.Series, pd.DataFrame)):
            y_train_processed_np = y_train_processed.to_numpy()
        else:
            y_train_processed_np = y_train_processed

        if y_train_processed_np.dtype == object or not np.issubdtype(y_train_processed_np.dtype, np.number):
            logger.info("[TuningManager] Converting non-numeric labels...")
            le = LabelEncoder()
            y_train_processed_np = le.fit_transform(y_train_processed_np)
            if not hasattr(model, 'label_encoder_'):
                 model.label_encoder_ = le

        def sft_episode_splitter(X, y):
            y_series = pd.Series(y)
            test_size = config["query_set_ratio"]
            if y_series.nunique() > 1 and y_series.value_counts().min() > 1:
                return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
            else:
                return train_test_split(X, y, test_size=test_size, random_state=42)

        logger.info(f"[TuningManager] Creating a single SFT task from {len(X_train_processed_np)} samples...")
        training_datasets = model.get_preprocessed_datasets(
            X_train_processed_np, 
            y_train_processed_np, 
            sft_episode_splitter, 
            config["max_episode_size"] # <-- This makes it ONE episode
        )

        episode_dataloader = DataLoader(
            training_datasets, 
            batch_size=1, 
            collate_fn=meta_dataset_collator,
            shuffle=False
        )

        for epoch in range(1, config["epochs"] + 1):
            
            iterable = tqdm(episode_dataloader, desc=f"SFT Epoch {epoch}", leave=False)
            epoch_losses = []
            
            for (X_support, X_query, y_support, y_query, cat_ixs, confs) in iterable:
                if len(np.unique(y_support)) != len(np.unique(y_query)):
                    logger.warning("[TuningManager] Skipping epoch: Inconsistent classes in SFT split")
                    continue

                X_support = _move_to_device(X_support, device)
                y_support = _move_to_device(y_support, device)
                X_query = _move_to_device(X_query, device)
                y_query = _move_to_device(y_query, device)

                optimizer.zero_grad()
                
                # 1. Set the (large) Support Set as the prompt
                model.fit_from_preprocessed(X_support, y_support, cat_ixs, confs)
                
                # 2. Predict on the (large) Query Set
                predictions = model.forward(X_query, return_logits=True)
                
                if isinstance(predictions, torch.Tensor) and predictions.device != device:
                    predictions = predictions.to(device)
                    
                target = y_query[0] if isinstance(y_query, list) else y_query
                
                # 3. Calculate loss and backpropagate
                loss = loss_function(predictions, target)
                loss.backward()
                
                # SFT HINT 4: Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.model_.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_losses.append(loss.item())
                iterable.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
            logger.info(f"[TuningManager] Epoch [{epoch}/{config['epochs']}]: Task Loss = {avg_loss:.4f}")

        model.batched = False
        model.model_.eval()
        logger.info("[TuningManager] SFT-style finetuning complete")
        return model

    

    def _finetune_tabicl(self, model: (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier), X_train_processed: np.ndarray, y_train_processed: np.ndarray, params: dict | None = None, peft_config=None):
        logger.info("[TuningManager] Starting advanced TabICL/OrionMSP/OrionBix fine-tuning")
        
        config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "epochs": 5, "learning_rate": 2e-6, "show_progress": True,
            "support_size": 48, "query_size": 32, "n_episodes": 1000
        }
        if params:
            config.update(params)
            
        logger.debug(f"[TuningManager] Using fine-tuning config: {config}")
        
        model.fit(X_train_processed, y_train_processed)
        model._load_model()
        model.fit(X_train_processed, y_train_processed) 

        device = torch.device(config["device"])
        if peft_config:
            try:
                if isinstance(model, OrionBixClassifier): 
                    model_key = "OrionBix" 
                elif isinstance(model, OrionMSPClassifier):
                    model_key = "OrionMSP"
                else :
                    model_key = "TabICL"
                model.model_ = apply_tabular_lora(model_key, model.model_, peft_config)
                logger.info(f"[TuningManager] PEFT SUCCESS: Applied LoRA adapters to {model_key} model")
            except Exception as e:
                logger.warning(f"[TuningManager] PEFT FAILED: TabICL/OrionMSP/OrionBix incompatible with PEFT: {e}")
                logger.info("[TuningManager] FALLBACK: Proceeding with base fine-tuning (fully supported)")
                
        
        model.model_.to(device)
        model.model_.train()
        
        # --- discover the true logits width from a safe 1-class probe ---
        C_out = None
        with torch.no_grad():
            # make one tiny 1-class episode from the first few rows
            X_np = X_train_processed if isinstance(X_train_processed, np.ndarray) else X_train_processed.to_numpy()
            y_np = y_train_processed if isinstance(y_train_processed, np.ndarray) else y_train_processed.to_numpy()

            # pick a class that has >= (support_size + query_size) examples; fall back to any class
            s_sz = int(config.get("support_size", 48))
            q_sz = int(config.get("query_size", 32))
            need = s_sz + q_sz

            cls, idx = None, None
            for c in np.unique(y_np):
                cand = np.nonzero(y_np == c)[0]
                if cand.size >= need:
                    idx = cand[:need]
                    cls = c
                    break
            if idx is None:
                idx = np.arange(min(need, len(y_np)))
                cls = y_np[idx[0]]

            X_ep = torch.from_numpy(X_np[idx]).float().unsqueeze(0).to(device)   # [1, S+Q, F]
            ys   = torch.full((s_sz,), 0, dtype=torch.long, device=device)       # all support -> class 0
            # pack as your forward expects: first S as support, rest as query
            logits_probe = model.model_(X_ep, ys.unsqueeze(0))                   # [1, Q, C_eff] typically
            C_out = int(logits_probe.squeeze(0).size(-1))

        # safety
        if C_out <= 0:
            raise RuntimeError("Could not infer logits width (C_out).")



        for param in model.model_.parameters():
            param.data = param.data.to(device)
        
        optimizer = Adam(model.model_.parameters(), lr=config["learning_rate"])
        loss_fn = torch.nn.CrossEntropyLoss()

        meta_dataset = TabICLMetaDataset(
            X_train_processed, y_train_processed,
            support_size=int(config.get("support_size", 48)),
            query_size=int(config.get("query_size", 32)),
            n_episodes=int(config.get("n_episodes", 1000))
        )
        
        dataloader = DataLoader(meta_dataset, batch_size=1, shuffle=True)
        
        for epoch in range(1, config["epochs"] + 1):
            iterable = dataloader
            if config["show_progress"]:
                iterable = tqdm(dataloader, desc=f"Finetuning Epoch {epoch}")
            for X_episode, y_support, y_query in iterable:
                X_episode, y_support, y_query = X_episode.to(device), y_support.to(device), y_query.to(device)
                optimizer.zero_grad()

                ys = y_support.squeeze(0).long()
                yq = y_query.squeeze(0).long()

                supp = torch.unique(ys)
                # keep at most C_out classes so the head can represent them
                keep = supp[:C_out]

                # build map only for kept classes; others -> -1 (excluded)
                yq_m = torch.full_like(yq, -1)
                ys_m = torch.full_like(ys, -1)
                for i, c in enumerate(keep):
                    ys_m[ys == c] = i
                    yq_m[yq == c] = i

                # prune support rows that were dropped
                keep_mask = (ys_m >= 0)
                if not keep_mask.any():
                    continue
                ys_m = ys_m[keep_mask]
                X_support_kept = X_episode[:, :ys.shape[0], :][:, keep_mask, :]
                X_query_part   = X_episode[:, ys.shape[0]:, :]
                X_episode = torch.cat([X_support_kept, X_query_part], dim=1)

                # if any query label was excluded, skip this episode (avoids OOB gathers)
                if (yq_m < 0).any():
                    continue

                # forward with episodic labels (contiguous, ≤ C_out)
                logits = model.model_(X_episode, ys_m.unsqueeze(0))  # [1, Q, <=C_out]
                logits = logits.squeeze(0) # [Q, <=C_out]
                 # ensure mapping fits the actual head width (in case adapters changed it mid-run)
                if logits.size(-1) < yq_m.max().item() + 1:
                    continue  # skip this episode if it exceeds head capacity
                loss = loss_fn(logits, yq_m)


                
                loss.backward()
                optimizer.step()
                if config["show_progress"]:
                    iterable.set_postfix(loss=f"{loss.item():.4f}")
        logger.info("[TuningManager] Fine-tuning complete")


    def _finetune_tabicl_pure_sft(self, model: (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier) , X_train_processed, y_train_processed, params=None, peft_config=None):
        """
        PURE SFT FINE-TUNING (Not Recommended for TabICL)
    
        Standard supervised fine-tuning on full batches WITHOUT episodic structure.
    
        WARNING: This ignores TabICL's meta-learning design and may:
        - Reduce generalization to new tasks
        - Increase catastrophic forgetting
        - Overfit to the specific target task
    
        Use ONLY for:
        - Benchmarking against traditional fine-tuning
        - Comparison studies
        - Tasks where you explicitly want to sacrifice generalization for accuracy
        """
        logger.warning("[TuningManager] WARNING: Pure SFT on TabICL breaks its meta-learning design")
        logger.warning("[TuningManager] This approach may reduce generalization to new tasks")
        logger.info("[TuningManager] RECOMMENDATION: Use episodic or SFT-hybrid instead")
        logger.info("[TuningManager] PROCEED: Using pure SFT (use only for comparisons)")
    
        config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "epochs": 10,
            "learning_rate": 1e-5,
            "batch_size": 32,
            "show_progress": True,
            "weight_decay": 1e-4,
            "warmup_epochs": 1
        }
        if params:
            config.update(params)
        logger.debug(f"[TuningManager] Using config: {config}")
    
        device = torch.device(config["device"])
        model.fit(X_train_processed, y_train_processed)
        model._load_model()
    
        model.model_.to(device)
        model.model_.train()
        
        C_out = None
        with torch.no_grad():
            # make one tiny 1-class episode from the first few rows
            X_np = X_train_processed if isinstance(X_train_processed, np.ndarray) else X_train_processed.to_numpy()
            y_np = y_train_processed if isinstance(y_train_processed, np.ndarray) else y_train_processed.to_numpy()

            # pick a class that has >= (support_size + query_size) examples; fall back to any class
            s_sz = int(config.get("support_size", 48))
            q_sz = int(config.get("query_size", 32))
            need = s_sz + q_sz

            cls, idx = None, None
            for c in np.unique(y_np):
                cand = np.nonzero(y_np == c)[0]
                if cand.size >= need:
                    idx = cand[:need]
                    cls = c
                    break
            if idx is None:
                idx = np.arange(min(need, len(y_np)))
                cls = y_np[idx[0]]

            X_ep = torch.from_numpy(X_np[idx]).float().unsqueeze(0).to(device)   # [1, S+Q, F]
            ys   = torch.full((s_sz,), 0, dtype=torch.long, device=device)       # all support -> class 0
            # pack as your forward expects: first S as support, rest as query
            logits_probe = model.model_(X_ep, ys.unsqueeze(0))                   # [1, Q, C_eff] typically
            C_out = int(logits_probe.squeeze(0).size(-1))

        # safety
        if C_out <= 0:
            raise RuntimeError("Could not infer logits width (C_out).")
    
        for param in model.model_.parameters():
            param.data = param.data.to(device)
    
        if peft_config:
            try:
                model.model_ = apply_tabular_lora("TabICL", model.model_, peft_config)
                logger.info("[TuningManager] Applied LoRA adapters to TabICL (pure SFT)")
            except Exception as e:
                logger.warning(f"[TuningManager] LoRA failed: {e}. Proceeding with base pure SFT fine-tuning")
    
    # Create standard supervised dataset
        dataset = TensorDataset(
            torch.from_numpy(X_train_processed).float(),
            torch.from_numpy(y_train_processed).long()
        )
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
        optimizer = torch.optim.Adam(model.model_.parameters(),
                                lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])
        loss_fn = torch.nn.CrossEntropyLoss()
    
        # Optional: Learning rate scheduler
        total_steps = len(dataloader) * config["epochs"]
        warmup_steps = len(dataloader) * config["warmup_epochs"]
    
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
        step = 0
        for epoch in range(1, config["epochs"] + 1):
            iterable = dataloader
            if config["show_progress"]:
                iterable = tqdm(dataloader, desc=f"Pure SFT Epoch {epoch}", leave=False)
        
            epoch_loss = 0
            for X_batch, y_batch in iterable:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                # split batch into a small pseudo-episode: half support, half query
                mid = X_batch.size(0) // 2
                X_support, y_support = X_batch[:mid], y_batch[:mid]
                X_query,   y_query   = X_batch[mid:], y_batch[mid:]
                X_episode = torch.cat([X_support, X_query], dim=0).unsqueeze(0)

                ys = y_support.squeeze(0).long()
                yq = y_query.squeeze(0).long()

                supp = torch.unique(ys)
                # keep at most C_out classes so the head can represent them
                keep = supp[:C_out]

                # build map only for kept classes; others -> -1 (excluded)
                yq_m = torch.full_like(yq, -1)
                ys_m = torch.full_like(ys, -1)
                for i, c in enumerate(keep):
                    ys_m[ys == c] = i
                    yq_m[yq == c] = i

                # prune support rows that were dropped
                keep_mask = (ys_m >= 0)
                if not keep_mask.any():
                    continue
                ys_m = ys_m[keep_mask]
                X_support_kept = X_episode[:, :ys.shape[0], :][:, keep_mask, :]
                X_query_part   = X_episode[:, ys.shape[0]:, :]
                X_episode = torch.cat([X_support_kept, X_query_part], dim=1)

                # if any query label was excluded, skip this episode (avoids OOB gathers)
                if (yq_m < 0).any():
                    continue

                # forward with episodic labels (contiguous, ≤ C_out)
                logits = model.model_(X_episode, ys_m.unsqueeze(0))  # [1, Q, <=C_out]
                logits = logits.squeeze(0)                           # [Q, <=C_out]
                 # ensure mapping fits the actual head width (in case adapters changed it mid-run)
                if logits.size(-1) < yq_m.max().item() + 1:
                    continue  # skip this episode if it exceeds head capacity
                loss = loss_fn(logits, yq_m)


                loss.backward()
                optimizer.step()
                scheduler.step()
            
                epoch_loss += loss.item()
                step += 1
            
                if config["show_progress"]:
                    iterable.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr():.2e}")
        
            logger.info(f"[TuningManager] Epoch {epoch}: Avg Loss = {epoch_loss/len(dataloader):.4f}, "
                   f"LR = {scheduler.get_last_lr():.2e}")
    
        logger.warning("[TuningManager] Pure SFT training complete (remember: not recommended for TabICL)")



    def _finetune_mitra(self, model, X_train_processed, y_train_processed, params=None, peft_config=None):
        """
        Performs episodic fine-tuning for in-context models like Mitra (Tab2D).
        """
        logger.info(f"[TuningManager] Starting episodic fine-tuning for {type(model).__name__}")
        
        config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "epochs": 3,
            "learning_rate": 1e-5,
            "batch_size": 4,
            "support_size": 128,
            "query_size": 128,
            "steps_per_epoch": 50,
            "show_progress": True
        }
        if params:
            config.update(params)
        logger.debug(f"[TuningManager] Using fine-tuning config: {config}")

        device = torch.device(config["device"])
        if peft_config:
            try:
                model = apply_tabular_lora("Mitra", model, peft_config)
                logger.info("[TuningManager] PEFT SUCCESS: Applied LoRA adapters to Mitra (Tab2D) model")
            except Exception as e:
                logger.warning(f"[TuningManager] PEFT FAILED: Mitra (Tab2D) incompatible with PEFT: {e}")
                logger.info("[TuningManager] FALLBACK: Proceeding with base fine-tuning (fully supported)")
                
        model.to(device)
        model.train()

        for param in model.parameters():
            param.data = param.data.to(device)
        
        optimizer = Adam(model.parameters(), lr=config["learning_rate"])
        loss_fn = torch.nn.CrossEntropyLoss()
        
        n_samples = X_train_processed.shape[0]
        episode_size = config['support_size'] + config['query_size']

        for epoch in range(1, config["epochs"] + 1):
            iterable = range(config['steps_per_epoch'])
            if config["show_progress"]:
                iterable = tqdm(iterable, desc=f"Finetuning Epoch {epoch}")

            for step in iterable:
                optimizer.zero_grad()
                
                X_episodes, y_episodes = [], []
                for _ in range(config['batch_size']):
                    # episode size does not exceed available samples
                    if episode_size > n_samples:
                        logger.warning(f"[TuningManager] Warning: Episode size ({episode_size}) is larger than the dataset size ({n_samples}). Using all samples")
                        indices = np.arange(n_samples)
                        np.random.shuffle(indices)
                    else:
                        indices = np.random.choice(n_samples, episode_size, replace=False)

                    X_episodes.append(X_train_processed[indices])
                    y_episodes.append(y_train_processed[indices])
                
                X_batch = torch.from_numpy(np.stack(X_episodes)).to(device)
                y_batch = torch.from_numpy(np.stack(y_episodes)).long().to(device)
                
                s_size = config['support_size']
                X_support, X_query = X_batch[:, :s_size, :], X_batch[:, s_size:, :]
                y_support, y_query = y_batch[:, :s_size], y_batch[:, s_size:]
                
                b, f = X_support.shape[0], X_support.shape[2]
                padding_features = torch.zeros(b, f, dtype=torch.bool, device=device)
                padding_obs_support = torch.zeros_like(y_support, dtype=torch.bool, device=device)
                padding_obs_query = torch.zeros(b, X_query.shape[1], dtype=torch.bool, device=device)

                logits = model(
                    x_support=X_support, y_support=y_support, x_query=X_query,
                    padding_features=padding_features, padding_obs_support=padding_obs_support,
                    padding_obs_query__=padding_obs_query
                )
                
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_query.reshape(-1))
                loss.backward()
                optimizer.step()
                
                if config["show_progress"]:
                    iterable.set_postfix(loss=f"{loss.item():.4f}")
        
        logger.info("[TuningManager] Episodic fine-tuning complete")


    def _finetune_tabdpt(self, model: TabDPTClassifier, X_train_processed: np.ndarray, y_train_processed: np.ndarray, params: dict | None = None, processor=None, peft_config=None):
        """
        Performs episodic fine-tuning for the TabDPT model.
        """
        logger.info(f"[TuningManager] Starting episodic fine-tuning for {type(model).__name__}")
        
        # Determine number of classes from training data
        num_classes = len(np.unique(y_train_processed))
        logger.info(f"[TuningManager] Detected {num_classes} classes in training data")
        
        config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "epochs": 5,
            "learning_rate": 1e-5,
            "batch_size": 8, 
            "support_size": 512,
            "query_size": 256,
            "steps_per_epoch": 100,
            "show_progress": True
        }
        if params:
            config.update(params)
        logger.debug(f"[TuningManager] Using fine-tuning config: {config}")

        device = torch.device(config["device"])

        if peft_config:
            try:
                model.model = apply_tabular_lora("TabDPT", model.model, peft_config)
                logger.info("[TuningManager] PEFT SUCCESS: Applied LoRA to TabDPT model")
            except Exception as e:
                logger.warning(f"[TuningManager] PEFT not compatible with TabDPT: {e}. Proceeding with base fine-tuning")
                
        model.model.to(device)
        model.model.train()

        for param in model.model.parameters():
            param.data = param.data.to(device)
        for buffer in model.model.buffers():
            buffer.data = buffer.data.to(device)
        
        # Also ensure the model's device attribute is updated
        model.device = str(device)
        
        # TabDPT now handles projection internally, so only use model parameters
        trainable_params = list(model.model.parameters())

        optimizer = torch.optim.Adam(trainable_params, lr=config["learning_rate"])
        loss_fn = torch.nn.CrossEntropyLoss()
        
        n_samples = X_train_processed.shape[0]
        #episode_size = config['support_size'] + config['query_size']
        
        # Compute PCA basis on GPU once, no autograd
        if getattr(model, "feature_reduction", "pca") == "pca" and X_train_processed.shape[1] > model.max_features:
            with torch.no_grad():
                if not hasattr(model, "V"):
                    x_dev = torch.from_numpy(X_train_processed).to(device).float()
                    q = min(x_dev.shape[0], model.max_features)
                    _, _, V = torch.pca_lowrank(x_dev, q=q)
                    model.V = V
                    model.V.requires_grad_(False)
        

        for epoch in range(1, config["epochs"] + 1):
            iterable = range(config['steps_per_epoch'])
            if config["show_progress"]:
                iterable = tqdm(iterable, desc=f"Finetuning Epoch {epoch}")

            for step in iterable:
                optimizer.zero_grad()
                
                episode_size = config['support_size'] + config['query_size']
                if episode_size > n_samples:
                    scale = n_samples / float(episode_size)
                    s = max(1, int(config['support_size'] * scale))
                    q = max(1, int(config['query_size'] * scale))
                else:
                    s, q = config['support_size'], config['query_size']

                indices = np.random.choice(n_samples, s + q, replace=False)
                X_episode = torch.from_numpy(X_train_processed[indices]).float().to(device)
                y_episode = torch.from_numpy(y_train_processed[indices]).long().to(device)
                
                 # JIT PCA projection on GPU without affecting gradients
                if getattr(model, "feature_reduction", "pca") == "pca" and X_episode.shape[-1] > model.max_features and hasattr(model, "V"):
                    with torch.no_grad():
                        X_episode = X_episode @ model.V
                
                
                X_support = X_episode[:s].unsqueeze(0)
                y_support = y_episode[:s].unsqueeze(0)
                X_query   = X_episode[s:].unsqueeze(0)
                y_query   = y_episode[s:]

                # Apply padding to match model's expected feature count
                X_support = pad_x(X_support, model.max_features)
                X_query = pad_x(X_query, model.max_features)
                
                x_src = torch.cat([X_support, X_query], dim=1)
                                
                ys = y_support.squeeze(0).long()
                yq = y_query.long()

                supp = torch.unique(ys)
                max_id = int(max(int(ys.max()), int(yq.max())))
                emap = torch.full((max_id + 1,), -1, dtype=torch.long, device=ys.device)
                for i, c in enumerate(supp):
                    emap[int(c)] = i

                ys_m = emap[ys]
                yq_m = emap[yq]

                # Skip episode if query label isn't in support (avoids OOB inside model/CE)
                if (yq_m < 0).any():
                    continue

                logits = model.model(x_src=x_src, y_src=ys_m.unsqueeze(0).unsqueeze(-1).float(), task='cls')

                if logits.dim() == 3:
                    if logits.size(1) == 1:
                        logits = logits[:, 0, :]
                    elif logits.size(0) == 1:
                        logits = logits[0, :, :]
                    else:
                        Q = yq_m.size(0)
                        logits = logits[-Q:, 0, :]
                elif logits.dim() == 2:
                    pass
                elif logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected logits shape {tuple(logits.shape)}; expected 2D or 3D.")

                # --- Guard CE range and compute loss with EPISODIC targets ---
                if int(yq_m.max().item()) >= logits.size(-1):
                    continue
                loss = loss_fn(logits, yq_m)

                loss.backward()
                optimizer.step()
                
                if config["show_progress"]:
                    iterable.set_postfix(loss=f"{loss.item():.4f}")
        
        # Clean up: ensure model is in eval mode and on correct device after finetuning
        model.model.eval()
        model.model.to(device)
        
        # Ensure all parameters and buffers are on the correct device
        for param in model.model.parameters():
            param.data = param.data.to(device)
        for buffer in model.model.buffers():
            buffer.data = buffer.data.to(device)
        
        logger.info("[TuningManager] Episodic fine-tuning complete")



    def _finetune_mitra_pure_sft(self, model, X_train_processed, y_train_processed, params=None, peft_config=None):
        """
        PURE SFT FOR MITRA
    
        Unlike TabICL, pure SFT works naturally for Mitra because:
        1. Forward method is flexible with sequence dimensions
        2. Padding masks handle variable-length sequences
        3. Better for task-specific optimization
    
        This is suitable when you want to fully optimize for target task accuracy.
        """
        logger.info("[TuningManager] Starting Mitra Pure SFT Fine-tuning")

        config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "epochs": 5,
            "learning_rate": 1e-5,
            "batch_size": 128,
            "show_progress": True,
            "weight_decay": 1e-4,
            "warmup_epochs": 1
        }
        if params:
            config.update(params)
        logger.debug(f"[TuningManager] Using config: {config}")

        device = torch.device(config["device"])
        model.to(device)
        model.train()

        for param in model.parameters():
            param.data = param.data.to(device)

        if peft_config:
            try:
                model = apply_tabular_lora("Mitra", model, peft_config)
                logger.info("[TuningManager] Applied LoRA adapters to Mitra (pure SFT)")
            except Exception as e:
                logger.warning(f"[TuningManager] LoRA failed: {e}")

    # Create dataset
        dataset = TensorDataset(
            torch.from_numpy(X_train_processed).float(),
            torch.from_numpy(y_train_processed).long()
        )
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(),
                                lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])
        loss_fn = torch.nn.CrossEntropyLoss()

        # LR scheduler
        total_steps = len(dataloader) * config["epochs"]
        warmup_steps = len(dataloader) * config["warmup_epochs"]

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        for epoch in range(1, config["epochs"] + 1):
            iterable = dataloader
            if config["show_progress"]:
                iterable = tqdm(dataloader, desc=f"Pure SFT Epoch {epoch}", leave=False)

            epoch_loss = 0
            for X_batch, y_batch in iterable:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

            # Convert to episodic format for Mitra
            # [B, F] -> [B, 1, F] (treat entire batch as query with no support)
                X_support = X_batch.unsqueeze(1)
                y_support = y_batch.unsqueeze(1)
                X_query = X_batch.unsqueeze(1)
            
                b, f = X_support.shape[0], X_support.shape[2]
                padding_features = torch.zeros(b, f, dtype=torch.bool, device=device)
                padding_obs_support = torch.zeros_like(y_support, dtype=torch.bool, device=device)
                padding_obs_query = torch.zeros(b, X_query.shape[1], dtype=torch.bool, device=device)

                optimizer.zero_grad()
            
                logits = model(
                    x_support=X_support, y_support=y_support, x_query=X_query,
                    padding_features=padding_features,
                    padding_obs_support=padding_obs_support,
                    padding_obs_query__=padding_obs_query
                )

                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
    
                if config["show_progress"]:
                    iterable.set_postfix(loss=f"{loss.item():.4f}")

            logger.info(f"[TuningManager] Epoch {epoch}: Avg Loss = {epoch_loss/len(dataloader):.4f}")

        logger.info("[TuningManager] Pure SFT fine-tuning complete")


    def _finetune_tabdpt_pure_sft(self, model, X_train_processed, y_train_processed, params=None, processor=None, peft_config=None):
        """
        PURE SUPERVISED FINE-TUNING FOR TabDPT
    
        Standard batch-wise supervised training without episodic sampling.
        Works similarly to Mitra's pure SFT approach.
    
        Args:
            model: TabDPTClassifier instance
            X_train_processed: Preprocessed features (numpy array)
            y_train_processed: Target labels (numpy array)
            params: Fine-tuning hyperparameters
            processor: TabDPT processor with projector
            peft_config: PEFT configuration (optional)
        """
    
        logger.info("[TuningManager] Starting TabDPT Pure Supervised Fine-Tuning")
        
        # Normalize labels to contiguous 0..C-1 IDs (prevents CE out-of-range)
        classes, y_train_processed = np.unique(y_train_processed, return_inverse=True)
        y_train_processed = y_train_processed.astype(np.int64)
        num_classes = len(classes)
        logger.info(f"[TuningManager] Detected {num_classes} classes in training data (contiguous remap)")
        # (Optional) keep mapping if you need to inverse-transform later
        model.classes_ = classes


        config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "epochs": 5,
            "learning_rate": 2e-5,
            "batch_size": 32,
            "show_progress": True,
            "weight_decay": 1e-4,
            "warmup_epochs": 1
        }
        if params:
            config.update(params)
        logger.debug(f"[TuningManager] Using config: {config}")
    
        device = torch.device(config["device"])

        if peft_config:
            try:
                model.model = apply_tabular_lora("TabDPT", model.model, peft_config)
                logger.info("[TuningManager] Applied LoRA adapters to TabDPT (Pure SFT)")
            except Exception as e:
                logger.warning(f"[TuningManager] PEFT failed: {e}. Proceeding with base fine-tuning")
    
        model.model.to(device)
        model.model.train()
    
        for param in model.model.parameters():
            param.data = param.data.to(device)
        for buffer in model.model.buffers():
            buffer.data = buffer.data.to(device)
    
        model.device = str(device)
        
        # Compute PCA basis on GPU once, no autograd (only if needed)
        if getattr(model, "feature_reduction", "pca") == "pca" and X_train_processed.shape[1] > model.max_features:
            with torch.no_grad():
                if not hasattr(model, "V"):
                    x_dev = torch.from_numpy(X_train_processed).to(device).float()
                    q = min(x_dev.shape[0], model.max_features)
                    _, _, V = torch.pca_lowrank(x_dev, q=q)
                    model.V = V
                    model.V.requires_grad_(False)
    

        trainable_params = list(model.model.parameters())
        if processor and hasattr(processor, 'custom_preprocessor_') and hasattr(processor.custom_preprocessor_, 'projector_'):
            trainable_params += list(processor.custom_preprocessor_.projector_.parameters())
            logger.info("[TuningManager] Including projector parameters in optimizer")
    
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        dataset = TensorDataset(
            torch.from_numpy(X_train_processed).float(),
            torch.from_numpy(y_train_processed).long()
        )
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        total_steps = len(dataloader) * config["epochs"]
        warmup_steps = len(dataloader) * config["warmup_epochs"]
    
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
        step = 0
        for epoch in range(1, config["epochs"] + 1):
            epoch_loss = 0.0
            iterable = dataloader
        
            if config["show_progress"]:
                iterable = tqdm(dataloader, desc=f"Pure SFT Epoch {epoch}", leave=False)
        
            for X_batch, y_batch in iterable:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
            # JIT PCA projection on GPU without affecting gradients
                if getattr(model, "feature_reduction", "pca") == "pca" and X_batch.shape[-1] > model.max_features and hasattr(model, "V"):
                    with torch.no_grad():
                        X_batch = X_batch @ model.V
            
                X_support = X_batch.unsqueeze(1)
                y_support = y_batch.unsqueeze(1)
                X_query = X_batch.unsqueeze(1)
            
                X_support = pad_x(X_support, model.max_features)
                X_query = pad_x(X_query, model.max_features)
            
                x_src = torch.cat([X_support, X_query], dim=1)
            
                optimizer.zero_grad()
                
                logits = model.model(
                    x_src=x_src,
                    y_src=y_support.unsqueeze(-1).float(),
                    task='cls'
                )
                
                logits = logits[..., :num_classes]            # trim to observed classes cap
                if logits.dim() == 3:
                    logits = logits.squeeze(0)                # normalize to [B, C]
                elif logits.dim() != 2:
                    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

                # CE requires targets in [0, C-1]; if head width < num_classes, drop OOR rows
                C_eff = logits.size(-1)
                y_batch = y_batch.long()

                valid = (y_batch >= 0) & (y_batch < C_eff)
                if not valid.all():
                    # skip this minibatch if nothing valid remains
                    if not valid.any():
                        continue
                    logits = logits[valid]
                    y_batch = y_batch[valid]

                loss = loss_fn(logits, y_batch)

            
                loss.backward()
                optimizer.step()
                scheduler.step()
            
                epoch_loss += loss.item()
                step += 1
            
                if config["show_progress"]:
                    iterable.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}"
                    )
        
            avg_loss = epoch_loss / len(dataloader)
            logger.info(
                f"[TuningManager] Epoch [{epoch}/{config['epochs']}]: "
                f"Avg Loss = {avg_loss:.4f}, "
                f"LR = {scheduler.get_last_lr()[0]:.2e}"
            )
    
        model.model.eval()
        logger.info("[TuningManager] TabDPT Pure Supervised Fine-Tuning Complete")
    
        return model

    def _finetune_tabicl_simple_sft(self, model, X_train_processed, y_train_processed, params=None, peft_config=None):
        """
        TabICL : Convert supervised batches to episodic format
        """

        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'epochs': 5,
            'learning_rate': 1e-5,
            'batch_size': 16,
            'show_progress': True,
        }
        if params:
            config.update(params)
    
        device = torch.device(config['device'])
    
    # Initialize
        model.fit(X_train_processed, y_train_processed)
        model._load_model()
        model.model_.to(device).train()
        
        C_out = None
        with torch.no_grad():
            # make one tiny 1-class episode from the first few rows
            X_np = X_train_processed if isinstance(X_train_processed, np.ndarray) else X_train_processed.to_numpy()
            y_np = y_train_processed if isinstance(y_train_processed, np.ndarray) else y_train_processed.to_numpy()

            # pick a class that has >= (support_size + query_size) examples; fall back to any class
            s_sz = int(config.get("support_size", 48))
            q_sz = int(config.get("query_size", 32))
            need = s_sz + q_sz

            cls, idx = None, None
            for c in np.unique(y_np):
                cand = np.nonzero(y_np == c)[0]
                if cand.size >= need:
                    idx = cand[:need]
                    cls = c
                    break
            if idx is None:
                idx = np.arange(min(need, len(y_np)))
                cls = y_np[idx[0]]

            X_ep = torch.from_numpy(X_np[idx]).float().unsqueeze(0).to(device)   # [1, S+Q, F]
            ys   = torch.full((s_sz,), 0, dtype=torch.long, device=device)       # all support -> class 0
            # pack as your forward expects: first S as support, rest as query
            logits_probe = model.model_(X_ep, ys.unsqueeze(0))                   # [1, Q, C_eff] typically
            C_out = int(logits_probe.squeeze(0).size(-1))

        # safety
        if C_out <= 0:
            raise RuntimeError("Could not infer logits width (C_out).")
            
            

    
    # Standard dataset
        dataset = TensorDataset(
            torch.from_numpy(X_train_processed).float(),
            torch.from_numpy(y_train_processed).long()
        )
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
        optimizer = torch.optim.Adam(model.model_.parameters(), lr=config['learning_rate'])
        loss_fn = torch.nn.CrossEntropyLoss()
    
        for epoch in range(1, config['epochs'] + 1):
            iterable = tqdm(dataloader, desc=f"SFT Epoch {epoch}") if config['show_progress'] else dataloader
            epoch_loss = 0
        
            for X_batch, y_batch in iterable:
                batch_size = X_batch.shape[0]
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
            
            # Split batch in half: first half = support, second half = query
                mid = batch_size // 2
                if mid == 0:  # Skip if batch too small
                    continue
                X_support = X_batch[:mid]
                y_support = y_batch[:mid]
                X_query = X_batch[mid:]
                y_query = y_batch[mid:]
            
                # Ensure X_support and X_query are 2D [samples, features] before concatenation
                if X_support.dim() > 2:
                    X_support = X_support.view(mid, -1)  # Flatten extra dimensions
                if X_query.dim() > 2:
                    X_query = X_query.view(-1, X_query.shape[-1])  # Flatten extra dimensions
                
                X_episode = torch.cat([X_support, X_query], dim=0).unsqueeze(0)  # [1, batch_size, features]

                ys = y_support.squeeze(0).long() if y_support.dim() > 1 else y_support.long()
                yq = y_query.squeeze(0).long() if y_query.dim() > 1 else y_query.long()

                supp = torch.unique(ys)
                # keep at most C_out classes so the head can represent them
                keep = supp[:C_out]

                # build map only for kept classes; others -> -1 (excluded)
                yq_m = torch.full_like(yq, -1)
                ys_m = torch.full_like(ys, -1)
                for i, c in enumerate(keep):
                    ys_m[ys == c] = i
                    yq_m[yq == c] = i

                # prune support rows that were dropped
                keep_mask = (ys_m >= 0)
                if not keep_mask.any():
                    continue
                ys_m = ys_m[keep_mask]
                # Use mid directly for support size (before filtering) and apply keep_mask correctly
                # X_episode shape: [1, batch_size, features], mid is the original support size
                # Index support samples first, then apply keep_mask to avoid dimension issues
                X_support_all = X_episode[:, :mid, :]  # [1, mid, F]
                X_support_kept = X_support_all[:, keep_mask, :]  # [1, kept_support, F]
                X_query_part = X_episode[:, mid:, :]  # [1, query_size, F]
                # Ensure both tensors have same number of dimensions (both should be 3D)
                X_episode = torch.cat([X_support_kept, X_query_part], dim=1)

                # if any query label was excluded, skip this episode (avoids OOB gathers)
                if (yq_m < 0).any():
                    continue

                # forward with episodic labels (contiguous, ≤ C_out)
                logits = model.model_(X_episode, ys_m.unsqueeze(0))  # [1, Q, <=C_out]
                logits = logits.squeeze(0)        # [Q, <=C_out]
                # ensure mapping fits the actual head width (in case adapters changed it mid-run)
                if logits.size(-1) < yq_m.max().item() + 1:
                    continue  # skip this episode if it exceeds head capacity

                loss = loss_fn(logits, yq_m)


                loss.backward()
                optimizer.step()
            
                epoch_loss += loss.item()
                if config['show_progress']:
                    iterable.set_postfix(loss=f"{loss.item():.4f}")
        
            logger.info(f"[TuningManager] Epoch {epoch}: Loss = {epoch_loss/len(dataloader):.4f}")
    
        model.model_.eval()
        return model