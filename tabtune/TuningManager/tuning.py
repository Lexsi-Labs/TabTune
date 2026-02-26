import torch
from torch.optim import Adam
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from functools import partial
import numpy as np
import pandas as pd
import logging
import os
from sklearn.metrics import r2_score


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
from ..models.orionmsp_v15.sklearn.classifier import OrionMSPv15Classifier
from ..models.orion_msp.sklearn.preprocessing import OrionMSPMetaDataset
from ..models.contexttab.contexttab import ConTextTabClassifier
from ..models.mitra.tab2d import Tab2D
from torch.utils.data import TensorDataset
from ..models.orion_bix.sklearn.classifier import OrionBixClassifier
from ..models.tabdpt.classifier import TabDPTClassifier
from ..models.tabdpt.utils import pad_x
from ..models.tabdpt.model import TabDPTModel
from ..models.limix.classifier import LimixClassifier
from ..models.regression.tabpfn.regressor import TabPFNRegressorWrapper
from ..models.regression.contexttab.regressor import ConTextTabRegressorWrapper
from ..models.regression.tabdpt.regressor import TabDPTRegressorWrapper
from ..models.regression.mitra.regressor import MitraRegressorWrapper
from ..models.regression.limix.regressor_wrapper import LimixRegressorWrapper


from ..models.contexttab.contexttab import to_device

from .peft_utils import apply_tabular_lora

logger = logging.getLogger(__name__)



class TuningManager:
    """
    Handles the model adaptation process
    """
    def tune(self, model, X_train, y_train, strategy='inference', params=None, processor=None):
        
        params_copy = dict(params) if isinstance(params, dict) else {}
        finetune_mode = params_copy.get('finetune_mode', 'turn_by_turn')

        # --- Regression wrappers: allow ContextTab finetune (turn-by-turn) ---
        if isinstance(model, (TabPFNRegressorWrapper, ConTextTabRegressorWrapper,
                              TabDPTRegressorWrapper, MitraRegressorWrapper,LimixRegressorWrapper)):
            if strategy == 'inference':
                logger.info("[TuningManager] Regression model - inference path")
                model.fit(X_train, y_train)
                return model

            # ContextTab regression finetune (turn-by-turn)
            if isinstance(model, ConTextTabRegressorWrapper) and strategy == 'finetune':
                if finetune_mode not in ('turn_by_turn', 'tbt'):
                    raise ValueError(
                        f"ContextTab regression finetune currently supports finetune_mode "
                        f"'turn_by_turn' (or 'tbt'). Got '{finetune_mode}'."
                    )
                return self._finetune_contexttab_regression_turn_by_turn(model, X_train, y_train, params_copy)

            # Limix regression finetune
            if isinstance(model, LimixRegressorWrapper) and strategy == "finetune":
                return self._finetune_limix_regression(model, X_train, y_train, params_copy)

            if isinstance(model, TabDPTRegressorWrapper) and strategy == "finetune":
                params_copy = dict(params) if isinstance(params, dict) else {}
                return self._finetune_tabdpt_regression_turn_by_turn(model, X_train, y_train, params_copy)

            # Mitra regression finetune (turn-by-turn)
            if isinstance(model, MitraRegressorWrapper) and strategy == "finetune":
                if finetune_mode not in ("turn_by_turn", "tbt"):
                    raise ValueError(f"Mitra regression finetune supports finetune_mode 'turn_by_turn' (or 'tbt'). Got '{finetune_mode}'.")
                return self._finetune_mitra_regression_turn_by_turn(model, X_train, y_train, params_copy)

            # TabPFN regression finetune (turn-by-turn)
            if isinstance(model, TabPFNRegressorWrapper) and strategy == "finetune":
                if finetune_mode not in ("turn_by_turn", "tbt"):
                    raise ValueError(
                        f"TabPFN regression finetune supports finetune_mode 'turn_by_turn' (or 'tbt'). Got '{finetune_mode}'."
                    )
                return self._finetune_tabpfn_regression_turn_by_turn(model, X_train, y_train, params_copy)



            raise NotImplementedError(
                f"Regression fine-tuning not implemented yet for model={type(model).__name__}. "
                f"Currently implemented: ContextTab + strategy='finetune' + finetune_mode='turn_by_turn'."
            )
        
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
            selected_strategy = 'finetune'

        is_finetuned = False
        original_is_tab2d = isinstance(model, Tab2D)


        if (isinstance(model, Tab2D) or original_is_tab2d) and selected_strategy in ('finetune', 'peft'):
            if finetune_mode == 'sft':
                logger.info("[TuningManager] Using Pure SFT for Mitra (task-optimized)")
                self._finetune_mitra_pure_sft(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            else:  # default: 'meta-learning'
                logger.info("[TuningManager] Using Episodic Meta-Learning for Mitra (default)")
                self._finetune_mitra(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            is_finetuned = True
        
        elif isinstance(model, TabPFNClassifier) and selected_strategy in ('finetune', 'peft'):
            if finetune_mode == 'sft':
                logger.info("[TuningManager] Using Pure SFT for TabPFN (task-optimized)")
                self._finetune_tabpfn_pure_sft(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            else:  # default: 'meta-learning'
                logger.info("[TuningManager] Using Episodic Meta-Learning for TabPFN (default)")
                self._finetune_tabpfn(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            is_finetuned = True
        
        elif isinstance(model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, OrionMSPv15Classifier)) and selected_strategy in ('finetune', 'peft'):
            if finetune_mode == 'meta-learning':
                logger.info("[TuningManager] Meta Learning based FT")
                self._finetune_tabicl(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            else:
                logger.info("[TuningManager] Performing SFT")
                self._finetune_tabicl_simple_sft(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            is_finetuned = True
        
        elif isinstance(model, ConTextTabClassifier) and selected_strategy in ('finetune', 'peft'):
            self._full_finetune_model(model, X_train, y_train, params=params_copy, processor=processor, peft_config=peft_config)
            is_finetuned = True
        
        elif isinstance(model, TabDPTClassifier) and selected_strategy in ('finetune','peft'):
            if finetune_mode == 'sft':
                logger.info("[TuningManager] Using Pure SFT for TabDPT (task-optimized)")
                self._finetune_tabdpt_pure_sft(model, X_train, y_train, params=params_copy, processor=processor, peft_config=peft_config)
            else:  # default: 'meta-learning'
                logger.info("[TuningManager] Using Episodic Meta-Learning for TabDPT (default)")
                self._finetune_tabdpt(model, X_train, y_train, params=params_copy, processor=processor, peft_config=peft_config)
            is_finetuned = True


        elif isinstance(model, LimixClassifier) and selected_strategy in ('finetune', 'peft'):
            msg = "[TuningManager] Limix fine-tuning not supported; falling back to inference-mode fit (.fit) only."
            print(msg)
            logger.warning(msg)
            logger.info("falling back to inference mode")
            # Fall back to the inference behavior (your existing inference branch calls .fit)
            model.fit(X_train, y_train)

            # Not finetuned -> don't save/reload checkpoint
            is_finetuned = False


        
        elif isinstance(model, (Tab2D)) and selected_strategy == 'inference':
            logger.info("[TuningManager] In-context learning model in inference mode. No training needed.")
            pass
        elif isinstance(model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, LimixClassifier, OrionMSPv15Classifier)) and selected_strategy == 'inference':
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
            logger.info("[TuningManager] RECOMMENDATION: Use standard finetune strategy for ConTextTab instead of 'peft'")
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
            logger.info("[TuningManager] RECOMMENDATION: Use standard finetune strategy for TabPFN instead of 'peft'")
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

    

    def _finetune_tabicl(self, model: (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, OrionMSPv15Classifier), X_train_processed: np.ndarray, y_train_processed: np.ndarray, params: dict | None = None, peft_config=None):
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
                elif isinstance(model, OrionMSPv15Classifier):
                    model_key = "OrionMSPv1.5"
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


    def _finetune_tabicl_pure_sft(self, model: (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, OrionMSPv15Classifier) , X_train_processed, y_train_processed, params=None, peft_config=None):
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


    def get_default_config(self, model, selected_strategy: str, finetune_mode: str, processor=None) -> dict:
        """
        Return the default config that would be used for this model/strategy/mode.
        This must match the dicts defined inside the _finetune_* methods.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # TabICL / Orion MSP / Orion Bix
        if isinstance(model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, OrionMSPv15Classifier)):
            if finetune_mode == "meta-learning":
                return {
                    "device": device,
                    "epochs": 5,
                    "learning_rate": 2e-6,
                    "show_progress": True,
                    "support_size": 48,
                    "query_size": 32,
                    "n_episodes": 1000,
                    # keep these visible too if you support them
                    # "finetune_method": None,
                    # "peft_config": None,
                }
            else:
                # simple SFT defaults (_finetune_tabicl_simple_sft)
                return {
                    "device": device,
                    "epochs": 5,
                    "learning_rate": 1e-5,
                    "batch_size": 16,
                    "show_progress": True,
                }

        # TabPFN
        if isinstance(model, TabPFNClassifier):
            if finetune_mode == "sft":
                return {
                    "device": device,
                    "epochs": 25,
                    "learning_rate": 1e-5,
                    "show_progress": True,
                    "max_episode_size": None,   # you can set to len(X) only at fit-time
                    "query_set_ratio": 0.3,
                    "weight_decay": 1e-4,
                }
            else:
                return {
                    "device": device,
                    "epochs": 3,
                    "learning_rate": 1e-5,
                    "batch_size": 256,
                    "show_progress": True,
                }

        # ConTextTab full FT
        if isinstance(model, ConTextTabClassifier):
            return {
                "device": device,
                "epochs": 5,
                "learning_rate": 1e-4,
                "batch_size": 128,
                "show_progress": True,
            }

        # TabDPT
        if isinstance(model, TabDPTClassifier):
            if finetune_mode == "sft":
                return {
                    "device": device,
                    "epochs": 5,
                    "learning_rate": 2e-5,
                    "batch_size": 32,
                    "show_progress": True,
                    "weight_decay": 1e-4,
                    "warmup_epochs": 1,
                }
            else:
                return {
                    "device": device,
                    "epochs": 5,
                    "learning_rate": 1e-5,
                    "batch_size": 8,
                    "support_size": 512,
                    "query_size": 256,
                    "steps_per_epoch": 100,
                    "show_progress": True,
                }

        # Mitra / Tab2D
        if isinstance(model, Tab2D):
            if finetune_mode == "sft":
                return {
                    "device": device,
                    "epochs": 5,
                    "learning_rate": 1e-5,
                    "batch_size": 128,
                    "show_progress": True,
                    "weight_decay": 1e-4,
                    "warmup_epochs": 1,
                }
            else:
                return {
                    "device": device,
                    "epochs": 3,
                    "learning_rate": 1e-5,
                    "batch_size": 4,
                    "support_size": 128,
                    "query_size": 128,
                    "steps_per_epoch": 50,
                    "show_progress": True,
                }

        # Limix
        # if isinstance(model, LimixClassifier):
        #     return {
        #         "device": device,
        #         "epochs": 5,
        #         "learning_rate": 1e-5,
        #         "show_progress": True,
        #         "support_size": 48,
        #         "query_size": 32,
        #         "n_episodes": 1000,
        #     }

        # fallback: no tuning defaults known
        return {"device": device}



    def _contexttab_regression_make_episode(self, model, X_all, y_all, context_size, query_size, seed=None):
        """
        Build one regression episode for ContextTab:
        - context rows = training context (targets known)
        - query rows   = test/query (targets masked in train_target; true labels returned separately)
        Returns: tokenized_data dict ready for model forward + loss.
        """
        if not isinstance(X_all, pd.DataFrame):
            X_all = pd.DataFrame(X_all)
        if isinstance(y_all, (pd.DataFrame, pd.Series)):
            y_all = np.array(y_all).reshape(-1)
        y_all = np.asarray(y_all).astype(float)

        n = len(X_all)
        if n < (context_size + 1):
            # fall back: tiny datasets
            context_size = max(1, n - 1)
            query_size = 1

        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)

        ctx_idx = idx[:context_size]
        remaining = idx[context_size:]
        if len(remaining) == 0:
            # if context consumes all rows, reuse some rows as query
            qry_idx = rng.choice(ctx_idx, size=query_size, replace=True)
        else:
            qry_idx = rng.choice(remaining, size=min(query_size, len(remaining)), replace=False)
            if len(qry_idx) < query_size:
                # top up with replacement from remaining/context
                pool = remaining if len(remaining) > 0 else ctx_idx
                extra = rng.choice(pool, size=(query_size - len(qry_idx)), replace=True)
                qry_idx = np.concatenate([qry_idx, extra])

        X_ctx = X_all.iloc[ctx_idx].copy()
        y_ctx = pd.DataFrame({'TARGET': y_all[ctx_idx]}, index=X_ctx.index)

        X_qry = X_all.iloc[qry_idx].copy()
        y_qry = pd.DataFrame({'TARGET': y_all[qry_idx]}, index=X_qry.index)

        # tokenizer returns:
        # - data: dict with 'target' where query rows are masked (<= -99 sentinel)
        # - labels: true labels for query rows (normalized for l2 if tokenizer does it)
        data, labels, label_classes = model.tokenizer(
            X_ctx, y_ctx,
            X_qry, y_qry,
            model.classification_or_regression
        )

        target_mean, target_std = 0, 0
        if model.classification_or_regression == 'regression' and getattr(model, 'regression_type', 'l2') == 'l2':
            _, target_mean, target_std = model.tokenizer.standard_scale_column(y_ctx, y_qry)

        tokenized = {
            'data': data,
            'num_rows': context_size + query_size,
            'num_cols': X_all.shape[1] + 1,  # incl target col
            'labels': labels,                # <-- IMPORTANT for training
            'is_regression': torch.tensor(True),
            'label_classes': np.asarray(label_classes),
            'target_mean': target_mean,
            'target_std': target_std
        }
        return tokenized


    def _finetune_contexttab_regression_turn_by_turn(self, model, X_train, y_train, params):
        """
        Turn-by-turn regression fine-tuning for ContextTab.
        Uses episodic (context, query) batches and optimizes regression loss on query rows.
        """
        logger = logging.getLogger(__name__)

        device = params.get('device', getattr(model, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # loop params
        epochs = int(params.get('epochs', 1))
        steps_per_epoch = int(params.get('steps_per_epoch', 200))
        context_size = int(params.get('context_size', 256))
        query_size = int(params.get('query_size', 64))
        seed = int(params.get('seed', 42))

        # optim params
        lr = float(params.get('lr', 1e-5))
        weight_decay = float(params.get('weight_decay', 0.01))
        clip_grad_norm = float(params.get('clip_grad_norm', 1.0))

        # checkpoint (optional)
        save_checkpoint_path = params.get('save_checkpoint_path', None)

        # ensure context is stored (ConTextTab API expectation)
        model.fit(X_train, y_train)

        # train mode
        model.model.train()
        model.model.to(device)

        opt = AdamW(model.model.parameters(), lr=lr, weight_decay=weight_decay)

        global_step = 0
        for ep in range(epochs):
            running_loss = 0.0

            step_iter = range(steps_per_epoch)
            if params.get("show_progress", False):
                step_iter = tqdm(
                    step_iter,
                    desc=f"ContextTab-Reg TBT | Epoch {ep+1}/{epochs}",
                    leave=True
                )
                
            for s in step_iter:
                tokenized = self._contexttab_regression_make_episode(
                    model=model,
                    X_all=X_train,
                    y_all=y_train,
                    context_size=context_size,
                    query_size=query_size,
                    seed=seed + global_step
                )

                # move to device
                tokenized = to_device(tokenized, device, raise_on_unexpected=False)

                out = model.model(**tokenized)

                # ContextTab forward sometimes returns (logits, aux) or a dict-like output
                if isinstance(out, tuple):
                    logits = out[0]
                elif isinstance(out, dict):
                    logits = out.get("logits", out.get("preds", out.get("output", out)))
                else:
                    logits = out


                # labels are required for regression loss (query rows)
                labels = tokenized.get('labels', None)
                if labels is None:
                    raise RuntimeError(
                        "ContextTab regression finetune requires tokenized['labels'] from tokenizer(). "
                        "If you see this, your tokenization path dropped labels."
                    )

                # compute regression loss on query rows (where train_target <= -99)
                train_target = tokenized ['data']['target']
                ret = model.model.compute_regression_output_loss_and_metric(
                    logits=logits,
                    labels=labels,
                    train_target=train_target
                )

                # ContextTab regression may return (loss, metric) OR (loss, metric, ...)
                if isinstance(ret, tuple):
                    loss = ret[1]
                    metric = ret[2] if len(ret) > 1 else None
                else:
                    loss = ret
                    metric = None

                # loss must be a scalar for backward()
                if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                    loss = loss.mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()

                if clip_grad_norm is not None and clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), clip_grad_norm)

                opt.step()

                running_loss += float(loss.detach().cpu().item())
                global_step += 1

            avg_loss = running_loss / max(1, steps_per_epoch)
            logger.info(
                f"[ContextTab-Regression-TBT] epoch={ep+1}/{epochs} "
                f"steps={steps_per_epoch} avg_loss={avg_loss:.5f}"
            )

            if save_checkpoint_path:
                os.makedirs(os.path.dirname(save_checkpoint_path), exist_ok=True)
                torch.save({'model_state_dict': model.model.state_dict()}, save_checkpoint_path)
                logger.info(f"[ContextTab-Regression-TBT] saved checkpoint -> {save_checkpoint_path}")

        model.model.eval()
        return model




    def safe_r2(y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        # finite + same length
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]

        # need at least 2 points
        if y_true.size < 2:
            return np.nan

        # undefined if variance is 0
        if np.allclose(y_true, y_true[0]):
            return np.nan

        return r2_score(y_true, y_pred)
    
    def _finetune_limix_regression(self, model, X_train, y_train, params):
        """
        Episodic fine-tuning for Limix regression.
        Builds (support, query) episodes and trains MSE on query predictions.
        """
        import numpy as np
        import torch
        from torch.optim import AdamW
        from tqdm import tqdm

        logger.info("[TuningManager] Starting Limix regression fine-tuning")

        config = {
            "device": params.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            "epochs": int(params.get("epochs", 3)),
            "steps_per_epoch": int(params.get("steps_per_epoch", 100)),
            "support_size": int(params.get("support_size", params.get("context_size", 256))),
            "query_size": int(params.get("query_size", 64)),
            "lr": float(params.get("lr", 1e-5)),
            "weight_decay": float(params.get("weight_decay", 0.01)),
            "clip_grad_norm": float(params.get("clip_grad_norm", 1.0)),
            "seed": int(params.get("seed", 42)),
            "show_progress": bool(params.get("show_progress", True)),
        }

        device = torch.device(config["device"])

        # Ensure dataframe/series -> numpy
        if hasattr(X_train, "to_numpy"):
            X_np = X_train.to_numpy()
        else:
            X_np = np.asarray(X_train)

        if hasattr(y_train, "to_numpy"):
            y_np = y_train.to_numpy()
        else:
            y_np = np.asarray(y_train)

        X_np = X_np.astype(np.float32)
        y_np = y_np.astype(np.float32).reshape(-1)

        n = X_np.shape[0]
        if n < (config["support_size"] + 2):
            raise ValueError(f"Not enough rows for episodic finetune: n={n}, support={config['support_size']}")

        # Normalize target like LimixRegressor.fit does
        y_mean = float(np.mean(y_np))
        y_std = float(np.std(y_np)) if float(np.std(y_np)) > 1e-12 else 1.0
        y_norm = (y_np - y_mean) / y_std

        # Under the wrapper, you have an ensemble
        # Each estimator has .model (torch module)
        estimators = getattr(model, "estimators", None)

        # Some older codepaths might call it "models"
        if estimators is None:
            estimators = getattr(model, "models", None)

        # Single-estimator fallback
        if estimators is None and hasattr(model, "model"):
            estimators = [model]

        if not estimators:
         # Important: wrapper creates estimators only after fit()
            if hasattr(model, "fit"):
                model.fit(X_train, y_train)
                estimators = getattr(model, "estimators", None) or getattr(model, "models", None)
            if not estimators:
                raise AttributeError(
                    f"Could not find Limix estimators on {type(model).__name__}. "
                    "Expected `.estimators` (wrapper/ensemble) or `.model` (single regressor)."
            )

        mse = torch.nn.MSELoss()

        rng = np.random.default_rng(config["seed"])

        for est_i, est in enumerate(estimators):
            torch_model = getattr(est, "model", None)
            if torch_model is None or not isinstance(torch_model, torch.nn.Module):
                raise RuntimeError(f"Estimator {est_i} has no torch model to finetune.")

            torch_model.to(device)
            torch_model.train()

            opt = AdamW(torch_model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

            for epoch in range(config["epochs"]):
                it = range(config["steps_per_epoch"])
                if config["show_progress"]:
                    it = tqdm(it, desc=f"Limix-Reg FT | est {est_i+1}/{len(estimators)} | epoch {epoch+1}/{config['epochs']}")

                for step in it:
                    # sample episode indices
                    total = config["support_size"] + config["query_size"]
                    idx = rng.choice(n, size=total, replace=False)

                    s = config["support_size"]
                    sup_idx = idx[:s]
                    qry_idx = idx[s:]

                    X_sup = X_np[sup_idx]
                    y_sup = y_norm[sup_idx]
                    X_qry = X_np[qry_idx]
                    y_qry = y_norm[qry_idx]

                    # Build [1, S+Q, F]
                    X_episode = np.concatenate([X_sup, X_qry], axis=0)
                    X_t = torch.from_numpy(X_episode).unsqueeze(0).to(device)

                    # y input: true for support, dummy for query (prevents leakage)
                    y_in = np.concatenate([y_sup, np.zeros_like(y_qry)], axis=0)
                    y_t = torch.from_numpy(y_in).unsqueeze(0).to(device)

                    opt.zero_grad(set_to_none=True)

                    out = torch_model(X_t, y_t, eval_pos=s, task_type="reg")

                    # Robustly extract reg output
                    if isinstance(out, dict):
                        pred = out.get("reg_output", None)
                    elif isinstance(out, (tuple, list)) and len(out) > 0:
                        pred = out[0]
                    else:
                        pred = out

                    if pred is None:
                        raise RuntimeError("Limix forward did not return reg_output.")

                    # pred is typically [1, Q, 1]
                    pred = pred.squeeze(0).squeeze(-1)  # -> [Q]
                    target = torch.from_numpy(y_qry).to(device)

                    loss = mse(pred, target)
                    loss.backward()

                    if config["clip_grad_norm"] and config["clip_grad_norm"] > 0:
                        torch.nn.utils.clip_grad_norm_(torch_model.parameters(), config["clip_grad_norm"])

                    opt.step()

                    if config["show_progress"]:
                        it.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

            #torch_model.eval()

        # After finetune, set context for inference
        # Refit context on the SAME estimators (no recreation)
        for est in estimators:
            if hasattr(est, "fit"):
                est.fit(X_train, y_train)
                if hasattr(est, "model") and isinstance(est.model, torch.nn.Module):
                    est.model.eval()

        logger.info("[TuningManager] Limix regression fine-tuning complete")
        return model

    def _finetune_tabdpt_regression_turn_by_turn(self, model, X_train, y_train, params: dict):
        """
        Key correctness points vs common broken implementations:
        - Fits the wrapper FIRST to ensure TabDPT preprocessing/caches (imputer/scaler/PCA V/X_train/y_train) exist.
          This avoids finetuning on raw/unprocessed data (distribution mismatch + dtype/categorical crashes).
        - Robustly resolves the underlying torch module for wrapper/estimator conventions.
        - Fixes pred/y shape mismatch to avoid mse broadcasting.
        - Keeps weights safe: if `fit()` recreates the torch module, we restore the pre-fit weights if possible.
        - Leaves the model in eval mode at the end.
        """
        import logging
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
    
        logger = logging.getLogger(__name__)
        logger.info("[TuningManager] Starting TabDPT regression fine-tuning (turn-by-turn)")
    
        params = params or {}
    
        # ---------------------------
        # Device / reproducibility
        # ---------------------------
        device_str = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
    
        epochs = int(params.get("epochs", 5))
        steps_per_epoch = int(params.get("steps_per_epoch", 100))
        context_size = int(params.get("context_size", params.get("support_size", 512)))
        query_size = int(params.get("query_size", 128))
    
        lr = float(params.get("lr", params.get("learning_rate", 1e-5)))
        weight_decay = float(params.get("weight_decay", 0.0))
        clip_grad_norm = params.get("clip_grad_norm", None)
        show_progress = bool(params.get("show_progress", True))
        seed = params.get("seed", None)
    
        if seed is not None:
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
    
        # -------------------------------------------------------------------------
        # Resolve underlying torch module robustly for TabDPT wrappers
        # -------------------------------------------------------------------------
        def _resolve_torch_module_and_cfg_host(wrapper):
            """
            Returns (torch_model, cfg_host, inner_holder)
            - torch_model: the actual torch.nn.Module
            - cfg_host: object where config attrs live (max_features, feature_reduction, V, etc.)
            - inner_holder: the object that directly contains the torch module (for eval() at end)
            """
            est = getattr(wrapper, "model", None)
            est2 = getattr(wrapper, "model_", None)
    
            # Case 1: wrapper.model is a torch module
            if isinstance(est, torch.nn.Module):
                return est, wrapper, est
    
            # Case 2: wrapper.model is an estimator that has .model (torch module)
            if est is not None and hasattr(est, "model") and isinstance(est.model, torch.nn.Module):
                return est.model, est, est
    
            # Case 3: wrapper.model_ is a torch module
            if isinstance(est2, torch.nn.Module):
                return est2, wrapper, est2
    
            # Case 4: wrapper.model_ is an estimator that has .model (torch module)
            if est2 is not None and hasattr(est2, "model") and isinstance(est2.model, torch.nn.Module):
                return est2.model, est2, est2
    
            return None, None, None
    
        torch_model, cfg_host, inner_holder = _resolve_torch_module_and_cfg_host(model)
        if torch_model is None:
            raise AttributeError(
                "TabDPTRegressorWrapper: could not locate underlying torch module. "
                "Expected either (a) `wrapper.model` to be torch.nn.Module, or "
                "(b) `wrapper.model`/`wrapper.model_` to be an estimator with `.model` torch.nn.Module."
            )
    
        prefit_state = None
        try:
            prefit_state = {k: v.detach().cpu() for k, v in torch_model.state_dict().items()}
        except Exception:
            prefit_state = None
    
        # Keep model in eval while fitting preprocessing (fit should not train)
        try:
            torch_model.eval()
        except Exception:
            pass
    
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logger.exception("[TuningManager] model.fit(X_train, y_train) failed before finetune.")
            raise
    
        # Re-resolve after fit (in case wrapper swapped internals)
        torch_model2, cfg_host2, inner_holder2 = _resolve_torch_module_and_cfg_host(model)
        if torch_model2 is None:
            raise AttributeError(
                "After model.fit, TabDPTRegressorWrapper: could not locate underlying torch module."
            )
    
        # If module object changed, attempt to restore previous weights
        if torch_model2 is not torch_model and prefit_state is not None:
            try:
                missing, unexpected = torch_model2.load_state_dict(prefit_state, strict=False)
                logger.info(
                    f"[TuningManager] Torch module changed after fit(); restored weights "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})."
                )
            except Exception:
                logger.warning(
                    "[TuningManager] Torch module changed after fit(); could not restore weights. "
                    "Continuing with post-fit weights."
                )
    
        torch_model, cfg_host, inner_holder = torch_model2, cfg_host2, inner_holder2
    
        # Keep device consistent with estimator/wrapper
        try:
            setattr(model, "device", str(device))
        except Exception:
            pass
    
        torch_model.to(device)
        torch_model.train()
    
        optimizer = AdamW(torch_model.parameters(), lr=lr, weight_decay=weight_decay)
    
        # -------------------------------------------------------------------------
        # Use PREPROCESSED cached training arrays from the fitted estimator/wrapper
        # -------------------------------------------------------------------------
        X_np = getattr(model, "X_train", None)
        y_np = getattr(model, "y_train", None)
    
        # Some implementations store caches on cfg_host (estimator). Fall back if needed.
        if X_np is None or y_np is None:
            X_np = getattr(cfg_host, "X_train", None)
            y_np = getattr(cfg_host, "y_train", None)
    
        if X_np is None or y_np is None:
            raise RuntimeError(
                "Could not find preprocessed caches `X_train` / `y_train` after fit(). "
                "Expected wrapper/estimator to expose them."
            )
    
        X_np = np.asarray(X_np, dtype=np.float32)
        y_np = np.asarray(y_np, dtype=np.float32).reshape(-1)
    
        n = len(X_np)
        if n < 2:
            raise ValueError("Need at least 2 training rows for episodic finetuning.")
    
        if n < (context_size + query_size):
            context_size = max(4, min(context_size, n // 2))
            query_size = max(1, min(query_size, n - context_size))
            logger.warning(f"[TuningManager] Shrunk episode sizes: context={context_size}, query={query_size}")
    
        # -------------------------------------------------------------------------
        # Feature reduction / padding (reuse fitted config)
        # pad_x must be importable in this module scope as in your codebase.
        # -------------------------------------------------------------------------
        max_features = getattr(cfg_host, "max_features", None)
        feature_reduction = getattr(cfg_host, "feature_reduction", None)
        V = getattr(cfg_host, "V", None)  # should be set by fit() when PCA enabled
    
        def _to_tensor(x: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(x, dtype=torch.float32, device=device)
    
        def _prep_x(x_chunk: np.ndarray) -> torch.Tensor:
            x_t = _to_tensor(x_chunk)  # (T, F)
    
            # If PCA reduction enabled, use the *fitted* V (do not recompute here)
            if feature_reduction == "pca" and max_features is not None and x_t.shape[1] > max_features:
                if V is None:
                    raise RuntimeError(
                        "feature_reduction='pca' but V is None after fit(). "
                        "Ensure estimator.fit computes/stores V."
                    )
                v_dev = V.to(device) if hasattr(V, "to") else torch.as_tensor(V, device=device)
                x_t = x_t @ v_dev
    
            x_t = x_t.unsqueeze(0)  # (1, T, F_reduced)
            if max_features is not None:
                x_t = pad_x(x_t, max_features)
            return x_t
    
        def _prep_y(y_chunk: np.ndarray) -> torch.Tensor:
            y_t = torch.as_tensor(y_chunk, dtype=torch.float32, device=device)
            return y_t.view(1, -1, 1)  # (1, S, 1)
    
        total_steps = max(1, epochs * steps_per_epoch)
        step_counter = 0
        log_every = max(1, total_steps // 20)
    
        # -------------------------------------------------------------------------
        # Episodic fine-tune loop
        # -------------------------------------------------------------------------
        for _ep in range(epochs):
            for _ in range(steps_per_epoch):
                step_counter += 1
    
                idx = np.random.permutation(n)
                s_idx = idx[:context_size]
                q_idx = idx[context_size: context_size + query_size]
    
                X_support, y_support = X_np[s_idx], y_np[s_idx]
                X_query, y_query = X_np[q_idx], y_np[q_idx]
    
                x_support_t = _prep_x(X_support)  # (1, S, F)
                x_query_t = _prep_x(X_query)      # (1, Q, F)
    
                x_src = torch.cat([x_support_t, x_query_t], dim=1)  # (1, S+Q, F)
                y_src = _prep_y(y_support)                           # (1, S, 1)
    
                optimizer.zero_grad(set_to_none=True)
    
                pred = torch_model(x_src, y_src, task="reg")
    
                # Force predictions to shape (Q,)
                pred_q = pred
                if pred_q.dim() == 3:
                    pred_q = pred_q.squeeze(0)          # (S+Q,1) or (Q,1) depending on model; common is (Q,1)
                if pred_q.dim() == 2 and pred_q.size(-1) == 1:
                    pred_q = pred_q.squeeze(-1)         # (Q,)
                pred_q = pred_q.reshape(-1)
    
                # Targets: (Q,)
                y_q = torch.as_tensor(y_query, dtype=torch.float32, device=device).reshape(-1)
    
                # Defensive: if model returned S+Q preds, keep only last Q
                if pred_q.numel() != y_q.numel():
                    if pred_q.numel() >= y_q.numel():
                        pred_q = pred_q[-y_q.numel():]
                    else:
                        raise RuntimeError(
                            f"Prediction length mismatch: pred={pred_q.numel()}, target={y_q.numel()}"
                        )
    
                loss = F.mse_loss(pred_q, y_q)
                loss.backward()
    
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(torch_model.parameters(), float(clip_grad_norm))
    
                optimizer.step()
    
                if show_progress and (step_counter % log_every == 0 or step_counter == total_steps):
                    logger.info(
                        f"[TuningManager] TabDPT reg FT: step {step_counter}/{total_steps} | loss={loss.item():.6f}"
                    )
    
        logger.info("[TuningManager] TabDPT regression fine-tuning complete")
        try:
            torch_model.eval()
        except Exception:
            pass
    
        # In case wrapper holds estimator and estimator holds torch module
        try:
            if isinstance(inner_holder, torch.nn.Module):
                inner_holder.eval()
            elif hasattr(inner_holder, "model") and isinstance(inner_holder.model, torch.nn.Module):
                inner_holder.model.eval()
        except Exception:
            pass
    
        return model



    def _finetune_mitra_regression_turn_by_turn(self, model, X_train, y_train, params: dict):
        """
        - Calls model.fit(...) FIRST to reuse the wrapper's exact preprocessing behavior:
          * converts categorical/object columns to numeric codes
          * normalizes y to [0,1] via min-max and stores y_min/y_max
          * stores X_train/y_train caches used by predict()
        - Uses Tab2D forward signature (including padding_obs_query__ kwarg).
        - Shape-safe MSE without broadcasting issues.
        - Leaves model in eval mode.
        """
        import torch
        import numpy as np
        import logging
        from torch.optim import AdamW
        import torch.nn.functional as F
    
        logger = logging.getLogger(__name__)
        logger.info("[TuningManager] Starting Mitra regression fine-tuning (turn-by-turn)")
    
        params = params or {}
        device_str = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
    
        epochs = int(params.get("epochs", 5))
        steps_per_epoch = int(params.get("steps_per_epoch", 100))
        context_size = int(params.get("context_size", params.get("support_size", 512)))
        query_size = int(params.get("query_size", 128))
    
        lr = float(params.get("lr", params.get("learning_rate", 1e-5)))
        weight_decay = float(params.get("weight_decay", 0.0))
        clip_grad_norm = params.get("clip_grad_norm", None)
        show_progress = bool(params.get("show_progress", True))
        seed = params.get("seed", None)
    
        if seed is not None:
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
    
        # -------------------------------------------------------------------------
        # Resolve underlying torch module (wrapper.model may be torch module, or estimator.model)
        # -------------------------------------------------------------------------
        def _resolve_torch_module(wrapper):
            est = getattr(wrapper, "model", None)
            est2 = getattr(wrapper, "model_", None)
    
            if isinstance(est, torch.nn.Module):
                return est
            if est is not None and hasattr(est, "model") and isinstance(est.model, torch.nn.Module):
                return est.model
            if isinstance(est2, torch.nn.Module):
                return est2
            if est2 is not None and hasattr(est2, "model") and isinstance(est2.model, torch.nn.Module):
                return est2.model
            return None
    
        torch_model = _resolve_torch_module(model)
        if torch_model is None:
            raise AttributeError(
                "MitraRegressorWrapper: could not locate underlying torch module. "
                "Expected either (a) `wrapper.model` to be torch.nn.Module, or "
                "(b) `wrapper.model`/`wrapper.model_` to be an estimator with `.model` torch.nn.Module."
            )

        try:
            # Ensure fit doesn't accidentally run with dropout enabled
            try:
                torch_model.eval()
            except Exception:
                pass
            model.fit(X_train, y_train)
        except Exception as e:
            logger.exception("[TuningManager] model.fit(...) failed before Mitra finetune.")
            raise
    
        X_np = getattr(model, "X_train", None)
        y_norm = getattr(model, "y_train", None)
        if X_np is None or y_norm is None:
            raise RuntimeError(
                "Expected Mitra wrapper to expose caches `X_train` and `y_train` after fit()."
            )
    
        X_np = np.asarray(X_np, dtype=np.float32)
        y_norm = np.asarray(y_norm, dtype=np.float32).reshape(-1)
    
        n = len(X_np)
        if n < 2:
            raise ValueError("Need at least 2 training rows for episodic finetuning.")
    
        if n < (context_size + query_size):
            context_size = max(4, min(context_size, n // 2))
            query_size = max(1, min(query_size, n - context_size))
            logger.warning(f"[TuningManager] Shrunk episode sizes: context={context_size}, query={query_size}")
    
        torch_model.to(device)
        torch_model.train()
    
        optimizer = AdamW(torch_model.parameters(), lr=lr, weight_decay=weight_decay)
    
        def _to_tensor(x: np.ndarray, dtype=torch.float32):
            return torch.as_tensor(x, dtype=dtype, device=device)
    
        total_steps = max(1, epochs * steps_per_epoch)
        step_counter = 0
        log_every = max(1, total_steps // 20)
    
        for _ep in range(epochs):
            for _ in range(steps_per_epoch):
                step_counter += 1
    
                idx = np.random.permutation(n)
                s_idx = idx[:context_size]
                q_idx = idx[context_size: context_size + query_size]
    
                X_support = X_np[s_idx]
                y_support = y_norm[s_idx]
                X_query = X_np[q_idx]
                y_query = y_norm[q_idx]
    
                x_support_t = _to_tensor(X_support).unsqueeze(0)   # (1, S, F)
                y_support_t = _to_tensor(y_support).unsqueeze(0)   # (1, S)
                x_query_t = _to_tensor(X_query).unsqueeze(0)       # (1, Q, F)
    
                b, n_s, f = x_support_t.shape
                n_q = x_query_t.shape[1]
    
                # In repo predict(), these are all-false masks (no padding)
                padding_features = torch.zeros(b, f, dtype=torch.bool, device=device)
                padding_obs_support = torch.zeros(b, n_s, dtype=torch.bool, device=device)
                padding_obs_query = torch.zeros(b, n_q, dtype=torch.bool, device=device)
    
                optimizer.zero_grad(set_to_none=True)
    
                pred = torch_model(
                    x_support=x_support_t,
                    y_support=y_support_t,
                    x_query=x_query_t,
                    padding_features=padding_features,
                    padding_obs_support=padding_obs_support,
                    padding_obs_query__=padding_obs_query,  # <-- double underscore matches Tab2D.forward
                )
    
                # Shape-safe: pred -> (Q,)
                pred_q = pred
                if pred_q.dim() == 3:
                    pred_q = pred_q.squeeze(0)
                if pred_q.dim() == 2 and pred_q.size(-1) == 1:
                    pred_q = pred_q.squeeze(-1)
                if pred_q.dim() == 2 and pred_q.size(0) == 1:
                    pred_q = pred_q.squeeze(0)
                pred_q = pred_q.reshape(-1)
    
                y_q = _to_tensor(y_query).reshape(-1)
    
                # Defensive: if pred returns (S+Q,) or something odd, take last Q
                if pred_q.numel() != y_q.numel():
                    if pred_q.numel() >= y_q.numel():
                        pred_q = pred_q[-y_q.numel():]
                    else:
                        raise RuntimeError(
                            f"Prediction length mismatch: pred={pred_q.numel()}, target={y_q.numel()}"
                        )
    
                loss = F.mse_loss(pred_q, y_q)
                loss.backward()
    
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(torch_model.parameters(), float(clip_grad_norm))
    
                optimizer.step()
    
                if show_progress and (step_counter % log_every == 0 or step_counter == total_steps):
                    logger.info(
                        f"[TuningManager] Mitra reg FT: step {step_counter}/{total_steps} | loss={loss.item():.6f}"
                    )
    
        torch_model.eval()
        logger.info("[TuningManager] Mitra regression fine-tuning complete")

        return model



    def _finetune_tabpfn_regression_turn_by_turn(self, model, X_train, y_train, params: dict):
        import torch
        import numpy as np
        import logging
        from torch.optim import AdamW
    
        # TabPFN helper used in predict() to align borders
        from tabtune.models.tabpfn.utils import translate_probs_across_borders
    
        logger = logging.getLogger(__name__)
        logger.info("[TuningManager] Starting TabPFN regression fine-tuning (turn-by-turn)")
    
        params = params or {}
        device_str = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
    
        epochs = int(params.get("epochs", 5))
        steps_per_epoch = int(params.get("steps_per_epoch", 100))
        context_size = int(params.get("context_size", params.get("support_size", 512)))
        query_size = int(params.get("query_size", 128))
    
        lr = float(params.get("lr", params.get("learning_rate", 1e-5)))
        weight_decay = float(params.get("weight_decay", 0.0))
        clip_grad_norm = params.get("clip_grad_norm", None)
        show_progress = bool(params.get("show_progress", True))
        seed = params.get("seed", None)
    
        if seed is not None:
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
    
        # ----------------------------
        # Data
        # ----------------------------
        X_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
        y_np = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
        y_np = np.asarray(y_np).reshape(-1)
    
        n = len(X_np)
        if n < (context_size + query_size):
            context_size = max(4, min(context_size, n // 2))
            query_size = max(1, min(query_size, n - context_size))
            logger.warning(f"[TuningManager] Shrunk episode sizes: context={context_size}, query={query_size}")
    
        # -------------------------------------------------------------------------
        # ✅ CRITICAL: Ensure TabPFN internals exist (creates interface_config_, model_, device_, etc.)
        # -------------------------------------------------------------------------
        if not hasattr(model, "interface_config_"):
            model._initialize_model_variables()
    
        optimizer = None
        total_steps = epochs * steps_per_epoch
        step_counter = 0
    
        for _ep in range(epochs):
            for _ in range(steps_per_epoch):
                step_counter += 1
    
                idx = np.random.permutation(n)
                s_idx = idx[:context_size]
                q_idx = idx[context_size: context_size + query_size]
    
                X_support = X_np[s_idx]
                y_support = y_np[s_idx]
                X_query = X_np[q_idx]
                y_query = y_np[q_idx]
    
                # Episode dataset = support + query
                X_episode = np.concatenate([X_support, X_query], axis=0)
                y_episode = np.concatenate([y_support, y_query], axis=0)
    
                def _split_fn(X_full, y_full):
                    S = context_size
                    return X_full[:S], X_full[S:], y_full[:S], y_full[S:]
    
                ds = model.get_preprocessed_datasets(
                    X_raw=X_episode,
                    y_raw=y_episode,
                    split_fn=_split_fn,
                    max_data_size=None,
                )
    
                # -----------------------------------------------------------------
                # ✅ Your version returns 10 values (regression)
                # -----------------------------------------------------------------
                (
                    X_trains_pre,
                    X_tests_pre,
                    y_trains_pre,
                    y_test_std,
                    cat_ixs,
                    confs,
                    raw_space_bardist,
                    znorm_space_bardist,
                    _x_test_raw,
                    _y_test_raw,
                ) = ds[0]

                y_trains_pre = [yt.unsqueeze(-1) if (isinstance(yt, torch.Tensor) and yt.dim() == 1) else yt for yt in y_trains_pre]
    
                # -----------------------------------------------------------------
                # ✅ Ensure executor_ exists:
                # fit_from_preprocessed() creates executor_ with fit_mode="batched"
                # no_refit=True prevents reinitializing weights.
                # -----------------------------------------------------------------

                
                
                cat_ix_batched = cat_ixs
                if (
                    isinstance(cat_ixs, list)
                    and len(cat_ixs) > 0
                    and isinstance(cat_ixs[0], list)
                    and (len(cat_ixs[0]) == 0 or isinstance(cat_ixs[0][0], int))
                ):
                    # cat_ixs is List[List[int]] -> make it List[List[List[int]]] with batch_size=1
                    cat_ix_batched = [[ci] for ci in cat_ixs]
                
                model.fit_from_preprocessed(
                    X_preprocessed=X_trains_pre,
                    y_preprocessed=y_trains_pre,
                    cat_ix=cat_ix_batched,
                    configs=confs,
                    no_refit=True,
                )

    
                # Underlying torch module (created by _initialize_model_variables)
                torch_model = model.model_
                torch_model.to(device)
                torch_model.train()
    
                if optimizer is None:
                    optimizer = AdamW(torch_model.parameters(), lr=lr, weight_decay=weight_decay)
    
                optimizer.zero_grad(set_to_none=True)
    
                # Query targets are standardized already (TabPFN uses z-norm space for loss)
                yq = y_test_std
                if isinstance(yq, torch.Tensor) and yq.dim() == 2 and yq.size(-1) == 1:
                    yq = yq.squeeze(-1)
                    yq = yq.to(device).reshape(-1)

    
                # -----------------------------------------------------------------
                # ✅ Use TabPFN's forward() for batched engine + gradients
                # This returns:
                #   averaged_logits (unused here),
                #   outputs: list[tensor] per estimator (these are PROBS, not log-probs),
                #   borders: list[np.ndarray] per estimator
                # -----------------------------------------------------------------
                _avg, outputs, borders = model.forward(X_tests_pre, use_inference_mode=False)
    
                std_borders = model.znorm_space_bardist_.borders.to(device)
    
                # Align probs to standard borders + average across estimators
                transformed_probs = []
                for probs, b in zip(outputs, borders):
                    # probs shape commonly: (N_tokens, 1, n_bars) or (N_tokens, n_bars)
                    p = probs
                    if p.dim() == 3:
                        p = p.squeeze(1)  # (N_tokens, n_bars)
    
                    # Translate probs from per-config borders -> std borders
                    p = translate_probs_across_borders(
                        p,
                        frm=torch.as_tensor(b, device=device),
                        to=std_borders,
                    )
                    transformed_probs.append(p)
    
                # Average across estimators
                probs_mean = torch.stack(transformed_probs, dim=0).mean(dim=0)  # (N_tokens, n_bars)
    
                # Take last Q tokens as query tokens (same convention used earlier)
                q_probs = probs_mean[-len(yq):]  # (Q, n_bars)
    
                # Convert to log-probs for bardist NLL
                q_log_probs = (q_probs + 1e-12).log()
    
                crit = model.znorm_space_bardist_.to(device)
                loss = crit(q_log_probs, yq).mean()
    
                loss.backward()
    
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(torch_model.parameters(), float(clip_grad_norm))
    
                optimizer.step()
    
                if show_progress and (step_counter % max(1, total_steps // 20) == 0):
                    logger.info(
                        f"[TuningManager] TabPFN reg FT: step {step_counter}/{total_steps} | loss={float(loss.item()):.6f}"
                    )
    
        torch_model.eval()
        logger.info("[TuningManager] TabPFN regression fine-tuning complete")
    
        # -------------------------------------------------------------------------
        # Post-finetune: fit normally so predict/eval works with standard engine
        # (Note: this will reset fit_mode from 'batched' to normal, as per TabPFN code.)
        # -------------------------------------------------------------------------
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logger.warning(f"[TuningManager] Post-finetune model.fit failed (predict may break): {e}")
    
        return model










    

