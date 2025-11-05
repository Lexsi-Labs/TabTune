"""
Meta-learning trainer for OrionBix with k-NN support selection.
"""

from __future__ import annotations

import os
import timeit
import warnings
import functools
from contextlib import nullcontext

import math
import numpy as np

import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm
import wandb

from orion_bix import OrionBix
from orion_bix.prior.meta_dataset import MetaLearningDataset
from orion_bix.prior.episode_generator import EpisodeGenerator
from orion_bix.prior.dataset import PriorDataset
from orion_bix.prior.genload import LoadPriorDataset
from orion_bix.train.optim import get_scheduler
from orion_bix.train.train_config import build_parser

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


class Timer:
    """Context manager for timing code execution."""
    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False


def ddp_cleanup(func):
    """Decorator to clean up DDP process group after method execution."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.ddp:
                destroy_process_group()
    return wrapper


class MetaLearningTrainer:
    """Meta-learning trainer for OrionBix with k-NN support selection."""

    def __init__(self, config):
        self.config = config
        self.configure_ddp()
        self.configure_wandb()
        self.build_model()
        self.configure_meta_learning()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()
        # --- NEW: EMA pour lisser l'ETA (temps moyen par step) ---
        self._ema_step_sec = None
        self._ema_alpha = 0.10  # réactivité ~10 steps

    def configure_ddp(self):
        """Set up distributed training and system configuration."""
        # Setup distributed training
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

            # Adjust batch size for distributed training
            original_batch_size = self.config.batch_size
            self.config.batch_size = math.ceil(original_batch_size / self.ddp_world_size)

            if self.master_process:
                print(f"DDP training with {self.ddp_world_size} processes")
                if original_batch_size % self.ddp_world_size == 0:
                    print(f"Per-GPU batch size: {self.config.batch_size}")
                else:
                    print(
                        f"Original batch size ({original_batch_size}) cannot be divided by world size ({self.ddp_world_size}).\n"
                        f"Use ceiling division for equal per-GPU batch size: {self.config.batch_size}.\n"
                        f"Effective batch size is {self.config.batch_size * self.ddp_world_size}.\n"
                    )
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            print("No DDP training")
        
        self.curr_step = 0  # Initialize current step for training

        # Set random seeds
        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.np_seed + seed_offset)
        torch.manual_seed(self.config.torch_seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def configure_wandb(self):
        """Set up Weights & Biases logging."""
        if self.config.wandb_log and self.master_process:
            id_path = os.path.join(self.config.checkpoint_dir, "wand_id.txt")
            if self.config.wandb_id is None:
                if os.path.exists(id_path):
                    with open(id_path, "r") as f:
                        self.config.wandb_id = f.read().strip()

            self.wandb_run = wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                id=self.config.wandb_id,
                config=self.config,
                resume="allow",
                mode=self.config.wandb_mode,
            )

            with open(id_path, "w") as f:
                f.write(self.wandb_run.id)
        else:
            self.wandb_run = None

    def build_model(self):
        """Build and initialize the OrionBix model."""
        self.model_config = {
            "max_classes": self.config.max_classes,
            "embed_dim": self.config.embed_dim,
            "col_num_blocks": self.config.col_num_blocks,
            "col_nhead": self.config.col_nhead,
            "col_num_inds": self.config.col_num_inds,
            "row_num_blocks": self.config.row_num_blocks,
            "row_nhead": self.config.row_nhead,
            "row_num_cls": self.config.row_num_cls,
            "row_rope_base": self.config.row_rope_base,
            "icl_num_blocks": self.config.icl_num_blocks,
            "icl_nhead": self.config.icl_nhead,
            "ff_factor": self.config.ff_factor,
            "dropout": self.config.dropout,
            "activation": self.config.activation,
            "norm_first": self.config.norm_first,
            "col_attention_type": self.config.col_attention_type,
            "col_feature_map": self.config.col_feature_map,
            "row_attention_type": self.config.row_attention_type,
            "row_feature_map": self.config.row_feature_map,
            "icl_attention_type": self.config.icl_attention_type,
            "icl_feature_map": self.config.icl_feature_map,
            "debug": self.config.debug,
        }

        model = OrionBix(**self.model_config)
        model.to(device=self.config.device)
        print(self.model_config)

        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {num_params} parameters.")

        # Freeze model components if requested
        if self.config.freeze_col:
            model.col_embedder.eval()
            for param in model.col_embedder.parameters():
                param.requires_grad = False

        if self.config.freeze_row:
            if self.config.row_attention_type in ["standard","linear"]:
                model.row_interactor.eval()
                for param in model.row_interactor.parameters():
                    param.requires_grad = False
            else:  # bi_axial
                model.bi_axial_attention.eval()
                for param in model.bi_axial_attention.parameters():
                    param.requires_grad = False

        if self.config.freeze_icl:
            model.icl_predictor.eval()
            for param in model.icl_predictor.parameters():
                param.requires_grad = False
                
        # Compile model if requested
        if self.config.model_compile:
            model = torch.compile(model, dynamic=True)
            #model = torch.compile(model, mode="reduce-overhead")
            if self.master_process:
                print("Model compiled successfully.")
        
        # Enable gradient checkpointing to save memory
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        if self.ddp:
            # Check if we need to find unused parameters
            find_unused_parameters = False
            
            # If using bi_axial attention, row_interactor is None, so we need to find unused parameters
            if self.config.row_attention_type == "bi_axial":
                find_unused_parameters = True
            
            # If freezing components, we might also need to find unused parameters
            if self.config.freeze_col or self.config.freeze_row or self.config.freeze_icl:
                find_unused_parameters = True
            
            # Force synchronization before DDP
            torch.distributed.barrier()
            
            # Initialize DDP
            self.model = DDP(
                model, 
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=find_unused_parameters,
                broadcast_buffers=False
            )
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model
        
    def configure_meta_learning(self):
        """Sets up meta-learning components including episode generation and k-NN support selection."""
        # Create episode generator for meta-learning
        self.episode_generator = EpisodeGenerator(
            support_size=self.config.support_size,
            query_size=self.config.query_size,
            k_neighbors=self.config.k_neighbors,
            similarity_metric=self.config.similarity_metric,
            feature_normalization=self.config.feature_normalization,
            num_episodes_per_dataset=self.config.num_episodes_per_dataset,
            device=self.config.device,
            diversity_factor=self.config.diversity_factor,
            min_dataset_size=self.config.min_dataset_size,
            episode_method=self.config.episode_method,
        )

        # Use standard OrionBix logic for prior data
        if self.config.prior_dir is None:
            # Generate prior data on the fly
            dataset = PriorDataset(
                batch_size=self.config.batch_size,
                batch_size_per_gp=self.config.batch_size_per_gp,
                min_features=self.config.min_features,
                max_features=self.config.max_features,
                max_classes=self.config.max_classes,
                min_seq_len=self.config.min_seq_len,
                max_seq_len=self.config.max_seq_len,
                log_seq_len=self.config.log_seq_len,
                seq_len_per_gp=self.config.seq_len_per_gp,
                min_train_size=self.config.min_train_size,
                max_train_size=self.config.max_train_size,
                replay_small=self.config.replay_small,
                prior_type=self.config.prior_type,
                device=self.config.prior_device,
                n_jobs=1,  # Set to 1 to avoid nested parallelism during DDP
            )
        else:
            # Load pre-generated prior data from disk
            dataset = LoadPriorDataset(
                data_dir=self.config.prior_dir,
                batch_size=self.config.batch_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                start_from=self.config.load_prior_start,
                delete_after_load=self.config.delete_after_load,
                device=self.config.prior_device,
            )

        if self.master_process:
            print(dataset)

        # Create meta-learning dataset wrapper with memory-efficient chunking
        self.meta_dataset = MetaLearningDataset(
            base_prior=dataset,
            episode_generator=self.episode_generator,
            episodes_per_yield=self.config.batch_size * self.config.num_episodes_per_dataset,  # 2048 * 20 = 40,960 episodes
        )

        # Create dataloader with single worker to avoid memory issues
        self.dataloader = DataLoader(
            self.meta_dataset,
            batch_size=None,  # No additional batching
            shuffle=False,
            num_workers=0,  # Single process to avoid memory sharing issues
            pin_memory=False,  # Disable pin memory to reduce memory usage
        )

    def configure_optimizer(self):
        """Configure optimizer and scheduler."""
        # Use original learning rate (no reduction needed)
        self.optimizer = optim.AdamW(
            params=self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)

    def configure_amp(self):
        """Configure automatic mixed precision (AMP) for training."""
        self.amp = self.config.amp and "cuda" in self.config.device
        self.scaler = torch.GradScaler("cuda", enabled=self.amp)
        if self.amp:
            if self.master_process:
                print(f"Automatic Mixed Precision is enabled.")
            self.amp_ctx = torch.autocast(
                device_type="cuda", dtype=torch.float16 if self.config.dtype == "float16" else torch.float32
            )
        else:
            self.amp_ctx = nullcontext()

    def get_latest_checkpoint(self):
        """Returns the latest checkpoint from `checkpoint_dir`"""
        ckpt_dir = self.config.checkpoint_dir

        if not os.path.isdir(ckpt_dir):
            return None

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]

        if not checkpoints:
            return None

        # Sort the checkpoint files by step number and get the latest
        try:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))[-1]
            checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
            return checkpoint_path
        except Exception as e:
            print(f"Error parsing checkpoint filenames: {e}")
            return None

    def load_checkpoint(self):        
        """Load model and training state from checkpoint.

        First checks if `checkpoint_path` is directly specified. If not, attempts to find
        the latest checkpoint in the checkpoint directory.
        """

        checkpoint_path = None
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        elif hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return
        
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)

        # Load model state
        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model state")
                
        self.raw_model.load_state_dict(checkpoint["state_dict"])

        
        
        # Optionally load optimizer and scheduler state
        if self.config.only_load_model:
            print("Only loading model weights")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.curr_step = checkpoint["curr_step"]
            print(f"Resuming training at step {self.curr_step}")
        

    def save_checkpoint(self, name: str):
        """Save model and training state to checkpoint."""

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        checkpoint = {
            "config": self.model_config,
            "state_dict": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "curr_step": self.curr_step,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def get_memory_info(self):
        """Get current GPU memory usage information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved
            }
        return {}

    
    '''
    def manage_checkpoint(self):
        """Manage checkpoint storage by removing old checkpoints."""
        if not self.master_process or self.config.max_checkpoints <= 0:
            return

        checkpoints = []
        for filename in os.listdir(self.config.checkpoint_dir):
            if filename.endswith(".ckpt"):
                checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
                checkpoints.append((checkpoint_path, os.path.getmtime(checkpoint_path)))

        if len(checkpoints) > self.config.max_checkpoints:
            checkpoints.sort(key=lambda x: x[1])
            for checkpoint_path, _ in checkpoints[:-self.config.max_checkpoints]:
                os.remove(checkpoint_path)
    '''           
    def manage_checkpoint(self):
        """
        Manages the number of temporary checkpoints by deleting the oldest ones
        if the count exceeds `max_checkpoints`. Permanent checkpoints are ignored.
        """
        ckpt_dir = self.config.checkpoint_dir
        limit = self.config.max_checkpoints

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.split("-")[1].split(".")[0])
                # Consider a checkpoint temporary if its step is not divisible by save_perm_every
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except:
                continue  # Ignore files that don't match the format

        # Sort temporary checkpoints by step number (ascending)
        temp_checkpoints.sort(key=lambda x: x[0])

        # Remove oldest temporary checkpoints if limit is exceeded
        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            for step, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    print(f"Error removing checkpoint {ckpt_path}: {e}")

    def run_episode(self, episode_data):
        """Process a single episode with k-NN support selection."""
        support_X = episode_data['support_X'].to(self.config.device)
        support_y = episode_data['support_y'].to(self.config.device)
        query_X = episode_data['query_X'].to(self.config.device)
        query_y = episode_data['query_y'].to(self.config.device)
        d = episode_data['d'].to(self.config.device)
        seq_len = episode_data['seq_len']
        support_size = episode_data['support_size']
    
        # Combine support and query sets for processing
        total_X = torch.cat([support_X, query_X], dim=0).unsqueeze(0)
        total_y = torch.cat([support_y, query_y], dim=0).unsqueeze(0)
        total_d = d.unsqueeze(0)
    
        # Forward pass through model
        with self.amp_ctx:
            # The model expects y_train to be the support labels and returns predictions for query samples
            pred = self.model(total_X, support_y.unsqueeze(0), total_d)
            # pred already contains predictions for query samples
            query_pred = pred[0]  # Remove batch dimension
            query_true = query_y.long()
            
            loss = F.cross_entropy(query_pred, query_true)
            accuracy = (query_pred.argmax(dim=1) == query_true).float().mean()
    
        result = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "query_size": query_X.shape[0],
            "support_size": support_size,
            "loss_tensor": loss,  # Return the actual loss tensor for gradient accumulation
        }
    
        #torch.cuda.empty_cache()
        # OPTIMIZED MEMORY CLEANUP: Only delete what's not needed
        del support_X, support_y, query_X, query_y, d, total_X, total_y, total_d, pred, query_pred, query_true
    
        return result

    def run_batch(self, episodes):
        """Process episodes using standard OrionBix micro-batch logic."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
    
        batch_results = {
            "loss": 0.0,
            "accuracy": 0.0,
        }
    
        num_episodes = len(episodes)
        if num_episodes == 0:
            return batch_results
    
        # Standard OrionBix micro-batch logic
        episodes_per_dataset = self.config.num_episodes_per_dataset
        datasets_per_batch = self.config.batch_size
        datasets_per_micro_batch = self.config.micro_batch_size
        
        # Calculate episodes per micro-batch
        episodes_per_micro_batch = datasets_per_micro_batch * episodes_per_dataset
        
        num_micro_batches = math.ceil(num_episodes / episodes_per_micro_batch)
        successful_episodes = 0
        total_loss_tensor = 0.0  # Accumulate loss tensors, not scalar values
    
        if self.master_process:
            print(f"Processing {num_episodes} episodes in {num_micro_batches} micro-batches")
            print(f"Each micro-batch: {episodes_per_micro_batch} episodes ({datasets_per_micro_batch} datasets × {episodes_per_dataset} episodes)")
    
        # Process each micro-batch
        for micro_batch_idx in range(num_micro_batches):
            #torch.cuda.empty_cache()
            
            start_idx = micro_batch_idx * episodes_per_micro_batch
            end_idx = min(start_idx + episodes_per_micro_batch, num_episodes)
            micro_episodes = episodes[start_idx:end_idx]
            
            micro_batch_successful = 0
            micro_batch_loss_tensor = 0.0  # Accumulate loss tensors for this micro-batch
            
            # Process each episode in the micro-batch
            for episode_idx, episode_data in enumerate(micro_episodes):
                # Phase 0: local compute (no collectives yet)
                err  = torch.zeros(1, device=self.config.device, dtype=torch.int32)
                drop = torch.zeros(1, device=self.config.device, dtype=torch.int32)
                loss_tensor = None
                try:
                    episode_results = self.run_episode(episode_data)
                    # local “drop” checks
                    if episode_results["loss"] > 3.0:
                        drop.fill_(1)
                    else:
                        lt = episode_results["loss_tensor"]
                        if torch.isnan(lt) or torch.isinf(lt):
                            drop.fill_(1)
                        else:
                            loss_tensor = lt
                except Exception as e:
                    if self.master_process:
                        print(f"Warning: Error in episode {episode_idx}: {e}")
                    err.fill_(1)

                # Phase 1: if any rank errored, all skip uniformly
                if self.ddp:
                    torch.distributed.all_reduce(err, op=torch.distributed.ReduceOp.MAX)
                if err.item() > 0:
                    continue  # every rank skips
                # Phase 2: if any rank wants to drop (high loss / NaN), all skip uniformly
                if self.ddp:
                    torch.distributed.all_reduce(drop, op=torch.distributed.ReduceOp.MAX)
                if drop.item() > 0:
                    continue  # every rank skips
            
                # Phase 3: Accumulate scalar values for reporting
                batch_results["loss"]     += episode_results["loss"]
                batch_results["accuracy"] += episode_results["accuracy"]
                # Accumulate loss tensor for gradient computation
                micro_batch_loss_tensor   += episode_results["loss_tensor"]
                micro_batch_successful    += 1
                successful_episodes       += 1
                
            # Process micro-batch loss (accumulate gradients, don't update yet)
            if micro_batch_successful > 0:
                # Average the loss for this micro-batch
                avg_micro_batch_loss = micro_batch_loss_tensor / micro_batch_successful

                if not (torch.isnan(avg_micro_batch_loss) or torch.isinf(avg_micro_batch_loss)):
                    self.scaler.scale(avg_micro_batch_loss).backward()
                    total_loss_tensor += avg_micro_batch_loss

                    if self.master_process:
                        print(f"Micro-batch {micro_batch_idx + 1}/{num_micro_batches}: "
                            f"Processed {micro_batch_successful} episodes, "
                            f"Avg loss: {avg_micro_batch_loss.item():.4f}")
    
        # After processing all micro-batches, update the model
        if successful_episodes > 0:
            try:
                # Apply gradient clipping if enabled
                if self.config.gradient_clipping > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
    
                # Step optimizer and update scaler (only once after all micro-batches)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                
                if self.master_process:
                    print(f"Updated model after processing all {num_micro_batches} micro-batches")
                    print(f"Total episodes processed: {successful_episodes}")
                    print(f"Average loss across all episodes: {total_loss_tensor.item() / num_micro_batches:.4f}")
                
            except torch.cuda.OutOfMemoryError:
                if self.master_process:
                    print(f"Warning: OOM error in final optimizer step. Skipping update.")
                #torch.cuda.empty_cache()
                self.optimizer.zero_grad(set_to_none=True)
    
        # Average results only for successful episodes
        if successful_episodes > 0:
            batch_results["loss"] /= successful_episodes
            batch_results["accuracy"] /= successful_episodes
    
        return batch_results
    
   

    @ddp_cleanup
    def train(self):
        """Main meta-learning training loop with optimized timing."""
        if self.master_process:
            step_progress = tqdm(range(self.curr_step, self.config.max_steps), desc="Step", leave=True)
        else:
            step_progress = range(self.curr_step, self.config.max_steps)

        dataloader = iter(self.dataloader)
        for step in step_progress:
            # OPTIMIZATION: Use time.time() instead of Timer for less overhead
            import time
            episode_start = time.time()
            episodes = next(dataloader)
            episode_time = time.time() - episode_start

            train_start = time.time()
            results = self.run_batch(episodes)
            train_time = time.time() - train_start
            

            # --- NEW: total step time (end-to-end), EMA, ETA, throughputs ---
            total_step_sec = episode_time + train_time  # inclut data + compute
            if self._ema_step_sec is None:
                self._ema_step_sec = total_step_sec
            else:
                self._ema_step_sec = self._ema_alpha * total_step_sec + (1.0 - self._ema_alpha) * self._ema_step_sec
            steps_left = self.config.max_steps - (step + 1)
            eta_sec = max(0.0, self._ema_step_sec * steps_left)
            # débits (garde-fous contre division par zéro)
            train_sps = (1.0 / train_time) if train_time > 0 else float("nan")
            total_sps = (1.0 / total_step_sec) if total_step_sec > 0 else float("nan")
            ###
            
            self.curr_step = step + 1
            if self.master_process:
                results.update({
                    "episode_time": episode_time, 
                    "train_time": train_time,
                    #"steps_per_sec": total_sps,        # end-to-end (épisode complet)
                    "steps_per_sec": 1.0 / total_step_sec if total_step_sec > 0 else 0.0,
                    "eta_sec": eta_sec,
                    "eta_hour": eta_sec / 3600.0,
                })

                step_progress.set_postfix(**{k: round(v, 3) if isinstance(v, float) else v for k, v in results.items()})

                # Save checkpoints
                is_temp_save = self.curr_step % self.config.save_temp_every == 0
                is_perm_save = self.curr_step % self.config.save_perm_every == 0

                if is_temp_save or is_perm_save:
                    ckpt_name = f"step-{self.curr_step}.ckpt"
                    self.save_checkpoint(name=ckpt_name)

                    if is_temp_save and not is_perm_save and self.config.max_checkpoints > 0:
                        self.manage_checkpoint()

            # Logging to Weights & Biases
            if self.wandb_run is not None:
                results["lr"] = self.scheduler.get_last_lr()[0]
                wandb.log(results, step=self.curr_step)


if __name__ == "__main__":
    parser = build_parser()
    config = parser.parse_args()

    try:
        # Set the start method for subprocesses to 'spawn'
        set_start_method("spawn")
    except RuntimeError:
        pass  # Ignore the error if the context has already been set

    trainer = MetaLearningTrainer(config)
    trainer.train()