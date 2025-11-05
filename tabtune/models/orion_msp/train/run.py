from __future__ import annotations

import os
import timeit
import functools
from contextlib import nullcontext
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message=".*nested tensors is in prototype stage.*", category=UserWarning)

from orion_msp.model.orion_msp import OrionMSP
from orion_msp.prior.dataset import PriorDataset
from orion_msp.prior.genload import LoadPriorDataset
from orion_msp.train.optim import get_scheduler
from orion_msp.train.train_config import build_parser


class Timer:
    def __enter__(self): self.start_time = timeit.default_timer(); return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.elapsed = timeit.default_timer() - self.start_time; return False


def ddp_cleanup(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.ddp:
                destroy_process_group()
    return wrapper


class Trainer:
    def __init__(self, config):
        self.config = config
        self.configure_ddp()
        self.configure_wandb()
        self.build_model()
        self.configure_prior()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()

    def configure_ddp(self):
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

            original_batch_size = self.config.batch_size
            self.config.batch_size = math.ceil(original_batch_size / self.ddp_world_size)
            if self.master_process:
                print(f"DDP: {self.ddp_world_size} processes, per-GPU batch size {self.config.batch_size}")
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            print("Single-process training")

        self.curr_step = 0
        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.np_seed + seed_offset)
        torch.manual_seed(self.config.torch_seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def configure_wandb(self):
        if self.config.wandb_log and self.master_process:
            import wandb
            os.makedirs(self.config.checkpoint_dir or ".", exist_ok=True)
            id_path = os.path.join(self.config.checkpoint_dir or ".", "wand_id.txt")
            if self.config.wandb_id is None and os.path.exists(id_path):
                with open(id_path, "r") as f:
                    self.config.wandb_id = f.read().strip()
            self.wandb = wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                id=self.config.wandb_id,
                config=self.config,
                resume="allow",
                mode=self.config.wandb_mode,
            )
            with open(id_path, "w") as f:
                f.write(self.wandb.id)
        else:
            self.wandb = None

    def build_model(self):
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
            "row_num_global": self.config.row_num_global,
            "row_scales": tuple(self.config.row_scales),
            "row_window": self.config.row_window,
            "row_num_random": self.config.row_num_random,
            "row_group_mode": self.config.row_group_mode,
            "perc_num_latents": self.config.perc_num_latents,
            "perc_layers": self.config.perc_layers,
        }

        model = OrionMSP(**self.model_config).to(self.config.device)

        if self.config.freeze_col:
            model.col_embedder.eval()
            for p in model.col_embedder.parameters(): p.requires_grad = False
        if self.config.freeze_row:
            model.row_interactor.eval()
            for p in model.row_interactor.parameters(): p.requires_grad = False
        if self.config.freeze_icl:
            model.icl_predictor.eval()
            for p in model.icl_predictor.parameters(): p.requires_grad = False

        if self.config.model_compile:
            model = torch.compile(model, dynamic=True)
            if self.master_process: print("Model compiled.")

        if self.ddp:
            find_unused = (self.config.row_group_mode != "pma")
            self.model = DDP(
                model, 
                device_ids=[self.ddp_local_rank], 
                broadcast_buffers=False,
                find_unused_parameters=find_unused,
            )
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

        if self.master_process:
            num_params = sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {num_params:,}")

    def configure_prior(self):
        if self.config.prior_dir is None:
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
                n_jobs=1,
            )
        else:
            dataset = LoadPriorDataset(
                data_dir=self.config.prior_dir,
                batch_size=self.config.batch_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                start_from=self.config.load_prior_start,
                delete_after_load=self.config.delete_after_load,
                device=self.config.prior_device,
            )

        if self.master_process: print(dataset)
        self.dataloader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=1,
            prefetch_factor=4,
            pin_memory=True if self.config.prior_device == "cpu" else False,
            pin_memory_device=self.config.device if self.config.prior_device == "cpu" else "",
        )

    def configure_optimizer(self):
        from torch import optim
        self.optimizer = optim.AdamW(self.raw_model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)

    def configure_amp(self):
        self.amp = self.config.amp and "cuda" in self.config.device
        self.scaler = torch.GradScaler("cuda", enabled=self.amp)
        self.amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16 if self.config.dtype == "float16" else torch.float32)
            if self.amp else nullcontext()
        )
        if self.amp and self.master_process:
            print("AMP enabled.")

    def get_latest_checkpoint(self):
        ckpt_dir = self.config.checkpoint_dir
        if not ckpt_dir or not os.path.isdir(ckpt_dir): return None
        cks = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
        if not cks: return None
        try:
            cks.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
            return os.path.join(ckpt_dir, cks[-1])
        except Exception:
            return None

    def load_checkpoint(self):
        path = self.config.checkpoint_path or self.get_latest_checkpoint()
        if path is None or not os.path.exists(path):
            print("No checkpoint found; starting fresh.")
            return
        print(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.config.device, weights_only=True)
        self.raw_model.load_state_dict(ckpt["state_dict"])
        if not self.config.only_load_model:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.curr_step = ckpt["curr_step"]
            print(f"Resumed at step {self.curr_step}")

    def save_checkpoint(self, name: str):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, name)
        ckpt = {
            "config": self.model_config,
            "state_dict": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "curr_step": self.curr_step,
        }
        torch.save(ckpt, path)

    def manage_checkpoint(self):
        ckpt_dir = self.config.checkpoint_dir
        limit = self.config.max_checkpoints
        cks = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
        temp = []
        for fn in cks:
            try:
                step = int(fn.split("-")[1].split(".")[0])
                if step % self.config.save_perm_every != 0:
                    temp.append((step, fn))
            except:
                pass
        temp.sort(key=lambda x: x[0])
        to_delete = len(temp) - limit
        for _, fn in temp[:max(0, to_delete)]:
            try: os.remove(os.path.join(ckpt_dir, fn))
            except Exception as e: print(f"Failed to remove {fn}: {e}")

    @ddp_cleanup
    def train(self):
        iterator = iter(self.dataloader)
        if self.master_process:
            prog = tqdm(range(self.curr_step, self.config.max_steps), desc="Step", leave=True)
        else:
            prog = range(self.curr_step, self.config.max_steps)

        for step in prog:
            with Timer() as prior_timer:
                batch = next(iterator)
            with Timer() as train_timer:
                results = self.run_batch(batch)
            self.curr_step = step + 1

            if self.master_process:
                results.update({"prior_time": prior_timer.elapsed, "train_time": train_timer.elapsed})
                if isinstance(prog, tqdm):
                    prog.set_postfix(**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in results.items()})
                if self.curr_step % self.config.save_temp_every == 0 or self.curr_step % self.config.save_perm_every == 0:
                    self.save_checkpoint(f"step-{self.curr_step}.ckpt")
                    if self.curr_step % self.config.save_temp_every == 0 and self.curr_step % self.config.save_perm_every != 0:
                        if self.config.max_checkpoints > 0:
                            self.manage_checkpoint()

            if self.wandb is not None:
                results["lr"] = self.scheduler.get_last_lr()[0]
                self.wandb.log(results, step=self.curr_step)

    def validate_micro_batch(self, micro_seq_len, micro_train_size):
        if len(torch.unique(micro_seq_len)) > 1:
            raise ValueError("All datasets in the micro batch must have the same sequence length.")
        if len(torch.unique(micro_train_size)) > 1:
            raise ValueError("All datasets in the micro batch must have the same training size.")
        return micro_seq_len[0].item(), micro_train_size[0].item()

    def align_micro_batch(self, micro_X, micro_y, micro_d, seq_len):
        if micro_X.shape[1] > seq_len: micro_X = micro_X[:, :seq_len]
        if micro_y.shape[1] > seq_len: micro_y = micro_y[:, :seq_len]
        max_features = micro_d.max().item()
        if micro_X.shape[-1] > max_features: micro_X = micro_X[..., :max_features]
        return micro_X, micro_y

    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches):
        """
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        micro_X = micro_X.to(self.config.device)
        micro_y = micro_y.to(self.config.device)
        micro_d = micro_d.to(self.config.device)

        y_train = micro_y[:, :train_size]
        y_test = micro_y[:, train_size:]

        if self.ddp:
            self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1

        with self.amp_ctx:
            pred = self.model(micro_X, y_train, micro_d)  # (B, test_size, C)
            pred = pred.flatten(end_dim=-2)
            true = y_test.long().flatten()
            loss = F.cross_entropy(pred, true)

        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            micro_results = {"ce": scaled_loss.item(), "accuracy": (pred.argmax(dim=1) == true).float().mean().item() / num_micro_batches}
        return micro_results
        """
    
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)
        
        micro_X = micro_X.to(self.config.device)
        micro_y = micro_y.to(self.config.device)
        micro_d = micro_d.to(self.config.device)

        y_train = micro_y[:, :train_size]
        y_test  = micro_y[:, train_size:]

        if y_test.numel() == 0:
            return {"ce": 0.0, "accuracy": 0.0}

        if self.ddp:
            self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1

        with self.amp_ctx:
            logits = self.model(micro_X, y_train, micro_d)  # (B, Ttest, C)
            B, T, C = logits.shape
            pred  = logits.reshape(-1, C)
            true  = y_test.reshape(-1).long()

            valid = (true >= 0) & (true < C)
            if not torch.all(valid):
                true = true[valid]
                pred = pred[valid]
            if true.numel() == 0:
                return {"ce": 0.0, "accuracy": 0.0}

            loss = F.cross_entropy(pred, true)

        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
            
        
        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            micro_results = {
                "ce": scaled_loss.item(),
                "accuracy": (pred.argmax(dim=1) == true).float().mean().item() / num_micro_batches,
            }
        return micro_results


    def run_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]
        
        """
        num_micro_batches = math.ceil(self.config.batch_size / self.config.micro_batch_size)
        micro_batches = [torch.split(t, self.config.micro_batch_size, dim=0) for t in batch]
        micro_batches = list(zip(*micro_batches))
        """
        splits = [torch.split(t, self.config.micro_batch_size, dim=0) for t in batch]
        all_micros = list(zip(*splits))

        valid_micros = []
        for mb in all_micros:
            _, _, _, micro_seq_len, micro_train_size = mb
            seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
            if seq_len > train_size:
                valid_micros.append(mb)

        num_micro_batches = len(valid_micros)
        if num_micro_batches == 0:
            self.scheduler.step()
            return {"ce": 0.0, "accuracy": 0.0}

        micro_batches = valid_micros
        
        results = {"ce": 0.0, "accuracy": 0.0}
        failed = 0
        for i, micro in enumerate(micro_batches):
            try:
                res = self.run_micro_batch(micro, i, num_micro_batches)
                for k, v in res.items(): results[k] += v
            except torch.cuda.OutOfMemoryError:
                print(f"OOM in micro-batch {i+1}/{num_micro_batches} at step {self.curr_step}. Skipping.")
                torch.cuda.empty_cache(); failed += 1; continue
            except FloatingPointError as e:
                print(f"Non-finite loss in micro-batch {i+1}/{num_micro_batches} at step {self.curr_step}. Skipping.")
                failed += 1; continue

        if failed / max(1, len(micro_batches)) > 0.1:
            raise RuntimeError("Too many failed micro-batches. Reduce memory usage or check data quality.")
        
        if self.config.gradient_clipping > 0:
            self.scaler.unscale_(self.optimizer)
            total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
            if not torch.isfinite(total_norm):
                if self.master_process:
                    print(f"Non-finite grad norm at step {self.curr_step}; skipping optimizer step.")
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                self.scheduler.step()
                return results

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return results


if __name__ == "__main__":
    parser = build_parser()
    cfg = parser.parse_args()
    trainer = Trainer(cfg)
    trainer.train()
