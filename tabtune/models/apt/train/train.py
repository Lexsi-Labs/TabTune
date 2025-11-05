import os
import logging
from contextlib import nullcontext
from collections import defaultdict

import torch
from torch import nn

from .utils import init_dist, cleanup_dist, get_openai_lr, get_cosine_schedule_with_warmup, evaluate

from apt.model import APT
from apt.data import DataGenerator

def prepare(args, device):
    """
    Instantiates model and dataset to run an experiment.
    """

    # model
    if args.state_dict is None:
        model = APT(classification=args.classification,
            n_blocks=args.n_blocks, d_model=args.d_model,
            d_ff=args.d_ff, n_heads=args.n_heads
        )
    else:
        state_dict, init_args = torch.load(args.state_dict, map_location='cpu')
        model = APT(**init_args)
        model.load_state_dict(state_dict)
    model.to(device)
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M")

    # dataset
    data_loader = DataGenerator(classification=args.classification,
        batch_size=args.batch_size//args.aggregate_k_gradients,
        num_steps=args.steps_per_epoch*args.aggregate_k_gradients,
        data_size=args.data_size, num_datasets=args.num_datasets,
        num_trained_datasets=args.num_trained_datasets,
        device=device
    )
    if args.eval_data is not None:
        assert args.data_path is not None
        data_loader.set_eval_data(args.data_path, args.eval_data.split(','))

    return model, data_loader

def train(args, writer, save_dir):
    # device
    device = args.device if torch.cuda.is_available() else 'cpu'
    using_dist, rank, device = init_dist(device)

    # model & dataloader
    model, data_loader = prepare(args, device)
    if using_dist:
        logging.info("Distributed training")
        model = nn.parallel.DistributedDataParallel(model,
            device_ids=[rank], output_device=rank, broadcast_buffers=False
        )
        data_loader = nn.parallel.DistributedDataParallel(data_loader,
            device_ids=[rank], output_device=rank, broadcast_buffers=False
        )

    # optimizer & scheduler
    if args.lr is None:
        lr = get_openai_lr(model)
        logging.info(f"Using OpenAI max lr of {lr}.")
    else:
        lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        get_cosine_schedule_with_warmup(args.steps_per_epoch * args.max_epochs)
    )

    if args.data_lr is None:
        data_lr = get_openai_lr(data_loader)
        logging.info(f"Using OpenAI max lr of {data_lr}.")
    else:
        data_lr = args.data_lr
    data_optimizer = torch.optim.AdamW(data_loader.parameters(),
        lr=data_lr, weight_decay=args.data_weight_decay)

    # training
    iters = 1
    if args.initial_eval:
        # evaluating
        eval_stats = evaluate(model, data_loader.get_eval_data())
        if rank == 0 and args.eval_data is None:
            for key, val in data_loader.eval_data.items():
                torch.save(val, os.path.join(save_dir, key))

        # reporting
        logging.info('-' * 89)
        logging.info(f'  Epoch: 0')
        for key, val in eval_stats.items():
            logging.info(f'  {key}: {val:5.4f}')
            if rank == 0:
                writer.add_scalar(key, val, 0)
        logging.info('-' * 89)
    for epoch in range(1, args.max_epochs+1):
        # optimizing
        epoch_training_stats = train_epoch(model, optimizer, data_loader, data_optimizer,
            aggregate_k_gradients=args.aggregate_k_gradients, max_grad_norm=args.max_grad_norm,
            using_dist=using_dist, scheduler=scheduler
        )

        # recording
        training_stats = defaultdict(float)
        for key, vals in epoch_training_stats.items():
            for batch, val in vals.items():
                if rank == 0:
                    writer.add_scalar(key, val, iters+batch)
                training_stats[key] += val
            training_stats[key] = training_stats[key] / len(data_loader)
        iters = iters + len(data_loader)

        # reseting
        if epoch % args.reset_freq == 0:
            data_loader.reset_models()
            data_optimizer = torch.optim.AdamW(data_loader.parameters(),
                lr=data_lr, weight_decay=args.data_weight_decay)

        # saving
        if epoch % args.checkpoint_freq == 0 and rank == 0:
            module = model.module if using_dist else model
            module.save_checkpoint(os.path.join(save_dir, f"model_epoch={epoch}.pt"))

        # evaluating
        eval_stats = evaluate(model, data_loader.get_eval_data())

        # reporting
        logging.info('-' * 89)
        logging.info(f'  Epoch: {epoch:d}')
        logging.info(f'  (No Ensembling)')
        for key, val in training_stats.items():
            logging.info(f'  {key}: {val:5.4f}')
        for key, val in eval_stats.items():
            logging.info(f'  {key}: {val:5.4f}')
            if rank == 0:
                writer.add_scalar(key, val, epoch)
        logging.info('-' * 89)

    if using_dist:
        cleanup_dist()

def train_epoch(model, optimizer, data_loader, data_optimizer,
                aggregate_k_gradients=1, max_grad_norm=1.,
                using_dist=False, scheduler=None):
    model.train()
    optimizer.zero_grad()
    data_optimizer.zero_grad()

    epoch_training_stats = defaultdict(dict)
    for batch, (data, targets, split) in enumerate(data_loader):
        if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1):
            cm = model.no_sync()
        else:
            cm = nullcontext()
        with cm:
            loss, loss_dict = model.loss(data, targets, split=split)
            loss = loss / aggregate_k_gradients

            for key, val in loss_dict.items():
                epoch_training_stats[key][batch] = val

            loss.backward()
            if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

                for p in data_loader.parameters():
                    if p.grad is not None:
                        p.grad.data.neg_()
                nn.utils.clip_grad_norm_(data_loader.parameters(), max_grad_norm)
                data_optimizer.step()
                data_optimizer.zero_grad()
    return epoch_training_stats
