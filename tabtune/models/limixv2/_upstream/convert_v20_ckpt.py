#!/usr/bin/env python3
"""Convert a V2.0 training checkpoint into the slim inference checkpoint.

This script is self-contained. It must not import model.v2_0 (that package
may be slimmed later and would break a dependent converter).

Typical usage:

    python script/convert_v20_ckpt.py \\
        --src /path/to/checkpoint_epoch_1000.ckpt \\
        --dst /mnt/datagen/rg/05内部开源模型优化版/LimiX-V2.0-400M_preOpen.ckpt
"""

from __future__ import annotations

import argparse
import random
from fractions import Fraction
from pathlib import Path

import torch


DEFAULT_SRC = (
    "/mnt/public/zhangruiji/models/"
    "rex_rl_0830_cls_cauchyslim_reg_cauchybase_sv6_lt0p03_ht0p01_rep0p4_"
    "tl0p2_th1p0_clw0p9_rlw0p2_0831_0225/checkpoint_epoch_1000.ckpt"
)
DEFAULT_DST = "/mnt/datagen/rg/05内部开源模型优化版/LimiX-V2.0-400M_preOpen.ckpt"

CONFIG_KEY_ALIAS = {
    "emsize": "embed_dim",
}

NONE_DEFAULT_KEYS = {
    "yemb_freeze_type",
    "yemb_type",
    "model_structure_config",
    "x_induced_ffn_hidden_dim",
    "tf_mlp_y_hidden_size_ratio",
}

# Old training key fragments -> inference module names.
STATE_KEY_REPLACEMENTS = (
    ("encoder.0.", "encoder_x.0."),
    (
        "feature_positional_embedding_embeddings.",
        "feature_positional_embedding.",
    ),
)

CRITERION_RENAMES = {
    "reg_criterion.borders": "reg_borders",
    "reg_criterion.bucket_widths": "reg_bucket_widths",
    "reg_criterion._borders": "_reg_borders",
    "reg_criterion.log_widths": "reg_log_widths",
    "reg_criterion.min_border": "reg_min_border",
    "mask_feature_criterion.borders": "mask_feature_border",
    "mask_feature_criterion.bucket_widths": "mask_feature_bucket_widths",
}

DROP_STATE_KEYS = {
    "criterion",
    "cls_criterion.weight",
    "reg_criterion.losses_per_bucket",
    "dynamic_weight_4_loss.params",
    "mask_feature_criterion._borders",
    "mask_feature_criterion.log_widths",
    "reg_mse_criterion._borders",
    "reg_mse_criterion.log_widths",
}


def flatten_dict(input_dict: dict) -> dict:
    result = {}

    def flatten(node: dict) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                flatten(value)
            elif key not in result:
                result[key] = value

    flatten(input_dict)
    return result


def generate_init_seed(model_seed: int) -> dict:
    py_rng_state = random.getstate()
    random.seed(model_seed)
    max_seed = 2 ** 32 - 1
    seed_dict = {
        "decoder_cls_y_seed": random.randint(0, max_seed),
        "decoder_reg_y_seed": random.randint(0, max_seed),
        "decoder_feature_seed": random.randint(0, max_seed),
        "decoder_mask_indicator_seed": random.randint(0, max_seed),
        "feature_positional_embedding_seed": random.randint(0, max_seed),
        "decoder_feature_indicator_reinit_seed": random.randint(0, max_seed),
        "transformer_encoder_output_layernor_seed": random.randint(0, max_seed),
        "encoder_cls_y_seed": random.randint(0, max_seed),
        "encoder_reg_y_seed": random.randint(0, max_seed),
    }
    for layer_idx in range(32):
        for idx in range(16):
            seed_dict[f"layer{layer_idx}_FA{idx}_seed"] = random.randint(0, max_seed)
            seed_dict[f"layer{layer_idx}_SA{idx}_seed"] = random.randint(0, max_seed)
        for midx in range(8):
            for idx in range(32):
                seed_dict[f"layer{layer_idx}_MLP{midx}_{idx}_seed"] = random.randint(
                    0, max_seed
                )
    seed_dict["reg_cls_embedding_projection_seed"] = random.randint(0, max_seed)
    seed_dict["encoder_uniform"] = random.randint(0, max_seed)
    seed_dict["add_task_info_seed"] = random.randint(0, max_seed)
    seed_dict["x_induced_distribution_seed"] = random.randint(0, max_seed)
    random.setstate(py_rng_state)
    return seed_dict


def lookup_config(key: str, sources: list[dict], default=None, *, required=False):
    for source in sources:
        if key in source:
            return _normalize_value(source[key])
        alias = CONFIG_KEY_ALIAS.get(key)
        if alias is not None and alias in source:
            return _normalize_value(source[alias])
    if required and default is None and key not in NONE_DEFAULT_KEYS:
        raise KeyError(f"missing checkpoint config key: {key}")
    if default is None and key in NONE_DEFAULT_KEYS:
        return None
    return _normalize_value(default)


def _normalize_value(value):
    if isinstance(value, Fraction):
        return float(value)
    return value


def collect_sources(ckpt: dict) -> list[dict]:
    sources = []
    if isinstance(ckpt.get("config"), dict):
        sources.append(ckpt["config"])
        sources.append(flatten_dict(ckpt["config"]))
    info = ckpt.get("config_info")
    if isinstance(info, dict):
        if isinstance(info.get("model_config"), dict):
            sources.append(info["model_config"])
        if isinstance(info.get("train_config"), dict):
            sources.append(info["train_config"])
        sources.append(flatten_dict(info))
    return sources


def parse_model_structure_config(
    nlayers: int,
    embed_dim: int,
    nhead: int,
    nhid_factor: float,
    layer_arch: str,
    *,
    activation: str,
    dropout: float,
    pre_norm: bool,
    all_norm: bool,
) -> dict:
    hid_dim = int(embed_dim * nhid_factor)
    layers = []
    for idx in range(nlayers):
        layers.append(
            {
                "layer_idx": idx,
                "arch": layer_arch,
                "emsize": embed_dim,
                "nhead": nhead,
                "nhid_factor": nhid_factor,
                "hid_dim": hid_dim,
                "activation": activation,
                "dropout": dropout,
                "pre_norm": pre_norm,
                "all_norm": all_norm,
            }
        )
    return {
        "global_defaults": {
            "nhead": nhead,
            "nhid_factor": nhid_factor,
            "arch": layer_arch,
            "activation": activation,
            "dropout": dropout,
            "pre_norm": pre_norm,
            "all_norm": all_norm,
        },
        "layers": layers,
        "emsize": embed_dim,
        "nlayers": nlayers,
    }


def build_inference_config(ckpt: dict) -> dict:
    raw = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    if "encoder_config_x" in raw:
        config = dict(raw)
        config.pop("optimizer", None)
        config.pop("scheduler", None)
        return config

    sources = collect_sources(ckpt)
    g = lambda key, default=None, required=False: lookup_config(
        key, sources, default, required=required
    )

    emsize = g("emsize", required=True)
    nlayers = g("nlayers", required=True)
    nhead = g("nhead", required=True)
    nhid_factor = g("nhid_factor", 4)
    layer_arch = g("layer_arch", "smf")
    pre_norm = g("pre_norm", False)
    all_norm = g("all_norm", False)
    activation = g("activation", "gelu")
    # Inference always disables dropout / attention recompute.
    dropout = 0.0
    recompute_attn = g("recompute_attn", False) or False
    features_per_group = g("features_per_group", required=True)
    seed_dict = generate_init_seed(g("fixmodelseed", 3407))
    model_structure_config = g("model_structure_config")
    if model_structure_config is None:
        model_structure_config = parse_model_structure_config(
            nlayers=nlayers,
            embed_dim=emsize,
            nhead=nhead,
            nhid_factor=nhid_factor,
            layer_arch=layer_arch,
            activation=activation,
            dropout=dropout,
            pre_norm=pre_norm,
            all_norm=all_norm,
        )

    numeric_embed_type = g("numeric_embed_type", "linear")
    if numeric_embed_type == "original":
        numeric_embed_type = "linear"

    config = {
        "preprocess_config_x": {
            "num_features": features_per_group,
            "nan_handling_enabled": g("nan_handling_enabled", True),
            "normalize_on_train_only": g("normalize_on_train_only", True),
            "normalize_x": g("normalize_x", True),
            "remove_outliers": g("remove_outliers", False),
            "normalize_by_used_features": g("normalize_by_used_features", True),
        },
        "encoder_config_x": {
            "num_features": features_per_group,
            "embedding_size": emsize,
            "mask_embedding_size": g("mask_embedding_size", emsize),
            "encoder_use_bias": g("encoder_use_bias", True),
            "feature_embedding_type": g(
                "mask_feature_embedding_type", "mask_embedding"
            ),
            "numeric_embed_type": numeric_embed_type,
            "use_feature_group_encoder": g("use_feature_group_encoder", True),
            "feature_group_encoder_hidden_dim": g(
                "feature_group_encoder_hidden_dim", 192
            ),
            "feature_fusion_network_type": g(
                "feature_fusion_network_type", "original"
            ),
            "use_feature_mask_indicator": g("use_feature_mask_indicator", False),
            "mask_indicator_embed_dim": g("mask_indicator_embed_dim", 192),
            "emb_mlp_hidden_dim_ratio": g("emb_mlp_hidden_dim_ratio", 0.5),
            "mask_token_emb_type": g("mask_token_emb_type", "mask_emb"),
            "init_seed_dict": seed_dict,
        },
        "encoder_config_y": {
            "num_inputs": 1,
            "embedding_size": emsize,
            "nan_handling_y_encoder": g("nan_handling_y_encoder", True),
            "max_num_classes": g("max_num_classes", 10),
            "yemb_freeze_type": g("yemb_freeze_type", None),
            "cls_y_encoder_type": g("cls_y_encoder_type", "emby_embedding"),
            "y_encoder_type": g("RBF_reg_y_encoder_type", "linear"),
            "reg_y_numeric_embed_type": g("reg_y_numeric_embed_type", "linear"),
            "cls_emb_learning_scalar": g("cls_emb_learning_scalar", False),
            "cls_y_numeric_embed_type": g("cls_y_numeric_embed_type", "linear"),
            "emb_mlp_hidden_dim_ratio": g("emb_mlp_hidden_dim_ratio", 0.5),
            "mask_indicator_embed_dim": g("mask_indicator_embed_dim", 192),
            "reg_y_mask_indicator_type": g("reg_y_mask_indicator_type", "linear"),
            "mask_token_emb_type": g("mask_token_emb_type", "mask_emb"),
            "cls_random_mapping": g("cls_random_mapping", False),
            "num_features": features_per_group,
            "init_seed_dict": seed_dict,
        },
        "decoder_config": {"num_classes": g("max_num_classes", 10)},
        "feature_positional_embedding_type": g(
            "feature_positional_embedding_type", "subspace"
        ),
        "nlayers": nlayers,
        "nhead": nhead,
        "embed_dim": emsize,
        "hid_dim": int(emsize * nhid_factor),
        "tf_mlp_layer_type": g("tf_mlp_layer_type", "gated"),
        "tf_mlp_activation_fuction": g("tf_mlp_activation_fuction", "silu"),
        "use_gated_norm": g("use_gated_norm", False),
        "tf_mlp_use_bias": g("tf_mlp_use_bias", True),
        "tf_attention_layer_type": g("tf_attention_layer_type", "bert"),
        "tf_attention_use_bias": g("tf_attention_use_bias", True),
        "output_proj_w_init_type": g("output_proj_w_init_type", "original"),
        "tf_norm_type": g("tf_norm_type", "rmsnorm"),
        "rmsnorm_impl": g("rmsnorm_impl", "triton"),
        "tf_layer_norm_use_elementwise_affine": g(
            "tf_layer_norm_use_elementwise_affine", False
        ),
        "mask_feature_embedding_type": g(
            "mask_feature_embedding_type", "mask_embedding"
        ),
        "enable_mask_feature_pred": g("enable_mask_feature_pred", True),
        "enable_mask_indicator_pred": g("enable_mask_indicator_pred", False),
        "mask_prediction": False,
        "features_per_group": features_per_group,
        "dropout": dropout,
        "tf_mlp_dropout": 0.0,
        "decoder_dropout": 0.0,
        "pre_norm": pre_norm,
        "all_norm": all_norm,
        "activation": activation,
        "recompute_attn": recompute_attn,
        "layer_recompute": g("layer_recompute", False),
        "layer_recompute_start_layer": g("layer_recompute_start_layer", 0),
        "layer_arch": layer_arch,
        "use_separate_attention": g("use_separate_attention", False),
        "separate_attn_kv_combined": g("separate_attn_kv_combined", False),
        "sample_attention_cls_token_merge": g(
            "sample_attention_cls_token_merge", "reshape"
        ),
        "scale_y_sample_attention_heads": g("scale_y_sample_attention_heads", False),
        "cls_sample_attention_num_heads": g("cls_sample_attention_num_heads", 0),
        "cls_only_sample_attention_num_heads": g(
            "cls_only_sample_attention_num_heads", 0
        ),
        "feature_attention_cls_token_merge": g(
            "feature_attention_cls_token_merge", "none"
        ),
        "num_cls_tokens": g("num_cls_tokens", 1),
        "cls_only_start_layer": g("cls_only_start_layer", -1),
        "seq_att_use_softmax_scaling_mlp_for_q": g(
            "seq_att_use_softmax_scaling_mlp_for_q", False
        ),
        "fea_att_use_softmax_scaling_mlp_for_q": g(
            "fea_att_use_softmax_scaling_mlp_for_q", False
        ),
        "sa_temp_upper_bound": g("sa_temp_upper_bound", 0.5),
        "sa_temp_lower_bound": g("sa_temp_lower_bound", 0.0),
        "fa_temp_upper_bound": g("fa_temp_upper_bound", 0.4),
        "fa_temp_lower_bound": g("fa_temp_lower_bound", 0.0),
        "sa_scale_base_bound": g("sa_scale_base_bound", 5.0),
        "fa_scale_base_bound": g("fa_scale_base_bound", 1.0),
        "seq_att_use_q_norm": g("seq_att_use_q_norm", False),
        "fea_att_use_q_norm": g("fea_att_use_q_norm", False),
        "seq_att_use_k_norm": g("seq_att_use_k_norm", False),
        "fea_att_use_k_norm": g("fea_att_use_k_norm", False),
        "seq_att_q_norm_use_elementwise_affine": g(
            "seq_att_q_norm_use_elementwise_affine", False
        ),
        "fea_att_q_norm_use_elementwise_affine": g(
            "fea_att_q_norm_use_elementwise_affine", False
        ),
        "seq_att_k_norm_use_elementwise_affine": g(
            "seq_att_k_norm_use_elementwise_affine", False
        ),
        "fea_att_k_norm_use_elementwise_affine": g(
            "fea_att_k_norm_use_elementwise_affine", False
        ),
        "self_share_all_kv_heads": g("self_share_all_kv_heads", False),
        "cross_share_all_kv_heads": g("cross_share_all_kv_heads", True),
        "seq_attn_isolated": g("seq_attn_isolated", False),
        "seq_attn_serial": g("seq_attn_serial", False),
        "mask_feature_filling_type": g("mask_feature_filling_type", "mask_token"),
        "enable_classification_y_pred": g("enable_classification_y_pred", True),
        "enable_regression_y_pred": g("enable_regression_y_pred", True),
        "num_buckets": g("num_buckets", 0),
        "border_range": g("border_range", [-10.0, 10.0]),
        "learnable_borders": g("learnable_borders", False),
        "reg_borders_use_train_quantile": g("reg_borders_use_train_quantile", False),
        "use_rmsnorm": g("use_rmsnorm", False),
        "flash_attention_precision": g("flash_attention_precision", "inherit"),
        "init_seed_dict": seed_dict,
        "init_seed_mlp_mode": g("init_seed_mlp_mode", "split") or "split",
        "use_feature_post_adapter": g("use_feature_post_adapter", False),
        "feature_post_adapter_pattern": g(
            "feature_post_adapter_pattern", "linear-192"
        ),
        "use_feature_post_adapter_residual": g(
            "use_feature_post_adapter_residual", False
        ),
        "use_cls_post_adapter": g("use_cls_post_adapter", False),
        "cls_post_adapter_pattern": g("cls_post_adapter_pattern", "linear-192"),
        "use_cls_post_adapter_residual": g("use_cls_post_adapter_residual", False),
        "use_reg_post_adapter": g("use_reg_post_adapter", False),
        "reg_post_adapter_pattern": g("reg_post_adapter_pattern", "linear-192"),
        "use_reg_post_adapter_residual": g("use_reg_post_adapter_residual", False),
        "decoder_hidden_dim": g("decoder_hidden_dim", emsize * 4),
        "cls_y_decoder_type": g("cls_y_decoder_type", "original"),
        "cls_y_decoder_mlp_pattern": g(
            "cls_y_decoder_mlp_pattern", "linear-768_GELU_linear-10"
        ),
        "reg_y_decoder_type": g("reg_y_decoder_type", "original"),
        "reg_y_decoder_mlp_pattern": g(
            "reg_y_decoder_mlp_pattern", "linear-768_GELU_linear-10"
        ),
        "reg_y_mse_decoder_type": g("reg_y_mse_decoder_type", "original"),
        "feature_mse_decoder_type": g("feature_mse_decoder_type", "original"),
        "feature_decoder_type": g("feature_decoder_type", "origin_mse_head"),
        "use_shared_encoder": g("use_shared_encoder", False),
        "model_structure_config": model_structure_config,
        "feature_attention_pipeline": g("feature_attention_pipeline", "legacy"),
        "decoupled_attn_kv_combined": g("decoupled_attn_kv_combined", False),
        "dsti_summary_dim": g("dsti_summary_dim", 0),
        "dsti_relation_dim": g("dsti_relation_dim", 0),
        "dsti_value_dim": g("dsti_value_dim", 0),
        "dsti_num_heads": g("dsti_num_heads", 0),
        "dsti_num_summary_tokens": g("dsti_num_summary_tokens", 1),
        "dsti_feature_interaction_mode": g(
            "dsti_feature_interaction_mode", "sample_split"
        ),
        "dsti_y_token_aggregation": g("dsti_y_token_aggregation", "concat_linear"),
        "dsti_dropout": g("dsti_dropout", -1.0),
        "dsti_use_qk_norm": g("dsti_use_qk_norm", False),
        "dsti_qk_norm_eps": g("dsti_qk_norm_eps", 1e-5),
        "dsti_qk_norm_elementwise_affine": g(
            "dsti_qk_norm_elementwise_affine", True
        ),
        "dsti_use_softmax_scaling_mlp": g("dsti_use_softmax_scaling_mlp", False),
        "dsti_softmax_scaling_temp_upper_bound": g(
            "dsti_softmax_scaling_temp_upper_bound", 0.4
        ),
        "dsti_softmax_scaling_temp_lower_bound": g(
            "dsti_softmax_scaling_temp_lower_bound", 0.0
        ),
        "dsti_softmax_scaling_base_bound": g(
            "dsti_softmax_scaling_base_bound", 1.0
        ),
        "dsti_attention_backend": g("dsti_attention_backend", "auto"),
        "dsti_use_updated_x_for_y": g("dsti_use_updated_x_for_y", True),
        "dsti_summary_query_init_std": g("dsti_summary_query_init_std", 0.02),
        "separate_x_y_ffn": g("separate_x_y_ffn", False),
        "disable_test_use_train_first_head": g(
            "disable_test_use_train_first_head", False
        ),
        "feature_emb_layer": g("feature_emb_layer", nlayers - 1),
        "reg_y_emb_layer": g("reg_y_emb_layer", nlayers - 1),
        "cls_y_emb_layer": g("cls_y_emb_layer", nlayers - 1),
        "seq_att_kv_num_heads": g("seq_att_kv_num_heads", nhead),
        "encoder_out_norm_use_affine": g("encoder_out_norm_use_affine", False),
        "add_task_type": g("add_task_type", False),
        "tf_mlp_hidden_size_ratio": g("tf_mlp_hidden_size_ratio", 4.0),
        "tf_mlp_hidden_size_2_even": g("tf_mlp_hidden_size_2_even", False),
        "tf_mlp_y_hidden_size_ratio": g("tf_mlp_y_hidden_size_ratio", None),
        "induce_use_qk_norm": g("induce_use_qk_norm", g("seq_att_use_q_norm", False)),
        "induce_qk_norm_eps": g("induce_qk_norm_eps", 1e-5),
        "induce_qk_norm_elementwise_affine": g(
            "induce_qk_norm_elementwise_affine", True
        ),
        "induce_use_softmax_scaling_mlp": g(
            "induce_use_softmax_scaling_mlp",
            g("seq_att_use_softmax_scaling_mlp_for_q", False),
        ),
        "induce_softmax_scaling_temp_upper_bound": g(
            "induce_softmax_scaling_temp_upper_bound", 0.5
        ),
        "induce_softmax_scaling_temp_lower_bound": g(
            "induce_softmax_scaling_temp_lower_bound", 0.0
        ),
        "induce_softmax_scaling_base_bound": g(
            "induce_softmax_scaling_base_bound", 5.0
        ),
        "mlp_use_residual": g("mlp_use_residual", True),
        "qkv_w_init_method": g("qkv_w_init_method", "xavier_uniform"),
    }
    return config


def remap_state_key(key: str) -> str | None:
    if key in DROP_STATE_KEYS or key == "criterion":
        return None
    key = key.replace("module.", "")
    if key in DROP_STATE_KEYS:
        return None
    if "cls_criterion.weight" in key:
        return None
    if "reg_y_encoder" not in key:
        for old, new in STATE_KEY_REPLACEMENTS:
            if old in key:
                key = key.replace(old, new)
                break
    if key in CRITERION_RENAMES:
        key = CRITERION_RENAMES[key]
    return key


def remap_old_scaling_keys(state: dict) -> dict:
    """Map legacy 1-D softmax-scaling params onto scale_linear.{weight,bias}."""
    for key in list(state.keys()):
        if "softmax_scaling_mlp_for_q.base_scale_param_H" in key:
            new_key = key.replace(
                "softmax_scaling_mlp_for_q.base_scale_param_H",
                "softmax_scaling_mlp_for_q.scale_linear.bias",
            )
            state[new_key] = state.pop(key).reshape(-1)
        elif "softmax_scaling_mlp_for_q.logn_sensitivity_H" in key:
            new_key = key.replace(
                "softmax_scaling_mlp_for_q.logn_sensitivity_H",
                "softmax_scaling_mlp_for_q.scale_linear.weight",
            )
            state[new_key] = state.pop(key).reshape(-1, 1)
    return state


def convert_state_dict(raw_state: dict) -> dict:
    new_state = {}
    for key, value in raw_state.items():
        new_key = remap_state_key(key)
        if new_key is None:
            continue
        new_state[new_key] = value
    return remap_old_scaling_keys(new_state)


def convert_checkpoint(src: Path, dst: Path) -> dict:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise KeyError(f"{src} has no state_dict")
    config = build_inference_config(ckpt)
    state = convert_state_dict(dict(ckpt["state_dict"]))
    out = {
        "state_dict": state,
        "config": config,
        "arch_version": ckpt.get("arch_version", "2.0"),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return {
        "src": str(src),
        "dst": str(dst),
        "n_src": len(ckpt["state_dict"]),
        "n_dst": len(state),
        "config_keys": len(config),
        "has_reg_borders": "_reg_borders" in state,
        "has_encoder_config_x": "encoder_config_x" in config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Path to the source checkpoint")
    parser.add_argument("--dst", type=Path, required=True, help="Path to the destination checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = convert_checkpoint(args.src, args.dst)
    print("converted checkpoint")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
