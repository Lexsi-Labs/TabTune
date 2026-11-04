# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation utilities."""

from tabtune.models.kumo._sdm.evaluation.classification import to_class_indices, to_binary_class

__all__ = [
    "to_class_indices",
    "to_binary_class",
]
