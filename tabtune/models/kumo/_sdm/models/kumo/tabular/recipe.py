# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import tabtune.models.kumo._sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    def numerical_processor() -> sp.Sequential:
        return sp.Sequential(
            # In single precision, a few huge outliers collapse the
            # standardized values of all other rows onto one value.
            sp.Cast(torch.float64),
            sp.DropConstantColumns(),
            sp.Standardize(eps=1e-6),
            sp.Clip(-100.0, 100.0),
            sp.Choice(
                sp.Identity(),
                sp.PowerTransform(),
                [
                    sp.RobustScale(),
                    sp.ClipSoft(3.0),
                ],
                sp.RankGaussian(max_knots=8192),
                method="round_robin",
            ),
            sp.ClipSigma(threshold=4.0),
        )

    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                numerical=[
                    numerical_processor(),
                    sp.FlipSign(),
                ],
                categorical=[
                    sp.AlignCategories(sort_by="value"),
                    sp.AddCategoryCounts(min_cardinality=50),
                    sp.ToNumerical(),
                    numerical_processor(),
                ],
            ),
            sp.ShuffleColumns(method="latin"),
            sp.SelectColumns(500, method="first"),
            sp.Cast(torch.float32),
        ],
        target=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(),
                    sp.ShuffleCategories(method="shift"),
                ],
                numerical=[
                    sp.Standardize(),
                    sp.FlipSign(),
                ],
            ),
        ],
        output=sp.TaskDispatch(
            classification=[
                sp.AverageEstimators(),
                sp.Softmax(),
            ],
            regression=[
                sp.SortQuantiles(),
                sp.AverageEstimators(trim_fraction=0.2),
            ],
        ),
    )
