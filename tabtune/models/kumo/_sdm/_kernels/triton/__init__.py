# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

if sys.platform != "linux":
    raise ImportError("Triton kernels are only available on Linux")
