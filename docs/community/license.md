# License

TabTune's own code is released under the MIT License; the text is in `LICENSE` at the
repository root.

Vendored model code keeps its own licence, in the `LICENSE` file of its folder under
`tabtune/models/`. Weight licences differ from code licences and are recorded per
checkpoint in the registry; `license_mode="commercial"` checks them before any download.
The [Models](../models/overview.md) page has the full table, and each model page has the
details. The short version:

| Licence | Models |
|---|---|
| Apache-2.0 or MIT, commercial use fine | TabPFN v2 (with attribution), TabICL, TabICL v2, Orion-MSP, Orion-MSP v1.5, Orion-BiX, Mitra, Mitra v2, iLTM, xRFM, Kumo Tabular (OpenMDW-1.1 weights), LimiX v1 (with attribution), Chronos family, Toto 1 and 2 |
| Non-commercial weights | TabPFN v2.6, v3, v3.5 and v3.5-fast, TabFM, ConTextTab, EXAONE Tabular, Causilo, LimiX-2, TimesFM 3 |
| Unverified, read the terms | TabLDM, TabDPT, TiRex, TiRex-2, Time-MoE |

## Citation

```bibtex
@misc{tanna2025tabtuneunifiedlibraryinference,
      title={TabTune: A Unified Library for Inference and Fine-Tuning Tabular Foundation Models},
      author={Aditya Tanna and Pratinav Seth and Mohamed Bouadi and Utsav Avaiya and Vinay Kumar Sankarapu},
      year={2025},
      eprint={2511.02802},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.02802},
}
```
