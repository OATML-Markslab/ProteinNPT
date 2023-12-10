# ProteinNPT

This is the official code repository for the paper ["ProteinNPT: Improving Protein Property Prediction and Design with Non-Parametric Transformers"](https://www.biorxiv.org/content/10.1101/2023.12.06.570473v1)

## Overview
ProteinNPT is a semi-supervised conditional pseudo-generative model for protein property prediction and design.
It is a variant of [Non-Parametric Transformers](https://arxiv.org/abs/2106.02584) which learns a joint representation of full input batches of protein sequences and associated property labels.
It can be used to predict single or multiple protein properties, generate novel sequences via conditional sampling and support iterative protein redesign cycles via Bayesian optimization.

## Setup
We recommend installing the proteinnpt environment via conda as follows:
```
  conda env create -f proteinnpt_env.yml
  conda activate proteinnpt_env
```

## License
This project is available under the MIT license.

## Credits
This codebase leverages code from the [Tranception](https://github.com/OATML-Markslab/Tranception) and [ESM](https://github.com/facebookresearch/esm) GitHub repositories.

## References
If you use this codebase, please cite the following paper:
```bibtex
@article {Notin2023.12.06.570473,
	author = {Pascal Notin and Ruben Weitzman and Debora S Marks and Yarin Gal},
	title = {ProteinNPT: Improving Protein Property Prediction and Design with Non-Parametric Transformers},
	elocation-id = {2023.12.06.570473},
	year = {2023},
	doi = {10.1101/2023.12.06.570473},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/12/07/2023.12.06.570473},
	eprint = {https://www.biorxiv.org/content/early/2023/12/07/2023.12.06.570473.full.pdf},
	journal = {bioRxiv}
}
```
