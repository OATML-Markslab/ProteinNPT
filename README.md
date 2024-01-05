# ProteinNPT

This is the official code repository for the paper ["ProteinNPT: Improving Protein Property Prediction and Design with Non-Parametric Transformers"](https://www.biorxiv.org/content/10.1101/2023.12.06.570473v1)

## Overview
ProteinNPT is a semi-supervised conditional pseudo-generative model for protein property prediction and design.
It is a variant of [Non-Parametric Transformers](https://arxiv.org/abs/2106.02584) which learns a joint representation of full input batches of protein sequences and associated property labels.
It can be used to predict single or multiple protein properties, generate novel sequences via conditional sampling and support iterative protein redesign cycles via Bayesian optimization.

## Setup

### Step 1: Download the ProteinNPT data files
```
curl -o ProteinNPT_data.zip https://marks.hms.harvard.edu/ProteinNPT/ProteinNPT_data.zip
unzip ProteinNPT_data.zip && rm ProteinNPT_data.zip
```
We recommend saving this file in a location where disk space is not a concern. The above file will use 8.2GB when unzipped, but much more space will be needed as you download pre-trained model checkpoints (~10GB) and save sequence embeddings in subsequent steps (~1TB for all MSA Transformer embeddings).

### Step 2: Configure environment
Edit lines 2 & 3 of the setup.sh bash script under the `scripts` folder with the location of 1) the unzipped file downloaded in step 1 (the `data_path`) and 2) your local copy of this GitHub repository (the `repo_path`).
Then run the `setup.sh` script. This will sequentially:
- Create the conda environment and setup the repository locally
- Download the model checkpoints to the relevant location in the `data_path`

### Step 3: Edit config file
Edit lines 2 & 3 of the `config.sh` bash script with the `data_path` and `repo_path` (identical to lines 2 & 3 from `setup.sh`)

## Usage

## Step 1: Extract sequence embeddings (optional)
Run `embeddings_subs.sh` (or `embeddings_indels.sh`) to create sequence embeddings with the pretrained protein language model of interest, for the desired DMS assays.
This step is optional (embeddings are computed on-the-fly otherwise) and will require sufficient disk space to save pre-computed embeddings, but it will significantly reduce run time and memory requirements during training (especially for ProteinNPT).

## Step 2: Compute zero-shot fitness predictions (optional)
Run `zero_shot_fitness_subs.sh` (or `zero_shot_fitness_indels.sh`) to compute zero-shot fitness predictions with the relevant pretrained protein models. 

Adjust the following variables as needed:
- `assay_index` (index of desired DMS assay in the ProteinGym reference_file under `utils/proteingym`)
- `model_type` (name of the pre-trained model with which to compute embeddings)
- `model_location` (location of the pre-trained model embeddings -- you may use the relevant variables defined in `config.sh` for convenience)

Note that:
1. We provide all zero-shot predictions for ProteinGym DMS assays in `ProteinNPT_data.zip` and thus you do not need to recompute these if interested in these same assays
2. We have found that leveraging zero-shot fitness predictions as additional covariate or auxiliary label generally helps performance, especially when extrapolating to positions not seen during training. However, these zero-shot predictions are not strictly required for ProteinNPT or the various baselines to run, and may be less relevant for predicting properties that differ from fitness.

## Step 3: Train ProteinNPT models (or baselines)
Run `train_subs.sh` (or `train_indels.sh`) to train the desired ProteinNPT or baseline models.

Adjust the following variables as needed:
- `assay_index` (index of desired DMS assay in the ProteinGym reference_file under `utils/proteingym`)
- `model_config_location` (config file for ProteinNPT or baseline model -- you may use the relevant variables defined in `config.sh` for convenience)
- `sequence_embeddings_folder` (location of saved sequence embeddings on disk -- you may use the relevant variables defined in `config.sh` for convenience)
- `fold_variable_name` (type of cross-validation scheme to be used for training -- to be chosen within `fold_random_5`, `fold_contiguous_5`, or `fold_modulo_5`)

We also provide an example script to train a ProteiNPT or baseline model to predict several properties simultaneously in `train_multi_objectives.sh`.

## License
This project is available under the MIT license found in the LICENSE file in this GitHub repository.

## Acknowledgements
The `utils` in this codebase leverage code from:
- [Tranception](https://github.com/OATML-Markslab/Tranception)
- [ESM](https://github.com/facebookresearch/esm)
- hhfilter (from the [hhsuite](https://github.com/soedinglab/hh-suite))
- [clustal-omega](http://www.clustal.org/omega/)

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
