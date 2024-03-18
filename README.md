# ProteinNPT

This is the official code repository for the paper ["ProteinNPT: Improving Protein Property Prediction and Design with Non-Parametric Transformers"](https://papers.nips.cc/paper_files/paper/2023/hash/6a4d5d85f7a52f062d23d98d544a5578-Abstract-Conference.html)

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

The end-to-end process to train ProteinNPT (and other baselines) follows the 3 steps below. For convenience, we also provide a pipeline script (`scripts/pipeline.sh`) which performs these various steps sequentially with minimal user input (see details below).

### Step 1: Extract sequence embeddings (optional)
Run `embeddings_subs.sh` (or `embeddings_indels.sh`) to create sequence embeddings with the pretrained protein language model of interest, for the desired DMS assays.
This step is optional (embeddings are computed on-the-fly otherwise) and will require sufficient disk space to save pre-computed embeddings, but it will significantly reduce run time and memory requirements during training (especially for ProteinNPT).

### Step 2: Compute zero-shot fitness predictions (optional)
Run `zero_shot_fitness_subs.sh` (or `zero_shot_fitness_indels.sh`) to compute zero-shot fitness predictions with the relevant pretrained protein models. 

Adjust the following variables as needed:
- `assay_index` (index of desired DMS assay in the ProteinGym reference_file under `utils/proteingym`)
- `model_type` (name of the pre-trained model with which to compute embeddings)
- `model_location` (location of the pre-trained model embeddings -- you may use the relevant variables defined in `config.sh` for convenience)

Note that:
1. We provide all zero-shot predictions for ProteinGym DMS assays in `ProteinNPT_data.zip` and thus you do not need to recompute these if interested in these same assays
2. We have found that leveraging zero-shot fitness predictions as additional covariate or auxiliary label generally helps performance, especially when extrapolating to positions not seen during training. However, these zero-shot predictions are not strictly required for ProteinNPT or the various baselines to run, and may be less relevant for predicting properties that differ from fitness.

### Step 3: Train ProteinNPT models (or baselines)
Run `train_subs.sh` (or `train_indels.sh`) to train the desired ProteinNPT or baseline models.

Adjust the following variables as needed:
- `assay_index` (index of desired DMS assay in the ProteinGym reference_file under `utils/proteingym`)
- `model_config_location` (config file for ProteinNPT or baseline model -- you may use the relevant variables defined in `config.sh` for convenience)
- `sequence_embeddings_folder` (location of saved sequence embeddings on disk -- you may use the relevant variables defined in `config.sh` for convenience)
- `fold_variable_name` (type of cross-validation scheme to be used for training -- to be chosen within `fold_random_5`, `fold_contiguous_5`, or `fold_modulo_5`)

We also provide an example script to train a ProteinNPT or baseline model to predict several properties simultaneously in `train_multi_objectives.sh`.

### Pipeline
Run `pipeline.sh` to perform all 3 steps described above sequentially. For ProteinNPT, only 3 parameters are required:
- `assay_data_location` --> full path to the assay you want to train/test on (expects a `.csv` file). At a minimum this file requires 2 fields: mutated_sequence (full sequence of amino acids) and DMS_score (assay measurement). If no fold variable is included in the assay file, the pipeline script will automatically create a fold_random_5 variable, assigning each mutant to folds 0-4 at random. You may also use your own cross-validation scheme (eg., assign all training sequences to fold 0, all test sequences to fold 1). To that end, you only need to pass to the pipeline script the name of that fold variable via the `fold_variable_name` argument and specify the index of the test fold via the `test_fold_index` argument (if `test_fold_index` is not passed as argument, the script will automatically perform a full cross-validation, rotating the test fold index at each iteration).
- `MSA_location` --> full path to the MSA (in .a2m format) to be used to compute MSA Transformer embeddings in ProteinNPT (optional for ESM1v baselines). 
- `target_seq` --> wild type sequence that is mutated in the experimental assay.

By default, the pipeline script assumes that the length of the target sequence `target_seq` is the same as the length of all mutated sequences in the assay data, as well as all sequences in the MSA. If that is not the case (eg., working with indels, or MSA only covering a portion of the target sequence), please adjust the default parameters accordingly (eg., by adding `--indel_mode` to arguments, or by specifying the values of `MSA_start` and `MSA_end`). Please refer to the argsparse parameter description for more details on optional parameters.

## License
This project is available under the MIT license found in the LICENSE file in this GitHub repository.

## Acknowledgements
The `utils` in this codebase leverage code from:
- [Tranception](https://github.com/OATML-Markslab/Tranception)
- [ESM](https://github.com/facebookresearch/esm)
- hhfilter (from the [hhsuite](https://github.com/soedinglab/hh-suite))
- [clustal-omega](http://www.clustal.org/omega/)

## Links
- NeurIPS proceedings: https://papers.nips.cc/paper_files/paper/2023/hash/6a4d5d85f7a52f062d23d98d544a5578-Abstract-Conference.html
- Preprint: https://www.biorxiv.org/content/10.1101/2023.12.06.570473v1

## References
If you use this codebase, please cite the following paper:
```bibtex
@inproceedings{NEURIPS2023_6a4d5d85,
 author = {Notin, Pascal and Weitzman, Ruben and Marks, Debora and Gal, Yarin},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {33529--33563},
 publisher = {Curran Associates, Inc.},
 title = {ProteinNPT: Improving Protein Property Prediction and Design with Non-Parametric Transformers},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/6a4d5d85f7a52f062d23d98d544a5578-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
```
