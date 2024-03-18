source config.sh  # To get the paths to $proteinnpt_data_path and $proteinnpt_repo_path

# Environment setup - Conda
conda env create -f $proteinnpt_repo_path/proteinnpt_env.yml
conda activate proteinnpt_env

# Environment setup - Python venv (Optional: if you prefer python venv over conda)
# python -m venv $proteinnpt_data_path/proteinnpt_env
# source $proteinnpt_data_path/proteinnpt_env/bin/activate
# pip install -r $proteinnpt_repo_path/requirements.txt

# Download model checkpoints (you may comment lines corresponding to model checkpoints you will not use)
curl -o $proteinnpt_data_path/Tranception/Tranception_Large_checkpoint.zip https://marks.hms.harvard.edu/tranception/Tranception_Large_checkpoint.zip
unzip $proteinnpt_data_path/Tranception/Tranception_Large_checkpoint.zip -d $proteinnpt_data_path/Tranception && rm -f $proteinnpt_data_path/Tranception/Tranception_Large_checkpoint.zip
curl -o $proteinnpt_data_path/ESM/MSA_Transformer/esm_msa1b_t12_100M_UR50S.pt https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1_t12_100M_UR50S.pt
curl -o $proteinnpt_data_path/ESM/ESM1v/esm1v_t33_650M_UR90S_1.pt https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt

# If you would like to compute zero-shot predictions with ESM1v for new assays, you may also want to download checkpoint from other model seeds used in the scoring ensemble. Uncomment the lines below
#curl -o $proteinnpt_data_path/ESM/ESM1v/esm1v_t33_650M_UR90S_2.pt https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_2.pt
#curl -o $proteinnpt_data_path/ESM/ESM1v/esm1v_t33_650M_UR90S_3.pt https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_3.pt
#curl -o $proteinnpt_data_path/ESM/ESM1v/esm1v_t33_650M_UR90S_4.pt https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_4.pt
#curl -o $proteinnpt_data_path/ESM/ESM1v/esm1v_t33_650M_UR90S_5.pt https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_5.pt