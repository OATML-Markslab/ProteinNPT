# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import os,sys
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple
import torch

from utils import esm
from utils.esm import pretrained, MSATransformer

from utils.tranception.utils.scoring_utils import get_optimal_window

def label_row(row, sequence, token_probs, alphabet, offset_idx):
    score=0
    #We sum the scores for each mutant
    for mutation in row.split(":"):
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

        # add 1 for BOS
        score += (token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]).item()
    return score


def compute_pppl(row, sequence, model, alphabet, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    # modify the sequence
    sequence = sequence[:idx] + mt + sequence[(idx + 1) :]

    # encode the sequence
    data = [
        ("protein1", sequence),
    ]

    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # compute probabilities at each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
        log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
    return sum(log_probs)


def get_fitness_scores(dataset_location, sequence, model_locations=[], scoring_strategy="masked-marginals", scoring_window="optimal", mutation_col="mutant", offset_idx=1):
    # Load the deep mutational scan
    df = pd.read_csv(dataset_location)

    # inference for each model
    for model_location in model_locations:
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
        model_location = model_location.split("/")[-1].split(".")[0]
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            print("Transferred model to GPU")

        batch_converter = alphabet.get_batch_converter()

        if isinstance(model, MSATransformer):
            args.offset_idx = msa_start_index
            data = [read_msa(filename=args.msa_path, nseq=args.msa_samples, sampling_strategy=args.msa_sampling_strategy, random_seed=args.seed, weight_filename=MSA_weight_file_name,
                            filter_msa=args.filter_msa, hhfilter_min_cov=args.hhfilter_min_cov, hhfilter_max_seq_id=args.hhfilter_max_seq_id, hhfilter_min_seq_id=args.hhfilter_min_seq_id, path_to_hhfilter=args.path_to_hhfilter)]
            assert (
                scoring_strategy == "masked-marginals"
            ), "MSA Transformer only supports masked marginal strategy"

            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            all_token_probs = []
            for i in tqdm(range(batch_tokens.size(2))):
                batch_tokens_masked = batch_tokens.clone()
                batch_tokens_masked[0, 0, i] = alphabet.mask_idx  # mask out first sequence
                #print(batch_tokens_masked)
                #print(batch_tokens_masked.shape) #torch.Size([1, 384, 102])
                if batch_tokens.size(-1) > 1024: 
                    large_batch_tokens_masked=batch_tokens_masked.clone()
                    start, end = get_optimal_window(mutation_position_relative=i, seq_len_wo_special=len(sequence)+2, model_window=1024)
                    print("Start index {} - end index {}".format(start,end))
                    batch_tokens_masked = large_batch_tokens_masked[:,:,start:end]
                else:
                    start=0
                with torch.no_grad():
                    token_probs = torch.log_softmax(
                        model(batch_tokens_masked.cuda())["logits"], dim=-1
                    )
                all_token_probs.append(token_probs[:, 0, i-start])  # vocab size
            token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
            df[model_location] = df.apply(
                lambda row: label_row(
                    row[mutation_col], sequence, token_probs, alphabet, offset_idx
                ),
                axis=1,
            )

        else:
            data = [
                ("protein1", sequence),
            ]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            if scoring_strategy == "wt-marginals":
                # batch_tokens.shape --> #torch.Size([1, seq_len])
                with torch.no_grad():
                    if batch_tokens.size(1) > 1024 and scoring_window=="overlapping": 
                        batch_size, seq_len = batch_tokens.shape #seq_len includes BOS and EOS
                        token_probs = torch.zeros((batch_size,seq_len,len(alphabet))).cuda() # Note: batch_size = 1 (need to keep batch dimension to score with model though)
                        token_weights = torch.zeros((batch_size,seq_len)).cuda()
                        weights = torch.ones(1024).cuda() # 1 for 256≤i<1022-256
                        for i in range(1,257):
                            weights[i] = 1 / (1 + math.exp(-(i-128)/16))
                        for i in range(1022-256,1023):
                            weights[i] = 1 / (1 + math.exp((i-1022+128)/16))
                        start_left_window = 0
                        end_left_window = 1023 #First window is indexed [0-1023]
                        start_right_window = (batch_tokens.size(1) - 1) - 1024 + 1 #Last index is len-1
                        end_right_window = batch_tokens.size(1) - 1
                        while True: 
                            # Left window update
                            left_window_probs = torch.log_softmax(model(batch_tokens[:,start_left_window:end_left_window+1].cuda())["logits"], dim=-1)
                            token_probs[:,start_left_window:end_left_window+1] += left_window_probs * weights.view(-1,1)
                            token_weights[:,start_left_window:end_left_window+1] += weights
                            # Right window update
                            right_window_probs = torch.log_softmax(model(batch_tokens[:,start_right_window:end_right_window+1].cuda())["logits"], dim=-1)
                            token_probs[:,start_right_window:end_right_window+1] += right_window_probs * weights.view(-1,1)
                            token_weights[:,start_right_window:end_right_window+1] += weights
                            if end_left_window > start_right_window:
                                #we had some overlap between windows in that last scoring so we break from the loop
                                break
                            start_left_window+=511
                            end_left_window+=511
                            start_right_window-=511
                            end_right_window-=511
                        #If central overlap not wide engouh, we add one more window at the center
                        final_overlap = end_left_window - start_right_window + 1
                        if final_overlap < 511:
                            start_central_window = int(seq_len / 2) - 512
                            end_central_window = start_central_window + 1023
                            central_window_probs = torch.log_softmax(model(batch_tokens[:,start_central_window:end_central_window+1].cuda())["logits"], dim=-1)
                            token_probs[:,start_central_window:end_central_window+1] += central_window_probs * weights.view(-1,1)
                            token_weights[:,start_central_window:end_central_window+1] += weights
                        #Weight normalization
                        token_probs = token_probs / token_weights.view(-1,1) #Add 1 to broadcast
                    else:                    
                        token_probs = torch.log_softmax(model(batch_tokens.cuda())["logits"], dim=-1)
                df[model_location] = df.apply(
                    lambda row: label_row(
                        row[mutation_col],
                        sequence,
                        token_probs,
                        alphabet,
                        offset_idx,
                    ),
                    axis=1,
                )
            elif scoring_strategy == "masked-marginals":
                all_token_probs = []
                for i in tqdm(range(batch_tokens.size(1))):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0, i] = alphabet.mask_idx
                    if batch_tokens.size(1) > 1024 and scoring_window=="optimal": 
                        large_batch_tokens_masked=batch_tokens_masked.clone()
                        start, end = get_optimal_window(mutation_position_relative=i, seq_len_wo_special=len(sequence)+2, model_window=1024)
                        batch_tokens_masked = large_batch_tokens_masked[:,start:end]
                    elif batch_tokens.size(1) > 1024 and scoring_window=="overlapping": 
                        print("Overlapping not yet implemented for masked-marginals")
                        sys.exit(0)
                    else:
                        start=0
                    with torch.no_grad():
                        token_probs = torch.log_softmax(
                            model(batch_tokens_masked.cuda())["logits"], dim=-1
                        )
                    all_token_probs.append(token_probs[:, i-start])  # vocab size
                token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
                df[model_location] = df.apply(
                    lambda row: label_row(
                        row[mutation_col],
                        sequence,
                        token_probs,
                        alphabet,
                        offset_idx,
                    ),
                    axis=1,
                )
            elif scoring_strategy == "pseudo-ppl":
                tqdm.pandas()
                df[model_location] = df.progress_apply(
                    lambda row: compute_pppl(
                        row[mutation_col], sequence, model, alphabet, offset_idx
                    ),
                    axis=1,
                )

    return df