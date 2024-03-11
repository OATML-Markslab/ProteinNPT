import numpy as np
import pandas as pd
from collections import defaultdict
import random
import os
import torch
import subprocess
from proteinnpt.utils.msa_utils import MSA_processing

def filter_msa(msa_data, num_sequences_kept=3):
    """
    Helper function to filter an input MSA msa_data (obtained via process_msa_data) and keep only num_sequences_kept aligned sequences.
    If the MSA already has fewer sequences than num_sequences_kept, we keep the MSA as is.
    If filtering, we always keep the first sequence of the MSA (ie. the wild type) by default.
    Sampling is done without replacement.
    """
    if len(list(msa_data.keys())) <= num_sequences_kept:
        return  msa_data
    filtered_msa = {}
    wt_name = next(iter(msa_data)) 
    filtered_msa[wt_name] = msa_data[wt_name]
    del msa_data[wt_name]
    sequence_names = list(msa_data.keys())
    sequence_names_sampled  = random.sample(sequence_names,k=num_sequences_kept-1)
    for seq in sequence_names_sampled:
        filtered_msa[seq] = msa_data[seq]
    return filtered_msa

def process_msa_data(MSA_data_file):
    """
    Helper function that takes as input a path to a MSA file (expects a2m format) and returns a dict mapping sequence ID to the corresponding AA sequence.
    """
    msa_data = defaultdict(str)
    sequence_name = ""
    with open(MSA_data_file, "r") as msa_file:
        for i, line in enumerate(msa_file):
            line = line.rstrip()
            if line.startswith(">"):
                sequence_name = line
            else:
                msa_data[sequence_name] += line.upper()
    return msa_data

def get_one_hot_sequences_dict(msa_data,MSA_start,MSA_end,vocab):
    vocab_size = len(vocab.keys())
    num_sequences_msa = len(msa_data.keys())
    one_hots = np.zeros((num_sequences_msa,MSA_end-MSA_start,vocab_size))
    for i,seq_name in enumerate(msa_data.keys()):
        sequence = msa_data[seq_name]
        for j,letter in enumerate(sequence):
            if letter in vocab: 
                k = vocab[letter]
                one_hots[i,j,k] = 1.0
    return one_hots

def one_hot(sequence_string,vocab):
    one_hots = np.zeros((len(sequence_string),len(vocab.keys())))
    for j,letter in enumerate(sequence_string):
        if letter in vocab: 
            k = vocab[letter]
            one_hots[j,k] = 1.0
    return one_hots.flatten()

def get_msa_prior(MSA_data_file, MSA_weight_file_name, MSA_start, MSA_end, len_target_seq, vocab, retrieval_aggregation_mode="aggregate_substitution", filter_MSA=True, verbose=False):
    """
    Function to enable retrieval inference mode, via computation of (weighted) pseudocounts of AAs at each position of the retrieved MSA.
    MSA_data_file: (string) path to MSA file (expects a2m format).
    MSA_weight_file_name: (string) path to sequence weights in MSA.
    MSA_start: (int) Sequence position that the MSA starts at (1-indexing).
    MSA_end: (int) Sequence position that the MSA ends at (1-indexing).
    len_target_seq: (int) Full length of sequence to be scored.
    vocab: (dict) Vocabulary of the tokenizer.
    retrieval_aggregation_mode: (string) Mode for retrieval inference (aggregate_substitution Vs aggregate_indel). If None, places a uniform prior over each token.
    filter_MSA: (bool) Whether to filter out sequences with very low hamming similarity (< 0.2) to the reference sequence in the MSA (first sequence).
    verbose: (bool) Whether to print to the console processing details along the way.
    """
    msa_data = process_msa_data(MSA_data_file)
    vocab_size = len(vocab.keys())
    if verbose: print("Target seq len is {}, MSA length is {}, start position is {}, end position is {} and vocab size is {}".format(len_target_seq,MSA_end-MSA_start,MSA_start,MSA_end,vocab_size))

    if filter_MSA:
        if verbose: print("Num sequences in MSA pre filtering: {}".format(len(msa_data.keys())))
        list_sequence_names = list(msa_data.keys())
        focus_sequence_name = list(msa_data.keys())[0]
        ref_sequence_hot = one_hot(msa_data[focus_sequence_name],vocab)
        for sequence_name in list_sequence_names:
            seq_hot = one_hot(msa_data[sequence_name],vocab)
            hamming_similarity_seq_ref = np.dot(ref_sequence_hot,seq_hot) / np.dot(ref_sequence_hot,ref_sequence_hot)
            if hamming_similarity_seq_ref < 0.2:
                del msa_data[sequence_name]
        if verbose: print("Num sequences in MSA post filtering: {}".format(len(msa_data.keys())))

    if MSA_weight_file_name is not None:
        if verbose: print("Using weights in {} for sequences in MSA.".format(MSA_weight_file_name))
        assert os.path.exists(MSA_weight_file_name), "Weights file not located on disk."
        MSA_EVE = MSA_processing(
                MSA_location=MSA_data_file,
                use_weights=True,
                weights_location=MSA_weight_file_name
        )
        #We scan through all sequences to see if we have a weight for them as per EVE pre-processing. We drop them otherwise.
        dropped_sequences=0
        list_sequence_names = list(msa_data.keys())
        MSA_weight=[]
        for sequence_name in list_sequence_names:
            if sequence_name not in MSA_EVE.seq_name_to_sequence:
                dropped_sequences +=1
                del msa_data[sequence_name]
            else:
                MSA_weight.append(MSA_EVE.seq_name_to_weight[sequence_name])
        if verbose: print("Dropped {} sequences from MSA due to absent sequence weights".format(dropped_sequences))
    else:
        MSA_weight = [1] *  len(list(msa_data.keys()))

    if retrieval_aggregation_mode=="aggregate_substitution" or retrieval_aggregation_mode=="aggregate_indel":
        one_hots = get_one_hot_sequences_dict(msa_data,MSA_start,MSA_end,vocab)
        MSA_weight = np.expand_dims(np.array(MSA_weight),axis=(1,2))
        base_rate = 1e-5
        base_rates = np.ones_like(one_hots) * base_rate
        weighted_one_hots = (one_hots + base_rates) * MSA_weight
        MSA_weight_norm_counts = weighted_one_hots.sum(axis=-1).sum(axis=0)
        MSA_weight_norm_counts = np.tile(MSA_weight_norm_counts.reshape(-1,1), (1,vocab_size))
        one_hots_avg = weighted_one_hots.sum(axis=0) / MSA_weight_norm_counts
        msa_prior = np.zeros((len_target_seq,vocab_size))
        msa_prior[MSA_start:MSA_end,:]=one_hots_avg
    else:
        msa_prior = np.ones((len_target_seq,vocab_size)) / vocab_size
    
    if verbose:
        for idx, position in enumerate(msa_prior):
            if len(position)!=25:
                print("Size error")
            if not round(position.sum(),2)==1.0:
                print("Position at index {} does not add up to 1: {}".format(idx, position.sum()))
    
    return msa_prior


def update_retrieved_MSA_log_prior_indel(model, MSA_log_prior, MSA_start, MSA_end, mutated_sequence):
    """
    Function to process MSA when scoring indels.
    To identify positions to add / remove in the retrieved MSA, we append and align the sequence to be scored to the original MSA for that protein family with Clustal Omega.
    If the original MSA is relatively deep (over 100k sequences), we sample (by default) 100k rows at random from that MSA to speed computations.
    MSA sampling is performed only once (for the first sequence to be scored). Subsequent scoring use the same MSA sample.
    """
    if not os.path.isdir(model.MSA_folder + os.sep + "Sampled"):
        os.mkdir(model.MSA_folder + os.sep + "Sampled")
    sampled_MSA_location = model.MSA_folder + os.sep + "Sampled" + os.sep + "Sampled_" + model.MSA_filename.split(os.sep)[-1]
    
    if not os.path.exists(sampled_MSA_location):
        msa_data = process_msa_data(model.MSA_filename)
        msa_data_sampled = filter_msa(msa_data, num_sequences_kept=100000) #If MSA has less than 100k sequences, the sample is identical to original MSA
        with open(sampled_MSA_location, 'w') as sampled_write_location:
            for index, key in enumerate(msa_data_sampled):
                key_name = ">REFERENCE_SEQUENCE" if index==0 else key
                msa_data_sampled[key] = msa_data_sampled[key].upper()
                msa_data_sampled[key] = msa_data_sampled[key].replace(".","-")
                sampled_write_location.write(key_name+"\n"+"\n".join([msa_data_sampled[key][i:i+80] for i in range(0, len(msa_data_sampled[key]), 80)])+"\n")
    
    seq_to_align_location = model.MSA_folder + os.sep + "Sampled" + os.sep + "Seq_to_align_" + model.MSA_filename.split(os.sep)[-1]
    sequence_text_split = [mutated_sequence[i:i+80] for i in range(0, len(mutated_sequence), 80)]
    sequence_text_split_split_join = "\n".join([">SEQ_TO_SCORE"]+sequence_text_split)
    os.system("echo '"+sequence_text_split_split_join+"' > "+seq_to_align_location)
    
    expanded_MSA_location = model.MSA_folder + os.sep + "Sampled" + os.sep + "Expanded_" + model.MSA_filename.split(os.sep)[-1]
    command = [
        model.config.clustal_omega_location,
        "--profile1", sampled_MSA_location,
        "--profile2", seq_to_align_location,
        "--outfile", expanded_MSA_location,
        "--force"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error in running ClustalOmega alignment:")
        print(result.stderr)
    msa_data = process_msa_data(expanded_MSA_location)
    aligned_seqA, aligned_seqB = msa_data[">SEQ_TO_SCORE"], msa_data[">REFERENCE_SEQUENCE"]
    try:
        keep_column=[]
        for column_index_pairwise_alignment in range(len(aligned_seqA)):
            if aligned_seqA[column_index_pairwise_alignment]=="-" and aligned_seqB[column_index_pairwise_alignment]=="-":
                continue
            elif aligned_seqA[column_index_pairwise_alignment]=="-":
                keep_column.append(False)
            elif aligned_seqB[column_index_pairwise_alignment]=="-":
                MSA_log_prior=torch.cat((MSA_log_prior[:column_index_pairwise_alignment], torch.zeros(MSA_log_prior.shape[1]).view(1,-1).cuda(), MSA_log_prior[column_index_pairwise_alignment:]),dim=0)
                keep_column.append(True) #keep the zero column we just added
            else:
                keep_column.append(True)
        MSA_log_prior = MSA_log_prior[keep_column]
        MSA_end = MSA_start + len(MSA_log_prior)
    except: 
        print("Error when processing the following alignment: {}".format(expanded_MSA_location))
    return MSA_log_prior, MSA_start, MSA_end