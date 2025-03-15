import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
import copy
from Bio import SeqIO
from functools import partial

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.data.sampler import SequentialSampler
import torch.nn.functional as F
from transformers import DataCollatorForLanguageModeling

from proteinnpt.utils.tranception.utils.scoring_utils import get_sequence_slices
from proteinnpt.utils.msa_utils import weighted_sample_MSA, align_new_sequences_to_msa
from proteinnpt.utils.data_utils import collate_fn_protein_npt, slice_sequences

def get_ESM_dataloader(df, batch_size):
    dataset_dict = {}
    dataset_dict['mutant_mutated_seq_pairs'] = list(zip(list(df['mutant']),list(df['mutated_sequence'])))
    dataset = Dataset.from_dict(dataset_dict)
    dataloader = DataLoader(
                    dataset=dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    num_workers=0, 
                    pin_memory=True,
                    collate_fn=collate_fn_protein_npt
                )
    return dataloader

def get_Tranception_dataloader(df, batch_size, model, target_seq, indel_mode, slice=True):
    if slice:
        sliced_df = get_sequence_slices(df, 
            target_seq=target_seq, 
            model_context_len = model.config.n_ctx - 2, 
            indel_mode=indel_mode, 
            scoring_window="optimal"
        )
    else:
        sliced_df = df
        sliced_df["sliced_mutated_sequence"] = df["mutated_sequence"]
    mutant_index=0
    ds = Dataset.from_pandas(sliced_df)
    ds.set_transform(model.encode_batch)
    data_collator = DataCollatorForLanguageModeling(
                        tokenizer=model.config.tokenizer,
                        mlm=False)
    sampler = SequentialSampler(ds)
    dataloader = torch.utils.data.DataLoader(
        ds, 
        batch_size=batch_size, 
        sampler=sampler, 
        collate_fn=data_collator, 
        num_workers=0, 
        pin_memory=True, 
        drop_last=False
    )
    return dataloader

def fast_MSA_mode_setup(model, seqs_to_score, MSA_sequences, MSA_weights, num_MSA_sequences, path_to_clustalomega):
    model.MSA_sample_sequences = weighted_sample_MSA(
                MSA_all_sequences=MSA_sequences, 
                MSA_non_ref_sequences_weights=MSA_weights, 
                number_sampled_MSA_sequences=num_MSA_sequences
            )
    fast_MSA_short_names_mapping = {}
    fast_MSA_short_names = []
    #print("There are {} sequences to score".format(len(seqs_to_score)))
    for seq_index,seq in enumerate(list(seqs_to_score)):
        short_name = 'mutant_to_score_'+str(seq_index)
        fast_MSA_short_names_mapping[seq] = short_name
        fast_MSA_short_names.append(short_name)
    fast_MSA_aligned_sequences = align_new_sequences_to_msa(model.MSA_sample_sequences, list(seqs_to_score), fast_MSA_short_names, clustalomega_path=path_to_clustalomega)
    model.MSA_sample_sequences = [tup for tup in fast_MSA_aligned_sequences if tup[0] not in set(fast_MSA_short_names)]
    return model, fast_MSA_aligned_sequences, fast_MSA_short_names_mapping

def trim_batch_index_vec(mutant_mutated_seq_pairs,start_index,end_index):
    mutant_mutated_seq_pairs_copy = copy.deepcopy(mutant_mutated_seq_pairs)
    mutant_mutated_seq_pairs_copy = [(mutant,seq[start_index:end_index]) for (mutant,seq) in mutant_mutated_seq_pairs_copy]
    return mutant_mutated_seq_pairs_copy

def process_MSA_Transformer_batch(batch, model, alphabet, max_positions, MSA_start_position, MSA_end_position, MSA_weights, MSA_sequences, num_MSA_sequences, eval_mode, start_idx=1, long_sequences_slicing_method="left", indel_mode=False, fast_MSA_mode=False, clustalomega_path=None, batch_target_labels=None, batch_masked_targets=None, target_names=None, num_extra_tokens=1, slicing_long_sequences=False, slice_start=0, slice_end=1024):
    raw_sequence_length = len(batch['mutant_mutated_seq_pairs'][0][1]) #Selects the first mutant-seq pair, and the sequence is 1-indexed in that pair
    raw_batch_size = len(batch['mutant_mutated_seq_pairs']) # Effective number of mutated sequences to be scored (could be lower than num_MSA_sequences in practice, eg., if we're at the end of the train_iterator)
    if (MSA_start_position is not None) and (MSA_end_position is not None) and ((MSA_start_position > 1) or (MSA_end_position < raw_sequence_length)):
        MSA_start_index = MSA_start_position - 1
        MSA_end_index = MSA_end_position
        batch['mutant_mutated_seq_pairs'] = [ (mutant,seq[MSA_start_index:MSA_end_index]) for (mutant,seq) in batch['mutant_mutated_seq_pairs']]
        # Recompute sequence length (has potentially been chopped off above)
        raw_sequence_length = len(batch['mutant_mutated_seq_pairs'][0][1])
    #Sample MSA sequences based on sequence weights        
    assert MSA_weights is not None, "Trying to add MSA_sequences to scoring batch but no weights are provided"
    if model.MSA_sample_sequences is None:
        model.MSA_sample_sequences = weighted_sample_MSA(
            MSA_all_sequences=MSA_sequences, 
            MSA_non_ref_sequences_weights=MSA_weights, 
            number_sampled_MSA_sequences=num_MSA_sequences
        )
    num_MSA_sequences = min(num_MSA_sequences, len(MSA_sequences)) #Effective number of MSA sequences kept
    # Slice MSA sequences.
    if slicing_long_sequences:
        trimmed_MSA_sequences = trim_batch_index_vec(model.MSA_sample_sequences,slice_start, slice_end)
    else:
        trimmed_MSA_sequences = model.MSA_sample_sequences

    # Concatenate MSA sequences with labelled assay sequences
    batch['mutant_mutated_seq_pairs'] += trimmed_MSA_sequences
    if max_positions is not None and raw_sequence_length + 1 > max_positions: # Adding one for the BOS token
        if long_sequences_slicing_method=="center":
            #print("Center slicing method not adapted to PNPT embeddings as sequences would not be aligned in the same system anymore. Defaulting to 'left' mode.")
            long_sequences_slicing_method="left"
        batch['mutant_mutated_seq_pairs'], batch_target_labels, batch_masked_targets, batch_scoring_optimal_window = slice_sequences(
            list_mutant_mutated_seq_pairs = batch['mutant_mutated_seq_pairs'], 
            max_positions=max_positions,
            method=long_sequences_slicing_method,
            rolling_overlap=max_positions//4,
            eval_mode=eval_mode,
            batch_target_labels=batch_target_labels,
            batch_masked_targets=batch_masked_targets, 
            start_idx=start_idx,
            target_names=target_names,
            num_extra_tokens=num_extra_tokens
        )
    #Re-organize list of sequences to have training_num_assay_sequences_per_batch_per_gpu MSA batches, where in each the sequence to score is the first and the rest are the sampled MSA sequences.
    num_sequences = raw_batch_size + num_MSA_sequences
    assert len(batch['mutant_mutated_seq_pairs']) == num_sequences, f"Unexpected number of sequences ({len(batch['mutant_mutated_seq_pairs'])} vs {num_sequences})"
    sequences_to_score = batch['mutant_mutated_seq_pairs'][:raw_batch_size]
    MSA_sequences = batch['mutant_mutated_seq_pairs'][raw_batch_size:]
    
    if fast_MSA_mode:
        mutants_to_score, seqs_to_score = zip(*sequences_to_score)
        model, fast_MSA_aligned_sequences, fast_MSA_short_names_mapping = fast_MSA_mode_setup(model, seqs_to_score, MSA_sequences, MSA_weights, num_MSA_sequences, clustalomega_path)
        MSA_sequences = model.MSA_sample_sequences # Need to pull sequences re-aligned in new coordinate system
    else:
        fast_MSA_aligned_sequences = None
        fast_MSA_short_names_mapping = None
    
    if indel_mode and not fast_MSA_mode:
        #Assume batch size of 1
        assert len(sequences_to_score)==1, "With MSA Transformer, batch size should be 1 if fast_MSA_mode is not activated"
        mutant_to_score, new_sequence = sequences_to_score[0]
        MSA_sequences = align_new_sequences_to_msa(MSA_sequences, [new_sequence], ["mutant_to_score"], clustalomega_path=clustalomega_path)
        sequences_to_score = [(mutant_to_score,tup[1]) for tup in MSA_sequences if tup[0] == "mutant_to_score"] #Have to resort to this otherwise "mutant_to_score" is truncated
        MSA_sequences = [tup for tup in MSA_sequences if tup[0] != "mutant_to_score"]
    elif indel_mode and fast_MSA_mode:
        seq_scoring = []
        for sequence in sequences_to_score:
            seq_short_name = fast_MSA_short_names_mapping[sequence[1]]
            score_tuple = [(sequence[0],tup[1]) for tup in fast_MSA_aligned_sequences if tup[0] == seq_short_name]
            assert len(score_tuple)==1
            seq_scoring.append(score_tuple[0])
        sequences_to_score = seq_scoring
    
    if fast_MSA_mode:
        batch['mutant_mutated_seq_pairs'] = [sequences_to_score + MSA_sequences]
    else:
        batch['mutant_mutated_seq_pairs'] = [ [sequence] + MSA_sequences for sequence in sequences_to_score]
    
    token_batch_converter = alphabet.get_batch_converter()
    batch_sequence_names, batch_AA_sequences, batch_token_sequences = token_batch_converter(batch['mutant_mutated_seq_pairs'])    
    batch_token_sequences = batch_token_sequences.to(next(model.parameters()).device)
    processed_batch = {
        'input_tokens': batch_token_sequences,
        'mutant_mutated_seq_pairs': batch['mutant_mutated_seq_pairs']
    }
    if batch_target_labels is not None: processed_batch['target_labels'] = batch_target_labels
    if batch_masked_targets is not None: processed_batch['masked_targets'] = batch_masked_targets
    return processed_batch

def process_MSA_Transformer_batch_multiple_MSAs(batch, model, alphabet, MSA_folder, num_MSA_sequences, batch_target_labels=None, batch_masked_targets=None):
    """
    There is a unique MSA used for each input sequence. Does not have all details from MSA processing "single MSA" yet.
    We also do not worry about sequence weights to keep things computationally tractable.
    Assume a batch size of 1 --> mutant[0]
    """
    mutant, sequence = zip(*batch['mutant_mutated_seq_pairs'])
    MSA_sequences = [
            (record.description, str(record.seq)) for record in SeqIO.parse(MSA_folder + os.sep + mutant[0] + ".a3m", "fasta")
    ]
    model.MSA_sample_sequences = weighted_sample_MSA(
            MSA_all_sequences=MSA_sequences, 
            MSA_non_ref_sequences_weights=None, 
            number_sampled_MSA_sequences=num_MSA_sequences
    )
    batch['mutant_mutated_seq_pairs'] = [ [batch['mutant_mutated_seq_pairs']] + model.MSA_sample_sequences ]
    token_batch_converter = alphabet.get_batch_converter()
    batch_sequence_names, batch_AA_sequences, batch_token_sequences = token_batch_converter(batch['mutant_mutated_seq_pairs'])    
    batch_token_sequences = batch_token_sequences.to(next(model.parameters()).device)
    processed_batch = {
        'input_tokens': batch_token_sequences,
        'mutant_mutated_seq_pairs': batch['mutant_mutated_seq_pairs']
    }
    if batch_target_labels is not None: processed_batch['target_labels'] = batch_target_labels
    if batch_masked_targets is not None: processed_batch['masked_targets'] = batch_masked_targets
    return processed_batch

def pad_sequences_to_length(pairs, length):
        padded_pairs = []
        for mutant_id, sequence in pairs:
            if len(sequence) < length:
                sequence = sequence + '-' * (length - len(sequence))
            padded_pairs.append((mutant_id, sequence))
        return padded_pairs

def process_ESM_batch(batch, model, alphabet, max_positions, long_sequences_slicing_method, eval_mode, start_idx=1, batch_target_labels=None, target_names=None, num_extra_tokens=1):
    #raw_sequence_length = len(batch['mutant_mutated_seq_pairs'][0][1]) #Selects the first mutant-seq pair, and the sequence is 1-indexed in that pair
    raw_sequence_length = max(len(seq) for _, seq in batch['mutant_mutated_seq_pairs'])
    batch_size = len(batch['mutant_mutated_seq_pairs'])
    if max_positions is not None and raw_sequence_length + 1 > max_positions: # Adding one for the BOS token
        batch['mutant_mutated_seq_pairs'], batch_target_labels, _, _ = slice_sequences(
                list_mutant_mutated_seq_pairs = batch['mutant_mutated_seq_pairs'], 
                max_positions=max_positions,
                method=long_sequences_slicing_method,
                rolling_overlap=max_positions//4,
                eval_mode=eval_mode,
                batch_target_labels=batch_target_labels, 
                start_idx=start_idx,
                target_names=target_names,
                num_extra_tokens=num_extra_tokens
            )
        raw_sequence_length = max(len(seq) for _, seq in batch['mutant_mutated_seq_pairs']) #Updated potentially
    batch['mutant_mutated_seq_pairs'] = pad_sequences_to_length(batch['mutant_mutated_seq_pairs'], raw_sequence_length)
    token_batch_converter = alphabet.get_batch_converter()
    batch_sequence_names, batch_AA_sequences, batch_token_sequences = token_batch_converter(batch['mutant_mutated_seq_pairs'])    
    batch_token_sequences = batch_token_sequences.to(next(model.parameters()).device)
    batch_token_sequences = batch_token_sequences.view(batch_size,-1) #squeeze()
    processed_batch = {
        'input_tokens': batch_token_sequences,
        'mutant_mutated_seq_pairs': batch['mutant_mutated_seq_pairs']
    }
    return processed_batch

def process_Tranception_batch(batch, model):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(model.device)
    return batch

def get_embeddings_MSA_Transformer(model, processed_batch, alphabet_size, return_pseudo_ll=True, fast_MSA_mode=False):
    tokens = processed_batch['input_tokens']
    assert tokens.ndim == 3, "Finding dimension of tokens to be: {}".format(tokens.ndim)
    num_MSAs_in_batch, num_sequences_in_alignments, seqlen = tokens.size()
    batch_size = num_MSAs_in_batch if not fast_MSA_mode else num_sequences_in_alignments - len(model.MSA_sample_sequences)
    output = model(tokens, repr_layers=[12])
    embeddings = output["representations"][12][:] # B, N, L, D
    if fast_MSA_mode: # A single batch with multiple sequences to score in it to speed things up
        embeddings = embeddings[:,:batch_size,:,:] # 1, N, L, D 
        embeddings = [embeddings[:,i,:,:] for i in range(batch_size)]
        embeddings = torch.cat(embeddings)
        if return_pseudo_ll:
            logits = output["logits"][:,:batch_size].view(batch_size,-1,alphabet_size)
            tokens = tokens[:,:batch_size].view(batch_size,-1)
            pseudo_ll = - CrossEntropyLoss(reduction='none', label_smoothing=0.0)(logits.reshape(-1, alphabet_size), tokens.reshape(-1)).view(batch_size, seqlen)
            pseudo_ll = pseudo_ll.mean(dim=-1).view(-1).cpu()
            pseudo_ll = [psll.item() for psll in pseudo_ll]
        else:
            pseudo_ll = None
        if 'mutant_mutated_seq_pairs' in processed_batch: processed_batch['mutant_mutated_seq_pairs'] = processed_batch['mutant_mutated_seq_pairs'][0][:batch_size]
    else:
        embeddings = embeddings[:,0,:,:] # In each MSA batch the first sequence is what we care about. The other MSA sequences were present just to compute embeddings and logits
        if return_pseudo_ll:
            logits = output["logits"][:,0] # Filtering logits of points to score
            tokens = tokens[:,0] # Filtering tokens of points to score
            pseudo_ll = - CrossEntropyLoss(reduction='none', label_smoothing=0.0)(logits.reshape(-1, alphabet_size), tokens.reshape(-1)).view(num_MSAs_in_batch, seqlen)
            pseudo_ll = pseudo_ll.mean(dim=-1).view(-1).cpu() #Average across sequence length
            pseudo_ll = [psll.item() for psll in pseudo_ll]
        else:
            pseudo_ll = None
        if 'mutant_mutated_seq_pairs' in processed_batch: processed_batch['mutant_mutated_seq_pairs'] = [seq[0] for seq in processed_batch['mutant_mutated_seq_pairs']] # Remove MSA sequences from batch by selecting the first sequence in each MSA input sets
    return embeddings, pseudo_ll, processed_batch

def get_embeddings_ESM(model, model_type, processed_batch, alphabet_size, return_pseudo_ll=True):
    tokens = processed_batch['input_tokens']
    assert tokens.ndim == 2, "Finding dimension of tokens to be: {}".format(tokens.ndim)
    batch_size, seqlen = tokens.size()
    if model_type.startswith("ESM1v"):
        last_layer_index = 33
    elif model_type.startswith("ESM2_15B"):
        last_layer_index = 48
    elif model_type.startswith("ESM2_3B"):
        last_layer_index = 36
    elif model_type.startswith("ESM2_650M"):
        last_layer_index = 33
    output = model(tokens, repr_layers=[last_layer_index])
    embeddings = output["representations"][last_layer_index][:] # N, L, D
    if return_pseudo_ll:
        logits = output["logits"]
        pseudo_ll = - CrossEntropyLoss(reduction='none', label_smoothing=0.0)(logits.view(-1, alphabet_size), tokens.view(-1)).view(batch_size, seqlen)
        pseudo_ll = pseudo_ll.mean(dim=-1).view(-1).cpu()
        pseudo_ll = [psll.item() for psll in pseudo_ll]
    else:
        pseudo_ll = None
    return embeddings, pseudo_ll

def get_embeddings_Tranception(model, processed_batch, return_pseudo_ll=True):
    output = model(**processed_batch, return_dict=True, output_hidden_states=True)
    embeddings = output.hidden_states[-1] # Extract embeddings from last layer
    if return_pseudo_ll:
        logits = output.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = processed_batch['labels'][..., 1:].contiguous()
        pseudo_ll = - CrossEntropyLoss(reduction='none')(input=shift_logits.view(-1, shift_logits.size(-1)), target=shift_labels.view(-1)).view(shift_logits.shape[0],shift_logits.shape[1])
        pseudo_ll = pseudo_ll.mean(dim=-1).view(-1).cpu() # Shape (B,)
        pseudo_ll = [psll.item() for psll in pseudo_ll]
    else:
        pseudo_ll = None
    return embeddings, pseudo_ll