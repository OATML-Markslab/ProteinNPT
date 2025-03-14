import sys
from collections import defaultdict
import numpy as np
import h5py
import torch
from ..utils.data_utils import get_indices_retrieved_embeddings
from ..utils.embedding_utils import process_MSA_Transformer_batch, process_ESM_batch, process_Tranception_batch

def process_batch(batch, model, alphabet, args, device, MSA_sequences=None, MSA_weights=None, MSA_start_position=None, MSA_end_position=None, eval_mode=True, indel_mode=False, start_idx=1):
    """Processes a batch of sequences for baseline model input.

    Handles various aspects of sequence processing including:
    - Loading and processing pre-computed embeddings
    - MSA sequence processing and sampling
    - Sequence length adjustment and slicing
    - Model-specific tokenization (Tranception, ESM, MSA Transformer)
    - Special handling for insertions/deletions in sequences

    Args:
        batch (dict): Input batch containing:
            - mutant_mutated_seq_pairs: List of (mutant, sequence) pairs
            - Additional target values as specified in args.target_config
        model: The baseline model instance
        alphabet: Tokenizer alphabet for protein sequences
        args: Configuration object containing model parameters
        device: PyTorch device (CPU/GPU)
        MSA_sequences (list, optional): List of MSA sequences
        MSA_weights (numpy.ndarray, optional): Weights for MSA sequence sampling
        MSA_start_position (int, optional): Start position for MSA sequence slice
        MSA_end_position (int, optional): End position for MSA sequence slice
        eval_mode (bool, optional): Whether to run in evaluation mode. Defaults to True
        indel_mode (bool, optional): Whether to handle insertions/deletions. Defaults to False
        start_idx (int, optional): One-indexed position of first residue. Defaults to 1

    Returns:
        dict: Processed batch containing:
            - input_tokens: Tokenized sequences
            - target_labels: Target values for prediction
            - mutant_mutated_seq_pairs: Original sequence pairs
            - sequence_embeddings: Pre-computed embeddings if available
    """
    target_names = args.target_config.keys()
    raw_sequence_length = len(batch['mutant_mutated_seq_pairs'][0][1]) 
    raw_batch_size = len(batch['mutant_mutated_seq_pairs']) 

    if args.sequence_embeddings_location is not None and args.aa_embeddings!="One_hot_encoding":
        try:
            sequence_embeddings_list = []
            for embedding_location in args.sequence_embeddings_location:
                indices_retrieved_embeddings = get_indices_retrieved_embeddings(batch,embedding_location)
                assert len(indices_retrieved_embeddings)==raw_batch_size, "At least one embedding was missing"
                with h5py.File(embedding_location, 'r') as h5f:
                    sequence_embeddings = torch.tensor(np.array([h5f['embeddings'][i] for i in indices_retrieved_embeddings])).float()
                sequence_embeddings_list.append(sequence_embeddings)
            if len(sequence_embeddings_list)==1:
                sequence_embeddings = sequence_embeddings_list[0]
            else:
                sequence_embeddings = torch.cat(sequence_embeddings_list,dim=-1) #concatenate embeddings over the last dimension --> B, L, D1 + D2
        except:
            print("Error loading main sequence embeddings")
            sys.exit(0)
    else:
        sequence_embeddings = None

    batch_target_labels = defaultdict(list)
    for target_name in target_names: batch_target_labels[target_name] = batch[target_name].to(device)

    if args.augmentation=="zero_shot_fitness_predictions_covariate": 
        batch_target_labels['zero_shot_fitness_predictions'] = batch['zero_shot_fitness_predictions'].to(device) 

    if args.aa_embeddings=="MSA_Transformer" and sequence_embeddings is None:
        processed_batch = process_MSA_Transformer_batch(
                    batch = batch, 
                    model = model.aa_embedding, 
                    alphabet = alphabet, 
                    max_positions = args.max_positions, 
                    MSA_start_position = MSA_start_position, 
                    MSA_end_position = MSA_end_position, 
                    MSA_weights = MSA_weights, 
                    MSA_sequences = MSA_sequences,
                    num_MSA_sequences = args.num_MSA_sequences_per_training_instance,
                    eval_mode = eval_mode, 
                    start_idx = start_idx, 
                    long_sequences_slicing_method = args.long_sequences_slicing_method, 
                    indel_mode = indel_mode, 
                    fast_MSA_mode = args.fast_MSA_mode, 
                    clustalomega_path = args.clustalomega_path, 
                    batch_target_labels = batch_target_labels,
                    target_names = target_names,
                    num_extra_tokens=1
                    )
        if sequence_embeddings is not None:
            # Drop unnecessary dimension if embedding model is MSAT and embeddings are retrieved from disk
            num_MSAs_in_batch, num_sequences_in_alignments, seqlen = processed_batch['input_tokens'].size()
            processed_batch['input_tokens'] = processed_batch['input_tokens'].view(num_sequences_in_alignments, seqlen)
    elif args.aa_embeddings=="Tranception":
        processed_batch = process_Tranception_batch(
                    batch = batch, 
                    model = model.aa_embedding
                    )
        mutant, sequence = zip(*batch['mutant_mutated_seq_pairs'])
        tokenized_batch = model.config.tokenizer(sequence, add_special_tokens=True, truncation=True, padding=True, max_length=model.config.n_ctx)
        for k, v in tokenized_batch.items():
            processed_batch[k] = torch.tensor(v).to(model.device)
        processed_batch['input_tokens']=processed_batch['input_ids']
    else:
        processed_batch = process_ESM_batch(
                    batch = batch, 
                    model = model.aa_embedding, 
                    alphabet = alphabet, 
                    max_positions = args.max_positions, 
                    long_sequences_slicing_method = args.long_sequences_slicing_method, 
                    eval_mode = eval_mode, 
                    start_idx = start_idx,
                    num_extra_tokens = 2
                    )
    if 'target_labels' not in processed_batch.keys(): processed_batch['target_labels'] = batch_target_labels # Slicing within process_MSA_Transformer_batch may have included a sliced version of batch_target_labels already
    processed_batch['sequence_embeddings'] = sequence_embeddings
    return processed_batch
