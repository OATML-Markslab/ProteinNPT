import sys
from collections import defaultdict
import numpy as np
import h5py
import torch
from ..utils.data_utils import slice_sequences, get_indices_retrieved_embeddings
from ..utils.msa_utils import weighted_sample_MSA

def process_batch(batch, model, alphabet, args, device, MSA_sequences=None, MSA_weights=None, MSA_start_position=None, MSA_end_position=None, eval_mode = True, indel_mode=False, start_idx=1):
    """
    start_idx is the one-indexed postion of the first residue in the sequence. If full sequence is passed (as always assumed in this codebase) this is equal to 1.
    """
    target_names = args.target_config.keys()
    raw_sequence_length = len(batch['mutant_mutated_seq_pairs'][0][1]) 
    raw_batch_size = len(batch['mutant_mutated_seq_pairs']) 

    if args.sequence_embeddings_location is not None and args.aa_embeddings!="One_hot_encoding":
        try:
            indices_retrieved_embeddings = get_indices_retrieved_embeddings(batch,args.sequence_embeddings_location)
            assert len(indices_retrieved_embeddings)==raw_batch_size, "At least one embedding was missing"
            with h5py.File(args.sequence_embeddings_location, 'r') as h5f:
                sequence_embeddings = torch.tensor(np.array([h5f['embeddings'][i] for i in indices_retrieved_embeddings])).float()
        except:
            print("Error loading main sequence embeddings")
            sys.exit(0)
    else:
        sequence_embeddings = None

    batch_target_labels = defaultdict(list)
    for target_name in target_names: batch_target_labels[target_name] = batch[target_name].to(device)

    if args.augmentation=="zero_shot_fitness_predictions_covariate": batch_target_labels['zero_shot_fitness_predictions'] = batch['zero_shot_fitness_predictions'].to(device) 

    if args.aa_embeddings=="MSA_Transformer":
        # If MSAT and MSA does not cover full sequence length, we chop off all sequences to be scored as needed so that everything lines up properly.
        if (MSA_start_position is not None) and (MSA_end_position is not None) and ((MSA_start_position > 1) or (MSA_end_position < raw_sequence_length)) and args.sequence_embeddings_location is None:
            MSA_start_index = MSA_start_position - 1
            MSA_end_index = MSA_end_position
            batch['mutant_mutated_seq_pairs'] = [ (mutant,seq[MSA_start_index:MSA_end_index]) for (mutant,seq) in batch['mutant_mutated_seq_pairs']]
            # Recompute sequence length (has potentially been chopped off above)
            raw_sequence_length = len(batch['mutant_mutated_seq_pairs'][0][1])
        
        #Sample MSA sequences as needed 
        if args.sequence_embeddings_location is None and args.num_MSA_sequences_per_training_instance > 0:
            assert MSA_weights is not None, "Trying to add MSA_sequences to scoring batch but no weights are provided"
            if model.MSA_sample_sequences is None:
                model.MSA_sample_sequences = weighted_sample_MSA(
                    MSA_all_sequences=MSA_sequences, 
                    MSA_non_ref_sequences_weights=MSA_weights, 
                    number_sampled_MSA_sequences=args.num_MSA_sequences_per_training_instance
                )
            # Concatenate MSA sequences with labelled assay sequences
            batch['mutant_mutated_seq_pairs'] += model.MSA_sample_sequences
        
    # Slice sequences around mutation if sequence longer than context length
    if args.max_positions is not None and raw_sequence_length + 1 > args.max_positions and args.sequence_embeddings_location is None: # Adding one for the BOS token
        if args.long_sequences_slicing_method=="center" and args.aa_embeddings=="MSA_Transformer":
            print("Center slicing method not adapted to MSA Transformer embedding as sequences would not be aligned in the same coordinate system anymore. Defaulting to 'left' mode.")
            args.long_sequences_slicing_method="left"
        batch['mutant_mutated_seq_pairs'], batch_target_labels, _ = slice_sequences(
            list_mutant_mutated_seq_pairs = batch['mutant_mutated_seq_pairs'], 
            max_positions=args.max_positions,
            method=args.long_sequences_slicing_method,
            rolling_overlap=args.max_positions//4,
            eval_mode=eval_mode,
            batch_target_labels=batch_target_labels,
            start_idx=start_idx,
            target_names=target_names,
            num_extra_tokens=2 if args.aa_embeddings=="Tranception" else 1
        )
    
    # Tokenize input sequences with Tranception or ESM tokenizer
    if args.aa_embeddings=="Tranception":
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(model.device)
        processed_batch = batch
        mutant, sequence = zip(*batch['mutant_mutated_seq_pairs'])
        tokenized_batch=model.config.tokenizer(sequence, add_special_tokens=True, truncation=True, padding=True, max_length=model.config.n_ctx)
        for k, v in tokenized_batch.items():
            processed_batch[k] = torch.tensor(v).to(model.device)
        processed_batch['input_tokens']=processed_batch['input_ids']
        processed_batch['sequence_embeddings'] = sequence_embeddings
        processed_batch['target_labels'] = batch_target_labels
    else:
        if args.aa_embeddings == "MSA_Transformer" and args.training_num_assay_sequences_per_batch_per_gpu > 1 and args.sequence_embeddings_location is None: 
            #Re-organize list of sequences to have training_num_assay_sequences_per_batch_per_gpu MSA batches, where in each the sequence to score is the first and the rest are the sampled MSA sequences.
            num_sequences = raw_batch_size + args.num_MSA_sequences_per_training_instance
            assert len(batch['mutant_mutated_seq_pairs']) == num_sequences, "Unexpected number of sequences"
            sequences_to_score = batch['mutant_mutated_seq_pairs'][:raw_batch_size]
            MSA_sequences = batch['mutant_mutated_seq_pairs'][raw_batch_size:]
            batch['mutant_mutated_seq_pairs'] = [ [sequence] + MSA_sequences for sequence in sequences_to_score]
            
        token_batch_converter = alphabet.get_batch_converter()
        if indel_mode:
            mutants, mutated_seqs = list(zip(*batch['mutant_mutated_seq_pairs']))
            max_seq_length = max(len(s) for s in mutated_seqs)
            mutated_seqs_padded = [s.ljust(max_seq_length, "-") for s in mutated_seqs]
            batch['mutant_mutated_seq_pairs'] = list(zip(mutants,mutated_seqs_padded))
        batch_sequence_names, batch_AA_sequences, batch_token_sequences = token_batch_converter(batch['mutant_mutated_seq_pairs'])        
            
        if args.aa_embeddings=="MSA_Transformer" and args.sequence_embeddings_location is not None:
            # Drop unnecessary dimension if embedding model is MSAT and embeddings are retrieved from disk
            num_MSAs_in_batch, num_sequences_in_alignments, seqlen = batch_token_sequences.size()
            batch_token_sequences = batch_token_sequences.view(num_sequences_in_alignments, seqlen)
            
        batch_token_sequences = batch_token_sequences.to(device)
        processed_batch = {
            'input_tokens': batch_token_sequences,
            'target_labels': batch_target_labels,
            'mutant_mutated_seq_pairs': batch['mutant_mutated_seq_pairs'],
            'sequence_embeddings': sequence_embeddings
        }
    return processed_batch
