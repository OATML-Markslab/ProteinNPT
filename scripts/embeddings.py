import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import argparse
import h5py
import json
import tqdm

from torch.utils.data.sampler import SequentialSampler
from transformers import DataCollatorForLanguageModeling

from proteinnpt.utils.tranception.model_pytorch import get_tranception_tokenizer
from proteinnpt.utils.tranception.config import TranceptionConfig
from proteinnpt.utils.tranception.model_pytorch import TranceptionLMHeadModel
from proteinnpt.utils.tranception.utils.scoring_utils import get_sequence_slices
from proteinnpt.utils.esm.data import Alphabet
from proteinnpt.utils.esm.pretrained import load_model_and_alphabet
from proteinnpt.utils.msa_utils import weighted_sample_MSA, process_MSA, align_new_sequences_to_msa
from proteinnpt.utils.data_utils import collate_fn_protein_npt, slice_sequences


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract embeddings')
    parser.add_argument('--assay_reference_file_location', default=None, type=str, help='Path to reference file with list of assays to score')
    parser.add_argument('--assay_index', default=0, type=int, help='Index of assay in the ProteinGym reference file to compute embeddings for')
    parser.add_argument('--input_data_location', default=None, type=str, help='Location of input data with all mutated sequences [If a reference file is used, this is the location where are assays are stored. If a single csv file is passed, this is the full path to that assay data]')
    parser.add_argument('--output_data_location', default=None, type=str, help='Location of output embeddings')
    parser.add_argument('--model_type', default='Tranception', type=str, help='Model type to compute embeddings with')
    parser.add_argument('--model_location', default=None, type=str, help='Location of model used to embed protein sequences')
    parser.add_argument('--max_positions', default=1024, type=int, help='Maximum context length of embedding model')
    parser.add_argument('--long_sequences_slicing_method', default='center', type=str, help='Method to slice long sequences [rolling, center, left]')
    parser.add_argument('--batch_size', default=32, type=int, help='Eval batch size')
    parser.add_argument('--indel_mode', action='store_true', help='Use this mode if extracting embeddings of indel assays')
    parser.add_argument('--half_precision', action='store_true', help='Store embeddings as 16-bit floating point numbers (float16)')
    #MSA specific parameters (only relevant for MSA Transformer)
    parser.add_argument('--num_MSA_sequences', default=1000, type=int, help='Num MSA sequences to score each sequence with')
    parser.add_argument('--MSA_data_folder', default=None, type=str, help='Folder where all MSAs are stored')
    parser.add_argument('--MSA_weight_data_folder', default=None, type=str, help='Folder where MSA sequence weights are stored (for diversity sampling of MSAs)')
    parser.add_argument('--path_to_hhfilter', default=None, type=str, help='Path to hhfilter (for filtering MSAs)')
    parser.add_argument('--path_to_clustalomega', default=None, type=str, help='Path to clustal omega (to re-align sequences in indel assays) [indels only]')
    parser.add_argument('--fast_MSA_mode', action='store_true', help='Use this mode to speed up embedding extraction for MSA Transformer by scoring multiple mutated sequences (batch_size of them) at once (has minor impact on quality)')
    #If not using a reference file
    parser.add_argument('--target_seq', default=None, type=str, help='Wild type sequence mutated in the assay (to be provided if not using a reference file)')
    parser.add_argument('--MSA_location', default=None, type=str, help='Path to MSA file (.a2m)')
    parser.add_argument('--weight_file_name', default=None, type=str, help='Name of weight file')
    parser.add_argument('--MSA_start', default=None, type=int, help='Index of first AA covered by the MSA relative to target_seq coordinates (1-indexing)')
    parser.add_argument('--MSA_end', default=None, type=int, help='Index of last AA covered by the MSA relative to target_seq coordinates (1-indexing)')
    parser.add_argument("--use_cpu", action="store_true", help="Force the use of CPU instead of GPU (considerably slower). If this option is not chosen, the script will raise an error if the GPU is not available.")
    return parser.parse_args()

model_type_options = ["MSA_Transformer", "ESM1v", "Tranception"]

def process_embeddings_batch(batch, model, model_type, alphabet, device, max_positions, long_sequences_slicing_method, MSA_sequences=None, MSA_weights=None, MSA_start_position=None, MSA_end_position=None, num_MSA_sequences=None, eval_mode = True, start_idx=1, indel_mode=False, fast_MSA_mode=False, fast_MSA_aligned_sequences=None, fast_MSA_short_names_mapping=None, clustalomega_path=None):
    if model_type in ["MSA_Transformer","ESM1v"]:
        raw_sequence_length = len(batch['mutant_mutated_seq_pairs'][0][1]) #Selects the first mutant-seq pair, and the sequence is 1-indexed in that pair
        raw_batch_size = len(batch['mutant_mutated_seq_pairs']) # Effective number of mutated sequences to be scored (could be lower than num_MSA_sequences in practice, eg., if we're at the end of the train_iterator)
        # If model is MSAT and MSA does not cover full sequence length, we chop off all sequences to be scored as needed so that everything lines up properly.
        if (model_type == "MSA_Transformer") and (MSA_start_position is not None) and (MSA_end_position is not None) and ((MSA_start_position > 1) or (MSA_end_position < raw_sequence_length)):
            MSA_start_index = MSA_start_position - 1
            MSA_end_index = MSA_end_position
            batch['mutant_mutated_seq_pairs'] = [ (mutant,seq[MSA_start_index:MSA_end_index]) for (mutant,seq) in batch['mutant_mutated_seq_pairs']]
            # Recompute sequence length (has potentially been chopped off above)
            raw_sequence_length = len(batch['mutant_mutated_seq_pairs'][0][1])
        # Sample MSA sequences
        if model_type == "MSA_Transformer":
            assert MSA_weights is not None, "Trying to add MSA_sequences to scoring batch but no weights are provided"
            if model.MSA_sample_sequences is None:
                model.MSA_sample_sequences = weighted_sample_MSA(
                    MSA_all_sequences=MSA_sequences, 
                    MSA_non_ref_sequences_weights=MSA_weights, 
                    number_sampled_MSA_sequences=num_MSA_sequences
                )
            # Concatenate MSA sequences with labelled assay sequences
            batch['mutant_mutated_seq_pairs'] += model.MSA_sample_sequences
        # Slice sequences around mutation if sequence longer than context length
        if max_positions is not None and raw_sequence_length + 1 > max_positions: # Adding one for the BOS token
            if long_sequences_slicing_method=="center" and model_type in ["MSA_Transformer"]:
                print("Center slicing method not adapted to PNPT embeddings as sequences would not be aligned in the same system anymore. Defaulting to 'left' mode.")
                long_sequences_slicing_method="left"
            batch['mutant_mutated_seq_pairs'], batch_target_labels, _ = slice_sequences(
                list_mutant_mutated_seq_pairs = batch['mutant_mutated_seq_pairs'], 
                max_positions=max_positions,
                method=long_sequences_slicing_method,
                rolling_overlap=max_positions//4,
                eval_mode=eval_mode,
                batch_target_labels=None, 
                start_idx=start_idx,
                target_names=None
            )
        # Tokenize protein sequences
        if model_type=="MSA_Transformer": 
            #Re-organize list of sequences to have training_num_assay_sequences_per_batch_per_gpu MSA batches, where in each the sequence to score is the first and the rest are the sampled MSA sequences.
            num_sequences = raw_batch_size + num_MSA_sequences
            assert len(batch['mutant_mutated_seq_pairs']) == num_sequences, "Unexpected number of sequences"
            sequences_to_score = batch['mutant_mutated_seq_pairs'][:raw_batch_size]
            MSA_sequences = batch['mutant_mutated_seq_pairs'][raw_batch_size:]
            if indel_mode and not fast_MSA_mode: 
                mutant_to_score, new_sequence = sequences_to_score[0]
                MSA_sequences = align_new_sequences_to_msa(MSA_sequences, [new_sequence], ["mutant_to_score"], clustalomega_path=clustalomega_path)
                sequences_to_score = [(mutant_to_score,tup[1]) for tup in MSA_sequences if tup[0] == "mutant_to_score"] #Have to resort to this otherwise "mutant_to_score" is truncated
                MSA_sequences = [tup for tup in MSA_sequences if tup[0] != "mutant_to_score"]
            elif indel_mode and fast_MSA_mode:
                seq_scoring = []
                for sequence in sequences_to_score:
                    seq_short_name = fast_MSA_short_names_mapping[sequence[0]]
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
        if model_type!="MSA_Transformer": 
            num_MSAs_in_batch, num_sequences_in_alignments, seqlen = batch_token_sequences.size()
            batch_token_sequences = batch_token_sequences.view(num_sequences_in_alignments, seqlen) #drop the dummy batch dimension from the tokenizer when not using MSAT
        batch_token_sequences = batch_token_sequences.to(device)
        processed_batch = {
            'input_tokens': batch_token_sequences,
            'mutant_mutated_seq_pairs': batch['mutant_mutated_seq_pairs']
        }
    elif model_type=="Tranception":
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(model.device)
        processed_batch = batch
    else:
        print("Model type has to be one of 'Tranception'|'MSA_Transformer'|'ESM1v'")
        return None
    
    return processed_batch


def main(
    assay_reference_file_location=None,
    assay_index=0,
    input_data_location=None,
    output_data_location=None,
    model_type='Tranception',
    model_location=None,
    max_positions=1024,
    long_sequences_slicing_method='center',
    batch_size=32,
    indel_mode=False,
    half_precision=False,
    num_MSA_sequences=1000,
    MSA_data_folder=None,
    MSA_weight_data_folder=None,
    path_to_hhfilter=None,
    path_to_clustalomega=None,
    fast_MSA_mode=False,
    target_seq=None,
    MSA_location=None,
    weight_file_name=None,
    MSA_start=None,
    MSA_end=None,
    use_cpu=False,
    ):
    
    if not use_cpu and not torch.cuda.is_available():
        print("Error: CUDA not available. This script is intended to run on a GPU, to use a CPU run with --use_cpu")
        exit(1)
    elif use_cpu and torch.cuda.is_available():
        print("Note: CUDA is available, but will not be used because --use_cpu is specified. To use the GPU, remove the --use_cpu flag.")
    
    assert (indel_mode and not fast_MSA_mode and batch_size==1) or (fast_MSA_mode and model_type=="MSA_Transformer") or (not indel_mode), "Indel mode typically run with batch size of 1, unless when using fast_MSA_mode for MSA Transformer"

    # Path to the input CSV file
    if assay_reference_file_location is not None:
        assay_reference_file = pd.read_csv(assay_reference_file_location)
        assay_id=assay_reference_file["DMS_id"][assay_index]
        assay_file_name = assay_reference_file["DMS_filename"][assay_reference_file["DMS_id"]==assay_id].values[0]
        input_data_location = input_data_location + os.sep + assay_file_name
        target_seq = assay_reference_file["target_seq"][assay_reference_file["DMS_id"]==assay_id].values[0]
    else:
        assert target_seq is not None, "Reference file provided and target_seq not provided"
        assay_id = input_data_location.split(".csv")[0].split(os.sep)[-1]
        assay_file_name = input_data_location.split(os.sep)[-1]
        target_seq = target_seq
        if MSA_location:
            MSA_filename = MSA_location.split(os.sep)[-1]
            MSA_data_folder = os.sep.join(MSA_location.split(os.sep)[:-1])
        if (MSA_start is None) or (MSA_end is None): 
            if MSA_data_folder: print("MSA start and end not provided -- Assuming the MSA is covering the full WT sequence")
            MSA_start = 1
            MSA_end = len(target_seq)

    print("Assay: {}".format(assay_file_name))

    # Load the PyTorch model from the checkpoint
    if model_type in ["MSA_Transformer","ESM1v"]:
        alphabet = Alphabet.from_architecture("msa_transformer")
        alphabet_size = len(alphabet)
        model, _ = load_model_and_alphabet(model_location)
        model.MSA_sample_sequences=None
    elif model_type=="Tranception":
        config = json.load(open(model_location+os.sep+'config.json'))
        config = TranceptionConfig(**config)
        config.tokenizer = get_tranception_tokenizer()
        config.full_target_seq = target_seq
        config.inference_time_retrieval_type = None
        config.retrieval_aggregation_mode = None
        alphabet = None # Only used in process_embeddings_batch for ESM models
        model = TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=model_location,config=config)

    # Set the model to evaluation mode & move to cuda
    model.eval()
    if not use_cpu:
        model.cuda()
    #DMS file
    df = pd.read_csv(input_data_location)
    if 'mutated_sequence' not in df: df['mutated_sequence'] = df['mutant'] # May happen on indel assays
    df = df[['mutant','mutated_sequence']]

    # Path to the output file for storing embeddings and original sequences
    if not os.path.exists(output_data_location): os.mkdir(output_data_location)
    output_embeddings_path = output_data_location + os.sep + assay_file_name.split(".csv")[0] + '.h5'

    if model_type=="MSA_Transformer":
        MSA_filename = assay_reference_file["MSA_filename"][assay_reference_file["DMS_id"]==assay_id].values[0] if assay_reference_file_location is not None else MSA_filename
        MSA_weights_filename = assay_reference_file["weight_file_name"][assay_reference_file["DMS_id"]==assay_id].values[0] if assay_reference_file_location is not None else weight_file_name
        MSA_sequences, MSA_weights = process_MSA(MSA_data_folder=MSA_data_folder, MSA_weight_data_folder=MSA_weight_data_folder, MSA_filename=MSA_filename, MSA_weights_filename=MSA_weights_filename, path_to_hhfilter=path_to_hhfilter)
        MSA_start_position = int(assay_reference_file["MSA_start"][assay_reference_file["DMS_id"]==assay_id].values[0]) if assay_reference_file_location is not None else MSA_start
        MSA_end_position = int(assay_reference_file["MSA_end"][assay_reference_file["DMS_id"]==assay_id].values[0]) if assay_reference_file_location is not None else MSA_end
    else:
        MSA_sequences = None
        MSA_weights = None
        MSA_start_position = None
        MSA_end_position = None

    # Create empty lists to store the embeddings and original sequences
    embeddings_list = []
    pseudo_likelihood_list = []
    sequences_list = []
    mutants_list = []

    # Pre-processing over gaps in sequences: 
    # If single-sequence input model --> remove gaps everywhere
    # If MSA input (eg., MSA Transformer) --> keep all gaps and align MSA to the gaps of the first sequence (assumes the MSA was created for that first sequence without gaps). If no gaps in that first sequence we dont do anything.
    if args.model_type in ["Tranception","ESM1v"]:
        df['mutated_sequence'] = df['mutated_sequence'].apply(lambda x: x.replace("-",""))
    elif args.model_type in ["MSA_Transformer"]:
        first_sequence = df['mutated_sequence'].values[0]
        dash_positions = [i for i, char in enumerate(first_sequence) if char == "-"] #Indices of gaps in first sequence to score
        if len(dash_positions)>0:
            print("Gaps detected in reference sequence -- inserting gaps in the MSA accordingly")
            MSA_sequences_with_dashes = []
            for seq_id, sequence in MSA_sequences:
                # Convert sequence to a list to allow insertion
                sequence_list = list(sequence)
                for position in dash_positions:
                    # Insert "-" at each position. Adjust for the number of dashes already inserted.
                    sequence_list.insert(position, "-")
                # Convert list back to string
                sequence_with_dashes = ''.join(sequence_list)
                MSA_sequences_with_dashes.append((seq_id, sequence_with_dashes))
            MSA_sequences = MSA_sequences_with_dashes

    # Create a data loader to iterate over the input sequences. 
    # For ESM models (MSA Transformer in particular), the bulk of the work is done within process_embeddings_batch
    # For Tranception, utils for slicing already exist so we directly process & tokenize sequences below
    if model_type in ["MSA_Transformer","ESM1v"]:
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
    elif model_type=="Tranception":
        sliced_df = get_sequence_slices(df, 
            target_seq=target_seq, 
            model_context_len = model.config.n_ctx - 2, 
            indel_mode=indel_mode, 
            scoring_window="optimal"
        )
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

    # Loop over the batches of sequences in the input file
    with torch.no_grad():
        if fast_MSA_mode:
            model.MSA_sample_sequences = weighted_sample_MSA(
                        MSA_all_sequences=MSA_sequences, 
                        MSA_non_ref_sequences_weights=MSA_weights, 
                        number_sampled_MSA_sequences=num_MSA_sequences
                    )
            fast_MSA_short_names_mapping = {}
            fast_MSA_short_names = []
            print("There are {} sequences to score".format(len(df['mutated_sequence'])))
            for seq_index,seq in enumerate(list(df['mutated_sequence'])):
                short_name = 'mutant_to_score_'+str(seq_index)
                fast_MSA_short_names_mapping[seq] = short_name
                fast_MSA_short_names.append(short_name)
            fast_MSA_aligned_sequences = align_new_sequences_to_msa(model.MSA_sample_sequences, list(df['mutated_sequence']), fast_MSA_short_names, clustalomega_path=path_to_clustalomega)
            model.MSA_sample_sequences = [tup for tup in fast_MSA_aligned_sequences if tup[0] not in set(fast_MSA_short_names)]
        else:
            fast_MSA_aligned_sequences = None
            fast_MSA_short_names_mapping = None
        
        for batch in tqdm.tqdm(dataloader, desc="Embedding sequences", total=len(dataloader)):
            processed_batch = process_embeddings_batch(
                                batch = batch,
                                model = model,
                                model_type = model_type,
                                alphabet = alphabet,
                                MSA_sequences = MSA_sequences, 
                                MSA_weights = MSA_weights,
                                MSA_start_position = MSA_start_position, 
                                MSA_end_position = MSA_end_position,
                                num_MSA_sequences = num_MSA_sequences,
                                max_positions = max_positions,
                                long_sequences_slicing_method = long_sequences_slicing_method,
                                device = next(model.parameters()).device,
                                eval_mode=True,
                                indel_mode=indel_mode,
                                fast_MSA_mode=fast_MSA_mode,
                                fast_MSA_aligned_sequences=fast_MSA_aligned_sequences,
                                fast_MSA_short_names_mapping=fast_MSA_short_names_mapping,
                                clustalomega_path=path_to_clustalomega
                            )
            if model_type=="MSA_Transformer":
                tokens = processed_batch['input_tokens']
                assert tokens.ndim == 3, "Finding dimension of tokens to be: {}".format(tokens.ndim)
                num_MSAs_in_batch, num_sequences_in_alignments, seqlen = tokens.size()
                batch_size = num_MSAs_in_batch
                output = model(tokens, repr_layers=[12])
                embeddings = output["representations"][12][:] # B, N, L, D
                if fast_MSA_mode: # A single batch with multiple sequences to score in it to speed things up
                    embeddings = embeddings[:,:batch_size,:,:] # 1, N, L, D 
                    embeddings = [embeddings[:,i,:,:] for i in range(batch_size)]
                    logits = output["logits"][:,:batch_size].view(batch_size,-1,alphabet_size)
                    tokens = tokens[:,:batch_size].view(batch_size,-1)
                    pseudo_ppx = - CrossEntropyLoss(reduction='none', label_smoothing=0.0)(logits.reshape(-1, alphabet_size), tokens.reshape(-1)).view(batch_size, seqlen)
                    pseudo_ppx = pseudo_ppx.mean(dim=-1).view(-1)
                    batch['mutant_mutated_seq_pairs'] = batch['mutant_mutated_seq_pairs'][0][:batch_size]
                else:
                    embeddings = embeddings[:,0,:,:] # In each MSA batch the first sequence is what we care about. The other MSA sequences were present just to compute embeddings and logits
                    logits = output["logits"][:,0] # Filtering logits of points to score
                    tokens = tokens[:,0] # Filtering tokens of points to score
                    pseudo_ppx = - CrossEntropyLoss(reduction='none', label_smoothing=0.0)(logits.reshape(-1, alphabet_size), tokens.reshape(-1)).view(num_MSAs_in_batch, seqlen)
                    pseudo_ppx = pseudo_ppx.mean(dim=-1).view(-1) #Average across sequence length
                    batch['mutant_mutated_seq_pairs'] = [seq[0] for seq in batch['mutant_mutated_seq_pairs']] # Remove MSA sequences from batch by selecting the first sequence in each MSA input sets
                mutant, sequence = zip(*batch['mutant_mutated_seq_pairs'])
            elif model_type == "ESM1v":
                tokens = processed_batch['input_tokens']
                assert tokens.ndim == 2, "Finding dimension of tokens to be: {}".format(tokens.ndim)
                batch_size, seqlen = tokens.size()
                last_layer_index = 33
                output = model(tokens, repr_layers=[last_layer_index])
                embeddings = output["representations"][last_layer_index][:] # N, L, D
                logits = output["logits"]
                pseudo_ppx = - CrossEntropyLoss(reduction='none', label_smoothing=0.0)(logits.view(-1, alphabet_size), tokens.view(-1)).view(batch_size, seqlen)
                pseudo_ppx = pseudo_ppx.mean(dim=-1).view(-1)
                mutant, sequence = zip(*batch['mutant_mutated_seq_pairs'])
            elif model_type=="Tranception":
                output = model(**processed_batch, return_dict=True, output_hidden_states=True)
                embeddings = output.hidden_states[-1] # Extract embeddings from last layer
                logits = output.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = processed_batch['labels'][..., 1:].contiguous()
                pseudo_ppx = - CrossEntropyLoss(reduction='none')(input=shift_logits.view(-1, shift_logits.size(-1)), target=shift_labels.view(-1)).view(shift_logits.shape[0],shift_logits.shape[1])
                pseudo_ppx = pseudo_ppx.mean(dim=-1).view(-1) # Shape (B,)
                full_batch_length = len(processed_batch['input_ids'])
                sequence = np.array(df['mutated_sequence'][mutant_index:mutant_index+full_batch_length])
                mutant = np.array(df['mutant'][mutant_index:mutant_index+full_batch_length])
                mutant_index+=full_batch_length

            # Add the embeddings and original sequences to the corresponding lists
            pseudo_ppx = [pppx.cpu().item() for pppx in pseudo_ppx]
            pseudo_likelihood_list += pseudo_ppx
            assert len(embeddings.shape)==3, "Embedding tensor is not of proper size (batch_size,seq_len,embedding_dim)"
            B,L,D=embeddings.shape
            embeddings = [embedding.view(1,L,D).cpu() for embedding in embeddings]
            if half_precision: embeddings = [embedding.half() for embedding in embeddings]
            embeddings_list += embeddings
            sequences_list.append(list(sequence))
            mutants_list.append(list(mutant))

    # Concatenate the embeddings 
    if indel_mode:
        embedding_len_set = set([seq.size(1) for seq in embeddings_list])
        num_embeddings = len(embeddings_list)
        max_seq_length = max(embedding_len_set)
        embedding_dim = embeddings_list[0].shape[-1] # embeddings_list is a list of embeddings, each of them of shape (1,seq_len,embedding_dim)
        embeddings = torch.zeros(num_embeddings,max_seq_length,embedding_dim)
        if half_precision: embeddings = embeddings.half()
        for emb_index, embedding in enumerate(embeddings_list):
            assert len(embedding.shape)==3, "embedding index {} is of incorrect size: {}".format(emb_index, embedding.shape)
            embedding_len = embedding.shape[1]
            embeddings[emb_index,:embedding_len] = embedding
    else:
        embeddings = torch.cat(embeddings_list, dim=0)
    assert len(embeddings.shape)==3, "Embedding tensor is not of proper size (num_mutated_seqs,seq_len,embedding_dim)"
    pseudo_likelihoods = torch.tensor(pseudo_likelihood_list)
    sequences = sum(sequences_list, []) # Flatten the list if needed
    mutants = sum(mutants_list, []) # Flatten the list if needed

    embeddings_dict = {
        'embeddings': embeddings,
        'pseudo_likelihoods': pseudo_likelihoods,
        'sequences': sequences,
        'mutants': mutants
    }

    # Store data as HDF5
    with h5py.File(output_embeddings_path, 'w') as h5f:
        for key, value in embeddings_dict.items():
            h5f.create_dataset(key, data=value)


if __name__ == "__main__":
    args = parse_arguments()
    main(
        assay_reference_file_location=args.assay_reference_file_location,
        assay_index=args.assay_index,
        input_data_location=args.input_data_location,
        output_data_location=args.output_data_location,
        model_type=args.model_type,
        model_location=args.model_location,
        max_positions=args.max_positions,
        long_sequences_slicing_method=args.long_sequences_slicing_method,
        batch_size=args.batch_size,
        indel_mode=args.indel_mode,
        half_precision=args.half_precision,
        #MSA specific parameters
        num_MSA_sequences=args.num_MSA_sequences,
        MSA_data_folder=args.MSA_data_folder,
        MSA_weight_data_folder=args.MSA_weight_data_folder,
        path_to_hhfilter=args.path_to_hhfilter,
        path_to_clustalomega=args.path_to_clustalomega,
        fast_MSA_mode=args.fast_MSA_mode,
        #If not using a reference file
        target_seq=args.target_seq,
        MSA_location=args.MSA_location,
        use_cpu=args.use_cpu,
    )
