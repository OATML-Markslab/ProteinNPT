import os,sys
import torch
import numpy as np
import pandas as pd
import argparse
import h5py
import json
import tqdm

from proteinnpt.utils.tranception.model_pytorch import get_tranception_tokenizer
from proteinnpt.utils.tranception.config import TranceptionConfig
from proteinnpt.utils.tranception.model_pytorch import TranceptionLMHeadModel
from proteinnpt.utils.esm.pretrained import load_model_and_alphabet
from proteinnpt.utils.msa_utils import process_MSA
from proteinnpt.utils.data_utils import cleanup_ids_assay_data
from proteinnpt.utils.embedding_utils import get_ESM_dataloader, get_Tranception_dataloader, process_MSA_Transformer_batch, \
    process_ESM_batch, process_Tranception_batch, get_embeddings_MSA_Transformer, get_embeddings_ESM, get_embeddings_Tranception, \
    process_MSA_Transformer_batch_multiple_MSAs

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract embeddings')
    parser.add_argument('--assay_reference_file_location', default=None, type=str, help='Path to reference file with list of assays to score')
    parser.add_argument('--assay_index', default=0, type=int, help='Index of assay in the ProteinGym reference file to compute embeddings for')
    parser.add_argument('--input_data_location', default=None, type=str, help='Location of input data with all mutated sequences [If a reference file is used, this is the location where are assays are stored. If a single csv file is passed, this is the full path to that assay data]')
    parser.add_argument('--output_data_location', default=None, type=str, help='Location of output embeddings')
    parser.add_argument('--model_type', default='Tranception', type=str, help='Model type to compute embeddings with ["MSA_Transformer", "ESM1v", "Tranception"]')
    parser.add_argument('--model_location', default=None, type=str, help='Location of model used to embed protein sequences')
    parser.add_argument('--max_positions', default=1024, type=int, help='Maximum context length of embedding model')
    parser.add_argument('--long_sequences_slicing_method', default='center', type=str, help='Method to slice long sequences [rolling, center, left]')
    parser.add_argument('--batch_size', default=1, type=int, help='Eval batch size')
    parser.add_argument('--indel_mode', action='store_true', help='Use this mode if extracting embeddings of indel assays')
    parser.add_argument('--half_precision', action='store_true', help='Store embeddings as 16-bit floating point numbers (float16)')
    #MSA-specific parameters (only relevant for MSA Transformer)
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

# Note: to add a new embedding model:
# 1. Create dataloader (ie. single data transformation for all forward passes)
# 2. Create function to process batch (ie. batch specific data transformations)
# 3. Create function to get embeddings (ie. applies model on batch to get embeddings and pseudo-ll)
# 4. Update main script in all if statements involving model_type

def preprocess_gaps_in_sequences(df, mode="drop_gaps", MSA_sequences=None):
    # If single-sequence input model --> remove gaps everywhere
    # If MSA input (eg., MSA Transformer) --> keep all gaps and align MSA to the gaps of the first sequence (assumes the MSA was created for that first sequence without gaps). If no gaps in that first sequence we dont do anything.
    if mode=="drop_gaps":
        df['mutated_sequence'] = df['mutated_sequence'].apply(lambda x: x.replace("-",""))
        return df, MSA_sequences
    elif mode=="insert_focus_seq_gaps_in_MSA":
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
        return df, MSA_sequences

def standardize_embeddings_length(embeddings, target_length, pad_value=0):
    """Pad or trim embeddings to a target length"""
    batch_size, seq_len, emb_dim = embeddings.shape
    
    if seq_len < target_length:
        # Pad
        padding = torch.ones(batch_size, target_length - seq_len, emb_dim) * pad_value
        padded_embeddings = torch.cat([embeddings, padding.to(embeddings.device)], dim=1)
        return padded_embeddings
    elif seq_len > target_length:
        # Trim
        return embeddings[:, :target_length, :]
    else:
        return embeddings

def main(
    assay_reference_file_location=None,
    assay_index=0,
    input_data_location=None,
    output_data_location=None,
    model_type='Tranception',
    model_location=None,
    max_positions=1024,
    long_sequences_slicing_method='center',
    batch_size=1,
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
    start_idx=1,
    use_cpu=False,
    ):
    
    if not use_cpu and not torch.cuda.is_available():
        print("Error: CUDA not available. This script is intended to run on a GPU, to use a CPU run with --use_cpu")
        exit(1)
    elif use_cpu and torch.cuda.is_available():
        print("Note: CUDA is available, but will not be used because --use_cpu is specified. To use the GPU, remove the --use_cpu flag.")
    assert (indel_mode and not fast_MSA_mode and batch_size==1) or (fast_MSA_mode and model_type=="MSA_Transformer") or (not indel_mode), "Indel mode typically run with batch size of 1, unless when using fast_MSA_mode for MSA Transformer"

    # Path to the input CSV file
    MSA_filename = None
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
        if MSA_location:
            MSA_filename = MSA_location.split(os.sep)[-1]
            MSA_data_folder = os.sep.join(MSA_location.split(os.sep)[:-1])
        if (MSA_start is None) or (MSA_end is None): 
            if MSA_data_folder: print("MSA start or MSA end not provided -- Assuming the MSA is covering the full WT sequence")
            MSA_start = 1
            MSA_end = len(target_seq)

    print("Assay: {}".format(assay_file_name))

    # Load the PyTorch model from the checkpoint
    if model_type=="MSA_Transformer" or model_type.startswith("ESM"):
        #alphabet = Alphabet.from_architecture("msa_transformer") if model_type=="MSA_Transformer" else Alphabet.from_architecture("ESM-1b")
        model, alphabet = load_model_and_alphabet(model_location)
        alphabet_size = len(alphabet)
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
    if not use_cpu: model.cuda()
    #DMS file
    df = pd.read_csv(input_data_location)
    df = cleanup_ids_assay_data(df,indel_mode=indel_mode, target_seq=target_seq)
    df = df[['mutant','mutated_sequence']]

    # Path to the output file for storing embeddings and original sequences
    if not os.path.exists(output_data_location): os.makedirs(output_data_location, exist_ok=True)
    output_embeddings_path = output_data_location + os.sep + assay_file_name.split(".csv")[0] + '.h5'

    if model_type=="MSA_Transformer":
        MSA_filename = assay_reference_file["MSA_filename"][assay_reference_file["DMS_id"]==assay_id].values[0] if (assay_reference_file_location is not None and "MSA_filename" in assay_reference_file) else MSA_filename
        MSA_weights_filename = assay_reference_file["weight_file_name"][assay_reference_file["DMS_id"]==assay_id].values[0] if (assay_reference_file_location is not None and "weight_file_name" in assay_reference_file) else weight_file_name
        MSA_sequences, MSA_weights = process_MSA(MSA_data_folder=MSA_data_folder, MSA_weight_data_folder=MSA_weight_data_folder, MSA_filename=MSA_filename, MSA_weights_filename=MSA_weights_filename, path_to_hhfilter=path_to_hhfilter)
        MSA_start_position = int(assay_reference_file["MSA_start"][assay_reference_file["DMS_id"]==assay_id].values[0]) if (assay_reference_file_location is not None and "MSA_start" in assay_reference_file) else MSA_start
        MSA_end_position = int(assay_reference_file["MSA_end"][assay_reference_file["DMS_id"]==assay_id].values[0]) if (assay_reference_file_location is not None and "MSA_end" in assay_reference_file)  else MSA_end
    else:
        MSA_sequences, MSA_weights, MSA_start_position, MSA_end_position = None, None, None, None
        
    # Create empty lists to store the embeddings and original sequences
    embeddings_list = []
    pseudo_likelihood_list = []
    sequences_list = []
    mutants_list = []

    # Pre-processing over gaps in sequences: 
    if model_type=="Tranception" or model_type.startswith("ESM"):
        df, MSA_sequences = preprocess_gaps_in_sequences(df, mode="drop_gaps")
    elif model_type=="MSA_Transformer":
        df, MSA_sequences = preprocess_gaps_in_sequences(df, mode="insert_focus_seq_gaps_in_MSA", MSA_sequences=MSA_sequences)

    # Create a data loader to iterate over the input sequences. 
    # For ESM models (MSA Transformer in particular), the bulk of the work is done within process_embeddings_batch
    # For Tranception, utils for slicing already exist so we directly process & tokenize sequences below
    if model_type=="MSA_Transformer" or model_type.startswith("ESM"):
        dataloader = get_ESM_dataloader(df, batch_size)
    elif model_type=="Tranception":
        dataloader = get_Tranception_dataloader(df, batch_size, model, target_seq, indel_mode)
    
    # Loop over the batches of sequences in the input file
    mutant_index=0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Embedding sequences", total=len(dataloader)):
            if model_type=="MSA_Transformer":
                single_MSA = True #Default option for now
                if single_MSA:
                    processed_batch = process_MSA_Transformer_batch(
                        batch = batch, 
                        model = model, 
                        alphabet = alphabet, 
                        max_positions = max_positions, 
                        MSA_start_position = MSA_start_position, 
                        MSA_end_position = MSA_end_position, 
                        MSA_weights = MSA_weights,
                        MSA_sequences = MSA_sequences,
                        num_MSA_sequences = num_MSA_sequences, 
                        eval_mode = True, 
                        start_idx = start_idx, 
                        long_sequences_slicing_method = long_sequences_slicing_method, 
                        indel_mode = indel_mode, 
                        fast_MSA_mode = fast_MSA_mode, 
                        clustalomega_path = path_to_clustalomega,
                        num_extra_tokens = 1
                        )
                else:
                    processed_batch = process_MSA_Transformer_batch_multiple_MSAs(
                        batch = batch,
                        model = model,
                        alphabet = alphabet, 
                        MSA_folder = MSA_data_folder, 
                        num_MSA_sequences = num_MSA_sequences
                    )
                embeddings, pseudo_ll, processed_batch = get_embeddings_MSA_Transformer(
                    model = model, 
                    processed_batch = processed_batch, 
                    alphabet_size = alphabet_size, 
                    fast_MSA_mode=fast_MSA_mode
                    )
                mutant, sequence = zip(*processed_batch['mutant_mutated_seq_pairs'])
            elif model_type.startswith("ESM"):
                processed_batch = process_ESM_batch(
                    batch = batch, 
                    model = model, 
                    alphabet = alphabet, 
                    max_positions = max_positions, 
                    long_sequences_slicing_method = long_sequences_slicing_method, 
                    eval_mode = True, 
                    start_idx = start_idx,
                    num_extra_tokens = 2
                    )
                embeddings, pseudo_ll = get_embeddings_ESM(
                    model = model, 
                    model_type = model_type, 
                    processed_batch = processed_batch, 
                    alphabet_size = alphabet_size
                    )
                mutant, sequence = zip(*processed_batch['mutant_mutated_seq_pairs'])
            elif model_type=="Tranception":
                processed_batch = process_Tranception_batch(
                    batch = batch, 
                    model = model
                    )
                embeddings, pseudo_ll = get_embeddings_Tranception(
                    model = model, 
                    processed_batch = processed_batch
                    )
                full_batch_length = len(processed_batch['input_ids'])
                sequence = np.array(df['mutated_sequence'][mutant_index:mutant_index+full_batch_length])
                mutant = np.array(df['mutant'][mutant_index:mutant_index+full_batch_length])
                mutant_index+=full_batch_length
            else:
                print("Model type has to be one of 'Tranception'|'MSA_Transformer'|'ESM1v'")
                sys.exit(0)

            # Add the embeddings and original sequences to the corresponding lists
            assert len(embeddings.shape)==3, "Embedding tensor is not of proper size (batch_size,seq_len,embedding_dim)"
            B,L,D=embeddings.shape
            embeddings = [embedding.view(1,L,D).cpu() for embedding in embeddings]
            if half_precision: embeddings = [embedding.half() for embedding in embeddings]
            embeddings_list += embeddings
            pseudo_likelihood_list += list(pseudo_ll)
            sequences_list.append(list(sequence))
            mutants_list.append(list(mutant))

    # Concatenate the embeddings 
    if indel_mode:
        num_embeddings = np.array([embedding.shape[0] for embedding in embeddings_list]).sum() # embeddings_list is a list of embeddings, each of them of shape (batch_size_embedding_processing,seq_len,embedding_dim)
        embedding_len_set = set([seqs.size(1) for seqs in embeddings_list])
        max_seq_length = max(embedding_len_set)
        embedding_dim = embeddings_list[0].shape[-1]
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
        #MSA-specific parameters
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
