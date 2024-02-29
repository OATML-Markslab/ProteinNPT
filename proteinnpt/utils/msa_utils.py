import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import tempfile
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import subprocess

class MSA_processing:
    def __init__(self,
        MSA_location="",
        theta=0.2,
        use_weights=True,
        weights_location="./data/weights",
        preprocess_MSA=True,
        threshold_sequence_frac_gaps=0.5,
        threshold_focus_cols_frac_gaps=1.0,
        remove_sequences_with_indeterminate_AA_in_focus_cols=True
        ):
        
        """
        This class was borrowed from our EVE codebase: https://github.com/OATML-Markslab/EVE
        Parameters:
        - msa_location: (path) Location of the MSA data. Constraints on input MSA format: 
            - focus_sequence is the first one in the MSA data
            - first line is structured as follows: ">focus_seq_name/start_pos-end_pos" (e.g., >SPIKE_SARS2/310-550)
            - corespondding sequence data located on following line(s)
            - then all other sequences follow with ">name" on first line, corresponding data on subsequent lines
        - theta: (float) Sequence weighting hyperparameter. Generally: Prokaryotic and eukaryotic families =  0.2; Viruses = 0.01
        - use_weights: (bool) If False, sets all sequence weights to 1. If True, checks weights_location -- if non empty uses that; 
            otherwise compute weights from scratch and store them at weights_location
        - weights_location: (path) Location to load from/save to the sequence weights
        - preprocess_MSA: (bool) performs pre-processing of MSA to remove short fragments and positions that are not well covered.
        - threshold_sequence_frac_gaps: (float, between 0 and 1) Threshold value to define fragments
            - sequences with a fraction of gap characters above threshold_sequence_frac_gaps are removed
            - default is set to 0.5 (i.e., fragments with 50% or more gaps are removed)
        - threshold_focus_cols_frac_gaps: (float, between 0 and 1) Threshold value to define focus columns
            - positions with a fraction of gap characters above threshold_focus_cols_pct_gaps will be set to lower case (and not included in the focus_cols)
            - default is set to 0.3 (i.e., focus positions are the ones with 30% of gaps or less, i.e., 70% or more residue occupancy)
        - remove_sequences_with_indeterminate_AA_in_focus_cols: (bool) Remove all sequences that have indeterminate AA (e.g., B, J, X, Z) at focus positions of the wild type
        """
        self.MSA_location = MSA_location
        self.weights_location = weights_location
        self.theta = theta
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.use_weights = use_weights
        self.preprocess_MSA = preprocess_MSA
        self.threshold_sequence_frac_gaps = threshold_sequence_frac_gaps
        self.threshold_focus_cols_frac_gaps = threshold_focus_cols_frac_gaps
        self.remove_sequences_with_indeterminate_AA_in_focus_cols = remove_sequences_with_indeterminate_AA_in_focus_cols

        self.gen_alignment()

    def gen_alignment(self):
        """ Read training alignment and store basics in class instance """
        self.aa_dict = {}
        for i,aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        self.seq_name_to_sequence = defaultdict(str)
        name = ""
        with open(self.MSA_location, "r") as msa_data:
            for i, line in enumerate(msa_data):
                line = line.rstrip()
                if line.startswith(">"):
                    name = line
                    if i==0:
                        self.focus_seq_name = name
                else:
                    self.seq_name_to_sequence[name] += line

        
        ## MSA pre-processing to remove inadequate columns and sequences
        if self.preprocess_MSA:
            msa_df = pd.DataFrame.from_dict(self.seq_name_to_sequence, orient='index', columns=['sequence'])
            # Data clean up
            msa_df.sequence = msa_df.sequence.apply(lambda x: x.replace(".","-")).apply(lambda x: ''.join([aa.upper() for aa in x]))
            # Remove columns that would be gaps in the wild type
            non_gap_wt_cols = [aa!='-' for aa in msa_df.sequence[self.focus_seq_name]]
            msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join([aa for aa,non_gap_ind in zip(x, non_gap_wt_cols) if non_gap_ind]))
            assert 0.0 <= self.threshold_sequence_frac_gaps <= 1.0,"Invalid fragment filtering parameter"
            assert 0.0 <= self.threshold_focus_cols_frac_gaps <= 1.0,"Invalid focus position filtering parameter"
            msa_array = np.array([list(seq) for seq in msa_df.sequence])
            gaps_array = np.array(list(map(lambda seq: [aa=='-' for aa in seq], msa_array)))
            # Identify fragments with too many gaps
            seq_gaps_frac = gaps_array.mean(axis=1)
            seq_below_threshold = seq_gaps_frac <= self.threshold_sequence_frac_gaps
            print("Proportion of sequences dropped due to fraction of gaps: "+str(round(float(1 - seq_below_threshold.sum()/seq_below_threshold.shape)*100,2))+"%")
            # Identify focus columns
            columns_gaps_frac = gaps_array[seq_below_threshold].mean(axis=0)
            index_cols_below_threshold = columns_gaps_frac <= self.threshold_focus_cols_frac_gaps
            print("Proportion of non-focus columns removed: "+str(round(float(1 - index_cols_below_threshold.sum()/index_cols_below_threshold.shape)*100,2))+"%")
            # Lower case non focus cols and filter fragment sequences
            msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join([aa.upper() if upper_case_ind else aa.lower() for aa, upper_case_ind in zip(x, index_cols_below_threshold)]))
            msa_df = msa_df[seq_below_threshold]
            # Overwrite seq_name_to_sequence with clean version
            self.seq_name_to_sequence = defaultdict(str)
            for seq_idx in range(len(msa_df['sequence'])):
                self.seq_name_to_sequence[msa_df.index[seq_idx]] = msa_df.sequence.iloc[seq_idx]

        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper() and s!='-'] 
        self.focus_seq_trimmed = [self.focus_seq[ix] for ix in self.focus_cols]
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        # Connect local sequence index with uniprot index (index shift inferred from 1st row of MSA)
        focus_loc = self.focus_seq_name.split("/")[-1]
        start,stop = focus_loc.split("-")
        self.focus_start_loc = int(start)
        self.focus_stop_loc = int(stop)
        self.uniprot_focus_col_to_wt_aa_dict \
            = {idx_col+int(start):self.focus_seq[idx_col] for idx_col in self.focus_cols} 
        self.uniprot_focus_col_to_focus_idx \
            = {idx_col+int(start):idx_col for idx_col in self.focus_cols} 

        # Move all letters to CAPS; keeps focus columns only
        self.raw_seq_name_to_sequence = self.seq_name_to_sequence.copy()
        for seq_name,sequence in self.seq_name_to_sequence.items():
            sequence = sequence.replace(".","-")
            self.seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in self.focus_cols]

        # Remove sequences that have indeterminate AA (e.g., B, J, X, Z) in the focus columns
        if self.remove_sequences_with_indeterminate_AA_in_focus_cols:
            alphabet_set = set(list(self.alphabet))
            seq_names_to_remove = []
            for seq_name,sequence in self.seq_name_to_sequence.items():
                for letter in sequence:
                    if letter not in alphabet_set and letter != "-":
                        seq_names_to_remove.append(seq_name)
                        continue
            seq_names_to_remove = list(set(seq_names_to_remove))
            for seq_name in seq_names_to_remove:
                del self.seq_name_to_sequence[seq_name]

        # Encode the sequences
        print ("Encoding sequences")
        self.one_hot_encoding = np.zeros((len(self.seq_name_to_sequence.keys()),len(self.focus_cols),len(self.alphabet)))
        print("One-hot encoded sequences shape:" + str(self.one_hot_encoding.shape))
        for i,seq_name in enumerate(self.seq_name_to_sequence.keys()):
            sequence = self.seq_name_to_sequence[seq_name]
            for j,letter in enumerate(sequence):
                if letter in self.aa_dict: 
                    k = self.aa_dict[letter]
                    self.one_hot_encoding[i,j,k] = 1.0

        if self.use_weights:
            try:
                self.weights = np.load(file=self.weights_location)
                print("Loaded sequence weights from disk")
            except:
                print ("Computing sequence weights")
                list_seq = self.one_hot_encoding
                list_seq = list_seq.reshape((list_seq.shape[0], list_seq.shape[1] * list_seq.shape[2]))
                def compute_weight(seq):
                    number_non_empty_positions = np.dot(seq,seq)
                    if number_non_empty_positions>0:
                        denom = np.dot(list_seq,seq) / np.dot(seq,seq) 
                        denom = np.sum(denom > 1 - self.theta) 
                        return 1/denom
                    else:
                        return 0.0 #return 0 weight if sequence is fully empty
                self.weights = np.array(list(map(compute_weight,list_seq)))
                np.save(file=self.weights_location, arr=self.weights)
        else:
            # If not using weights, use an isotropic weight matrix
            print("Not weighting sequence data")
            self.weights = np.ones(self.one_hot_encoding.shape[0])

        self.Neff = np.sum(self.weights)
        self.num_sequences = self.one_hot_encoding.shape[0]
        self.seq_name_to_weight={}
        for i,seq_name in enumerate(self.seq_name_to_sequence.keys()):
            self.seq_name_to_weight[seq_name]=self.weights[i]

        print ("Neff =",str(self.Neff))
        print ("Data Shape =",self.one_hot_encoding.shape)
        print("Tmp Lood: weights shape:", self.weights.shape)
        assert self.weights.shape[0] == self.num_sequences  # == self.one_hot_encoding.shape[0]

def filter_msa(filename, path_to_hhfilter, hhfilter_min_cov=75, hhfilter_max_seq_id=90, hhfilter_min_seq_id=0):
    """
    We use the filtering defaults from the MSA Transformer paper wrt maximum sequence id: hhfilter_max_seq_id = 90; hhfilter_min_cov=75.
    Install hhfilter:
        wget https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-AVX2-Linux.tar.gz; tar xvfz hhsuite-3.3.0-AVX2-Linux.tar.gz; export PATH="$(pwd)/bin:$(pwd)/scripts:$PATH"
    """
    input_folder = '/'.join(filename.split('/')[:-1])
    msa_name = filename.split('/')[-1].split('.')[0]
    preprocessed_filename = input_folder+os.sep+'preprocessed'+os.sep+msa_name
    output_filename = input_folder+os.sep+'hhfiltered'+os.sep+msa_name+'_hhfiltered_cov_'+str(hhfilter_min_cov)+'_maxid_'+str(hhfilter_max_seq_id)+'_minid_'+str(hhfilter_min_seq_id)+'.a2m'
    
    if msa_name=="R1AB_SARS2_02-19-2022_b07": hhfilter_max_seq_id=100 #Otherwise we would only keep 1 sequence.
    if not os.path.isdir(input_folder+os.sep+'preprocessed'):
        os.mkdir(input_folder+os.sep+'preprocessed')
    if not os.path.isdir(input_folder+os.sep+'hhfiltered'):
        os.mkdir(input_folder+os.sep+'hhfiltered')
    if not os.path.exists(output_filename):
        os.system('cat '+filename+' | tr  "."  "-" >> '+preprocessed_filename+'.a2m')
        os.system('dd if='+preprocessed_filename+'.a2m of='+preprocessed_filename+'_UC.a2m conv=ucase')
        os.system(path_to_hhfilter+os.sep+'bin/hhfilter -cov '+str(hhfilter_min_cov)+' -id '+str(hhfilter_max_seq_id)+' -qid '+str(hhfilter_min_seq_id)+' -i '+preprocessed_filename+'_UC.a2m -o '+output_filename)
    return output_filename

def compute_sequence_weights(MSA_filename, MSA_weights_filename):
    """
    MSA_non_ref_sequences_weights contains the weights for all sequences besides the reference sequence
    """
    processed_MSA = MSA_processing(
        MSA_location=MSA_filename,
        use_weights=True,
        weights_location=MSA_weights_filename
    )
    print("Neff: "+str(processed_MSA.Neff))
    print("Name of focus_seq: "+str(processed_MSA.focus_seq_name))
    MSA_other_sequences=[]
    weights=[]
    MSA_reference_sequence=[]
    for seq_name in processed_MSA.raw_seq_name_to_sequence.keys():
        if seq_name == processed_MSA.focus_seq_name:
            MSA_reference_sequence.append((seq_name,processed_MSA.raw_seq_name_to_sequence[seq_name]))
            del processed_MSA.seq_name_to_weight[seq_name]
        else:
            if seq_name in processed_MSA.seq_name_to_weight: 
                MSA_other_sequences.append((seq_name,processed_MSA.raw_seq_name_to_sequence[seq_name]))
                weights.append(processed_MSA.seq_name_to_weight[seq_name])
    if len(MSA_other_sequences)>0:
        MSA_non_ref_sequences_weights = np.array(weights) / np.array(list(processed_MSA.seq_name_to_weight.values())).sum()
        print("Check sum weights MSA: "+str(np.array(weights).sum()))
    MSA_all_sequences = MSA_reference_sequence + MSA_other_sequences
    return MSA_all_sequences, MSA_non_ref_sequences_weights

def random_sample_MSA(filename, nseq):
    msa = [
            (record.description, str(record.seq)) for record in SeqIO.parse(filename, "fasta")
    ]
    nseq = min(len(msa),nseq) #ensures number of samples at most as large as pop size
    msa = random.sample(msa, nseq)
    msa = [(desc, seq.upper()) for desc, seq in msa]
    return msa
    
def weighted_sample_MSA(MSA_all_sequences, MSA_non_ref_sequences_weights, number_sampled_MSA_sequences):
    """
    We always enforce the first sequence in the MSA to be the refence sequence.
    """
    msa = [MSA_all_sequences[0]]
    msa.extend(random.choices(MSA_all_sequences[1:], weights=MSA_non_ref_sequences_weights, k=number_sampled_MSA_sequences-1))
    msa = [(desc, seq.upper()) for desc, seq in msa]
    return msa

def process_MSA(MSA_data_folder, MSA_weight_data_folder, MSA_filename, MSA_weights_filename, path_to_hhfilter):
    """
    Filter an MSA according to sequence identity (for e.g. MSATransformer) and then compute sequence weights
    """
    filtered_MSA_filename = filter_msa(filename=MSA_data_folder + os.sep + MSA_filename, path_to_hhfilter=path_to_hhfilter)
    MSA_all_sequences, MSA_non_ref_sequences_weights = compute_sequence_weights(MSA_filename=filtered_MSA_filename, MSA_weights_filename=os.path.join(MSA_weight_data_folder, "hhfiltered", MSA_weights_filename))
    return MSA_all_sequences, MSA_non_ref_sequences_weights

def align_new_sequences_to_msa(MSA_sequences, new_sequences, new_mutants, clustalomega_path):
    """
    Helps realign mutated sequences with sequences in a MSA. Useful to compute embeddings of indel 
    """
    # Convert MSA_sequences and new_sequence to a list of SeqRecord objects
    records = [SeqRecord(Seq(seq), id=name) for name, seq in MSA_sequences]
    for new_sequence,new_mutant in zip(new_sequences,new_mutants):
        records.append(SeqRecord(Seq(new_sequence), id=new_mutant))
    # Create a temporary file with the MSA and the new sequence combined
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
        tmp_filename = tmp_file.name
        for record in records:
            tmp_file.write(f">{record.id}\n{record.seq}\n")
    # Use ClustalOmega to realign everything
    output_filename = tempfile.NamedTemporaryFile(delete=False).name
    command = [
        clustalomega_path,
        "--in", tmp_filename,
        "--out", output_filename,
        "--verbose",
        "--auto",
        "--force"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error in running ClustalOmega:")
        print(result.stderr)
    os.remove(tmp_filename)
    # Parse the aligned sequences from the output file and convert to the desired format
    aligned_records = list(SeqIO.parse(output_filename, "fasta"))
    aligned_sequences = [(record.id, str(record.seq)) for record in aligned_records]
    # Remove the temporary output file
    os.remove(output_filename)
    return aligned_sequences