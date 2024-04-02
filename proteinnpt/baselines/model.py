import sys, os
import json
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import ConvBertConfig, ConvBertLayer

from ..utils.esm.modules import ESM1bLayerNorm
from ..utils.esm.axial_attention import RowSelfAttention, ColumnSelfAttention
from ..utils.esm.pretrained import load_model_and_alphabet
from ..utils.tranception.model_pytorch import get_tranception_tokenizer,TranceptionLMHeadModel
from ..utils.tranception.config import TranceptionConfig
from ..utils.model_utils import get_parameter_names

class AugmentedPropertyPredictor(nn.Module):
    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        print("Alphabet: {}".format(alphabet))
        print("Alphabet size: {}".format(self.alphabet_size))
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.target_names = self.args.target_config.keys() 
        self.MSA_sample_sequences = None 
        self.device = None
        self.model_type = args.model_type 
        if self.args.aa_embeddings == "MSA_Transformer" or self.args.aa_embeddings.startswith("ESM"):
            model, _ = load_model_and_alphabet(args.embedding_model_location)
            self.aa_embedding = model
            if self.args.aa_embeddings == "MSA_Transformer": self.args.seq_len = self.args.MSA_seq_len #If MSA does not cover full sequence length, we adjust seq_len param to be MSA_len (sequences truncated as needed in preprocessing)
        elif self.args.aa_embeddings == "Linear_embedding":
            self.aa_embedding = nn.Sequential(
                nn.Embedding(
                    self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
                ),
                nn.ReLU()
            )
        elif self.args.aa_embeddings == "One_hot_encoding":
            self.args.target_prediction_head == "One_hot_encoding"
        elif self.args.aa_embeddings == "Tranception":
            self.aa_embedding_dim = 1280
            config = json.load(open(args.embedding_model_location+os.sep+'config.json'))
            config = TranceptionConfig(**config)
            config.tokenizer = get_tranception_tokenizer()
            config.inference_time_retrieval_type = None
            config.retrieval_aggregation_mode = None
            self.aa_embedding = TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=args.embedding_model_location,config=config)
            self.config = config
        else:
            print("Error: Specified AA embedding invalid")
            sys.exit(0)

        if self.args.aa_embeddings != "One_hot_encoding": 
            self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
            self.dropout_module = nn.Dropout(self.args.dropout)

        if self.args.target_prediction_head == "AA_embeddings_mean_pooled":
            target_pred_input_dim = self.args.embed_dim
        elif self.args.target_prediction_head == "One_hot_encoding":
            target_pred_input_dim = (self.args.seq_len + 1) * self.alphabet_size if args.target_prediction_model!="CNN" else self.alphabet_size    #Add one for the BOS token
        else:
            print(self.args.target_prediction_head)
            print("Error: Specified embedding aggregation invalid")
            sys.exit(0)
        
        if args.target_prediction_model=="MLP":
            self.layer_pre_head = nn.Sequential(
                        nn.Linear(target_pred_input_dim, target_pred_input_dim),
                        nn.Dropout(self.args.dropout),
                        nn.ReLU()
            )
        elif args.target_prediction_model=="ConvBERT":
            configuration = ConvBertConfig(
                hidden_size = self.args.embed_dim,
                num_attention_heads = self.args.attention_heads if self.args.attention_heads is not None else 4,
                conv_kernel_size = self.args.conv_kernel_size,
                hidden_act = "gelu",
                hidden_dropout_prob = self.args.dropout,
                attention_probs_dropout_prob = self.args.dropout
            )
            self.layer_pre_head = ConvBertLayer(configuration)
        elif args.target_prediction_model=="CNN":
            self.layer_pre_head = nn.Sequential(
                nn.Conv1d(in_channels=target_pred_input_dim, out_channels=target_pred_input_dim, kernel_size = self.args.conv_kernel_size, padding='same'),
                nn.Dropout(self.args.dropout),
                nn.ReLU()
            )
            target_pred_input_dim = target_pred_input_dim if self.args.target_prediction_head != "One_hot_encoding" else target_pred_input_dim * (self.args.seq_len + 1)
        elif args.target_prediction_model=="light_attention":
            # Adapted from Stark et al (https://github.com/HannesStark/protein-localization)
            self.feature_convolution = nn.Conv1d(self.args.embed_dim, self.args.embed_dim, self.args.conv_kernel_size, stride=1, padding='same')
            self.attention_convolution = nn.Conv1d(self.args.embed_dim, self.args.embed_dim, self.args.conv_kernel_size, stride=1, padding='same')
            self.softmax = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(self.args.dropout)
            self.linear = nn.Sequential(
                nn.Linear(2 * self.args.embed_dim, 32),
                nn.Dropout(self.args.dropout),
                nn.ReLU(),
                nn.BatchNorm1d(32)
            )
            target_pred_input_dim = 32
        elif args.target_prediction_model=="linear":
            pass
        else:
            print("Error: Specified layer_pre_head invalid")
            sys.exit(0)

        if self.args.augmentation=="zero_shot_fitness_predictions_covariate":
            self.zero_shot_fitness_prediction_weight = nn.ModuleDict(
                { 
                    target_name: nn.Linear(1, self.args.target_config[target_name]["dim"], bias=False)
                    for target_name in self.target_names
                }
            )
            for target_name in self.target_names:
                torch.nn.init.constant_(self.zero_shot_fitness_prediction_weight[target_name].weight,1.0)

        self.target_pred_head = nn.ModuleDict(
                { 
                    target_name: nn.Linear(target_pred_input_dim, self.args.target_config[target_name]["dim"])
                    for target_name in self.target_names #If multiple targets, we learn a separate linear head for each separately
                }
        )
    
    def set_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        print("Model device: {}".format(self.device))

    def forward(self, tokens, zero_shot_fitness_predictions=None, sequence_embeddings=None, repr_layers=[]):
        if self.args.aa_embeddings == "MSA_Transformer" and self.args.sequence_embeddings_location is None:
            assert tokens.ndim == 3, "Finding dimension of tokens to be: {}".format(tokens.ndim)
            num_MSAs_in_batch, num_sequences_in_alignments, seqlen = tokens.size()
            batch_size = num_MSAs_in_batch
        else:
            assert tokens.ndim == 2, "Finding dimension of tokens to be: {}".format(tokens.ndim)
            batch_size, seqlen = tokens.size()
            
        if sequence_embeddings is not None:
            x = sequence_embeddings.to(self.device)
        else:
            if self.args.aa_embeddings == "MSA_Transformer":
                output = self.aa_embedding(tokens, repr_layers=[12])
                x = output["representations"][12][:] # B, N, L, D
                x = x[:,0,:,:] #In each MSA batch the first sequence is what we care about. The other MSA sequences were just to compute embeddings and logits
            elif self.args.aa_embeddings.startswith("ESM"):
                if self.args.aa_embeddings=="ESM1v":
                    last_layer_index = 33
                elif self.args.aa_embeddings=="ESM2_15B":
                    last_layer_index = 48
                elif self.args.aa_embeddings=="ESM2_3B":
                    last_layer_index = 36
                elif self.args.aa_embeddings=="ESM2_650M":
                    last_layer_index = 33
                output = self.aa_embedding(tokens, repr_layers=[last_layer_index])
                x = output["representations"][last_layer_index][:] # N, L, D
            elif self.args.aa_embeddings == "Tranception":
                processed_batch = {'input_ids': tokens, 'labels': tokens}
                output = self.aa_embedding(**processed_batch, return_dict=True, output_hidden_states=True)
                x = output.hidden_states[0]
            elif self.args.aa_embeddings =="Linear_embedding":
                x = self.aa_embedding(tokens)
            elif self.args.aa_embeddings == "One_hot_encoding":
                x = nn.functional.one_hot(tokens, num_classes=self.alphabet_size).view(batch_size,-1).float()
                if self.args.target_prediction_model == "CNN": x = x.view(batch_size,seqlen,self.alphabet_size)

        if self.args.aa_embeddings != "One_hot_encoding":
            x = self.emb_layer_norm_after(x)
            x = self.dropout_module(x)
        
        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if self.args.target_prediction_model == "CNN": 
            assert len(x.size())==3, "Size error input"
            N, L, D = x.size()
            x = x.permute(0,2,1) #N, D, L
            x = self.layer_pre_head(x)
            x = x.permute(0,2,1)
        elif self.args.target_prediction_model == "ConvBERT":
            x = self.layer_pre_head(x)[0]
        elif self.args.target_prediction_model=="light_attention":
            x = x.permute(0,2,1) #N, D, L
            o = self.feature_convolution(x)  
            o = self.dropout(o)
            attention = self.attention_convolution(x)
            o1 = torch.sum(o * self.softmax(attention), dim=-1)
            o2, _ = torch.max(o, dim=-1)
            o = torch.cat([o1, o2], dim=-1)
            x = self.linear(o)
        
        if self.args.target_prediction_head == "AA_embeddings_mean_pooled": x = x.mean(dim=-2)
        
        if self.args.target_prediction_model == "MLP": x = self.layer_pre_head(x)
        
        target_predictions = {}
        for target_name in self.target_names:
            target_predictions[target_name] = self.target_pred_head[target_name](x).view(-1,self.args.target_config[target_name]["dim"]).squeeze(dim=-1)
            if self.args.augmentation=="zero_shot_fitness_predictions_covariate":
                target_predictions[target_name] += self.zero_shot_fitness_prediction_weight[target_name](zero_shot_fitness_predictions).view(-1,self.args.target_config[target_name]["dim"]).squeeze(dim=-1)

        result = {"target_predictions": target_predictions, "representations": hidden_representations}
        
        return result
    
    def forward_with_uncertainty(self, tokens, zero_shot_fitness_predictions=None, sequence_embeddings=None, num_MC_dropout_samples=10):
        """
        Performs MC dropout to compute predictions and the corresponding uncertainties.
        Assumes 1D predictions (eg., prediction of continuous output).
        """
        self.eval() 
        for m in self.modules(): #Move all dropout layers in train mode to support MC dropout. Keep everything else in eval mode.
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        with torch.no_grad(): 
            predictions_dict = defaultdict(list)
            for _ in range(num_MC_dropout_samples):
                target_predictions_sample = self.forward(tokens, zero_shot_fitness_predictions=zero_shot_fitness_predictions, sequence_embeddings=sequence_embeddings)["target_predictions"]
                for target_name in self.target_names:
                    predictions_dict[target_name].append(target_predictions_sample[target_name])
            results_with_uncertainty={}
            for target_name in self.target_names:
                concatenated_target_pred = torch.cat([x.view(-1,1) for x in predictions_dict[target_name]],dim=-1)
                results_with_uncertainty[target_name] = {}
                results_with_uncertainty[target_name]['predictions_avg'] = concatenated_target_pred.mean(dim=-1)
                results_with_uncertainty[target_name]['uncertainty'] = concatenated_target_pred.std(dim=-1)
        return results_with_uncertainty

    @property
    def num_layers(self):
        return self.args.num_protein_npt_layers
    
    def max_tokens_per_msa_(self, value: int) -> None:
        """
        Batching attention computations when gradients are disabled as per MSA_Transformer
        Set this value to infinity to disable this behavior.
        """
        for module in self.modules():
            if isinstance(module, (RowSelfAttention, ColumnSelfAttention)):
                module.max_tokens_per_msa = value

    def prediction_loss(self, target_predictions, target_labels, label_smoothing=0.1):
        total_target_prediction_loss = 0.0
        target_prediction_loss_dict = {}
        for target_name in self.target_names:
            non_missing_target_indicator = ~torch.isnan(target_labels[target_name])
            if self.args.target_config[target_name]["type"]=="continuous":
                tgt_loss = MSELoss(reduction="mean")(target_predictions[target_name][non_missing_target_indicator], target_labels[target_name][non_missing_target_indicator])
            else:
                tgt_loss = CrossEntropyLoss(reduction="mean",label_smoothing=label_smoothing)(target_predictions[target_name][non_missing_target_indicator].view(-1, self.args.target_config[target_name]["dim"]), target_labels[target_name][non_missing_target_indicator].view(-1))
            target_prediction_loss_dict[target_name] = tgt_loss
            total_target_prediction_loss += tgt_loss
        return total_target_prediction_loss, target_prediction_loss_dict

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        Adapted from Huggingface Transformers library.
        """
        all_parameters = get_parameter_names(self, [nn.LayerNorm])
        decay_parameters = [name for name in all_parameters if ("bias" not in name and "pseudo_likelihood_weight" not in name and 'zero_shot_fitness_prediction_weight' not in name)]
        psl_decay_parameters = [name for name in all_parameters if ("bias" not in name and ("pseudo_likelihood_weight" in name or "zero_shot_fitness_prediction_weight" in name))]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.named_parameters() if n in psl_decay_parameters],
                    "weight_decay": 1e-8, #Small decay on pseudo-likelihood as in Hsu et al.
                },
                {
                    "params": [p for n, p in self.named_parameters() if (n not in decay_parameters and n not in psl_decay_parameters)],
                    "weight_decay": 0.0,
                },
            ]        
        optimizer_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "lr": self.args.max_learning_rate
            }
        optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
        return optimizer
    