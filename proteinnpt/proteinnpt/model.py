import sys,os
import json
from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import ConvBertConfig, ConvBertLayer

from ..utils.esm.modules import (
    AxialTransformerLayer,
    LearnedPositionalEmbedding,
    RobertaLMHead,
    ESM1bLayerNorm,
)
from ..utils.esm.pretrained import load_model_and_alphabet
from ..utils.esm.axial_attention import RowSelfAttention, ColumnSelfAttention
from ..utils.tranception.config import TranceptionConfig
from ..utils.tranception.model_pytorch import TranceptionLMHeadModel
from ..utils.model_utils import get_parameter_names

class ProteinNPTModel(nn.Module):
    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.target_names_input = self.args.target_config.keys()
        self.target_names = [x for x in self.args.target_config.keys() if self.args.target_config[x]["in_NPT_loss"]]
        self.num_targets_input = len(self.target_names_input) #Includes all targets, incl. zero-shot fitness predictions
        self.num_targets = len(self.target_names) #Number of actual targets we want to predict
        self.MSA_sample_sequences = None
        self.training_sample_sequences_indices = None
        self.device = None
        self.optimizer = None
        self.model_type = args.model_type
        self.PNPT_ensemble_test_num_seeds = -1
        self.PNPT_no_reconstruction_error = False
        self.deactivate_col_attention = False
        self.tranception_attention = False
        
        assert self.args.embed_dim % self.args.attention_heads ==0, "Embedding size {} needs to be a multiple of number of heads {}".format(self.args.embed_dim, self.args.attention_heads)
        if self.args.aa_embeddings=="MSA_Transformer" or args.aa_embeddings.startswith("ESM"):
            model, _ = load_model_and_alphabet(args.embedding_model_location)
            self.aa_embedding = model
            self.aa_embedding_dim = self.aa_embedding.embed_tokens.weight.shape[-1]
        elif self.args.aa_embeddings == "Tranception":
            self.aa_embedding_dim = 1280
            config = json.load(open(args.embedding_model_location+os.sep+'config.json'))
            config = TranceptionConfig(**config)
            config.tokenizer = self.alphabet
            config.inference_time_retrieval_type = None
            config.retrieval_aggregation_mode = None
            self.aa_embedding = TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=args.embedding_model_location,config=config)
        elif self.args.aa_embeddings == "Linear_embedding":
            self.aa_embedding = nn.Embedding(
                                    self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
                                )
            self.aa_positions_embedding = LearnedPositionalEmbedding(
                                    self.args.max_positions,
                                    self.args.embed_dim,
                                    self.padding_idx,
                                )
            self.aa_embedding_dim = self.args.embed_dim

        if self.aa_embedding_dim != self.args.embed_dim: #Need to project internally
            self.token_embedding_projection = nn.Linear(
                        self.aa_embedding_dim,
                        self.args.embed_dim
                    )
            self.token_embedding_expansion = nn.Linear(
                        self.args.embed_dim,
                        self.aa_embedding_dim
                    )

        self.target_embedding =  nn.ModuleDict(
            {   
                target_name:
                nn.Linear(
                    self.args.target_config[target_name]["dim"] + 1, #Need to add one as we append the mask flag to each input target 
                    self.args.embed_dim
                )
                if self.args.target_config[target_name]["type"]=="continuous"
                else 
                nn.Embedding(
                    self.args.target_config[target_name]["dim"] + 1, #Size of the dictionary of embeddings. Need to add 1 for the mask flag as well
                    self.args.embed_dim
                )
                for target_name in self.target_names_input
            }
        )
        
        self.dropout_module = nn.Dropout(self.args.dropout)

        self.layers = nn.ModuleList(
                [
                    AxialTransformerLayer(
                        self.args.embed_dim,
                        self.args.ffn_embed_dim,
                        self.args.attention_heads,
                        self.args.dropout,
                        self.args.attention_dropout,
                        self.args.activation_dropout,
                        getattr(self.args, "max_tokens_per_msa", self.args.max_tokens_per_msa),
                        self.deactivate_col_attention,
                        self.tranception_attention,
                        self.num_targets_input,
                    )
                    for _ in range(self.args.num_protein_npt_layers)
                ]
            )
        self.emb_layer_norm_before = ESM1bLayerNorm(self.args.embed_dim)
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        
        if self.args.aa_embeddings=="MSA_Transformer" or args.aa_embeddings.startswith("ESM"):
            weight = self.aa_embedding.embed_tokens.weight
        elif self.args.aa_embeddings == "Tranception":
            weight = self.aa_embedding.lm_head.weight
        else:
            weight = self.aa_embedding.weight

        self.lm_head = RobertaLMHead(
            embed_dim=self.aa_embedding_dim,
            output_dim=self.alphabet_size,
            weight=weight
        )
        
        target_pred_input_dim = self.args.embed_dim

        if args.target_prediction_model=="MLP": 
            self.layer_pre_head = nn.ModuleDict(
                {
                    target_name:
                        nn.Sequential(
                        nn.Linear(target_pred_input_dim, target_pred_input_dim),
                        nn.Dropout(self.args.dropout),
                        nn.ReLU()
                        ) 
                    for target_name in self.target_names
                }
            )
            
        if args.target_prediction_model=="ConvBERT":
            configuration = ConvBertConfig(
                hidden_size = self.args.embed_dim,
                num_attention_heads = self.args.attention_heads,
                conv_kernel_size = self.args.conv_kernel_size,
                hidden_act = "gelu",
                hidden_dropout_prob = self.args.dropout,
                attention_probs_dropout_prob = self.args.dropout
            )
            self.layer_pre_head = ConvBertLayer(configuration)
        
        if args.target_prediction_model=="CNN":
            self.layer_pre_head = nn.Sequential(
                nn.Conv1d(in_channels=target_pred_input_dim, out_channels=target_pred_input_dim, kernel_size = self.args.conv_kernel_size, padding='same'),
                nn.Dropout(self.args.dropout),
                nn.ReLU()
            )
        
        if self.args.target_prediction_head == "Target_embeddings_only":
            target_pred_input_dim = target_pred_input_dim
        elif self.args.target_prediction_head == "Target_embeddings_and_AA_embeddings_mean_pooled":
            target_pred_input_dim = target_pred_input_dim * (1 + self.num_targets_input)

        if self.args.augmentation=="zero_shot_fitness_predictions_covariate":
            self.zero_shot_fitness_prediction_weight = nn.ModuleDict(
                { 
                    target_name: nn.Linear(1, self.args.target_config[target_name]["dim"], bias=False)
                    for target_name in self.target_names
                }
            )
            for target_name in self.target_names:
                torch.nn.init.constant_(self.zero_shot_fitness_prediction_weight[target_name].weight,1e-4)

        self.target_pred_head = nn.ModuleDict(
                { 
                    target_name: nn.Linear(target_pred_input_dim, self.args.target_config[target_name]["dim"])
                    for target_name in self.target_names
                }
        )
    
    def set_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        print("Model device: {}".format(self.device))
        
    def forward(self, tokens, targets=None, zero_shot_fitness_predictions=None, sequence_embeddings=None, repr_layers=[], need_head_weights=False):
        padding_mask = tokens.eq(self.padding_idx) 
        if not padding_mask.any(): padding_mask = None
        
        if self.args.aa_embeddings == "MSA_Transformer" and self.args.sequence_embeddings_location is None: #If loading MSAT embeddings from disk, we have dropped one dim already
            assert tokens.ndim == 3, "Finding dimension of tokens to be: {}".format(tokens.ndim)
            num_MSAs_in_batch, num_sequences_in_alignments, seqlen = tokens.size() # N, B, L (seqs with labels, seqs in MSA, seq length)
            batch_size = num_MSAs_in_batch
        else:
            assert tokens.ndim == 2, "Finding dimension of tokens to be: {}".format(tokens.ndim)
            batch_size, seqlen = tokens.size() # N, L (seqs with labels, seq length)
        
        if sequence_embeddings is not None:
            x = sequence_embeddings.to(self.device)
        else:
            if self.args.aa_embeddings == "MSA_Transformer":
                output = self.aa_embedding(tokens, repr_layers=[12])
                x = output["representations"][12][:] # N, B, L, D
                x = x[:,0,:,:] # N, L, D. #In each MSA batch the first sequence is what we care about. The other MSA sequences were just to compute embeddings and logits
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
            elif self.args.aa_embeddings =="Linear_embedding":
                x = self.aa_embedding(tokens)
                x = x + self.aa_positions_embedding(tokens.view(batch_size, seqlen)).view(x.size()) # Need position embedding in PNPT since we will apply axial attention
            else:
                print("AA embeddings not recognized")
                sys.exit(0)
    
        if self.aa_embedding_dim != self.args.embed_dim: x = self.token_embedding_projection(x)
        
        if self.args.target_prediction_head != "Target_embeddings_and_AA_embeddings_mean_pooled": #We mix AA embeddings pre NPT
            if self.args.target_prediction_model == "CNN": 
                assert len(x.size())==3, "Size error input"
                N, L, D = x.size()
                x = x.permute(0,2,1) #N, D, L
                x = self.layer_pre_head(x)
                x = x.permute(0,2,1)
            elif self.args.target_prediction_model == "ConvBERT":
                x = self.layer_pre_head(x)[0]

        x = x.view(1, batch_size, seqlen, self.args.embed_dim) # 1, N, L, D
        
        #Dimensions for each target (there are self.num_targets of them):
        y = []
        for target_name in self.target_names_input:
            if self.args.target_config[target_name]["type"]=="continuous":
                num_sequences_with_target, dim_targets = targets[target_name].shape # N, D_t #In most cases dim_targets = D_t = 2 (original dimension of continuous input + 1 dim for mask)
            else:
                num_sequences_with_target = targets[target_name].shape[0] #Input categorical targets is unidmensional ie a vector of category indices
                targets[target_name] = targets[target_name].long() #Ensure we cast to integer before passing to Embedding layer for categorical targets
            y.append(self.target_embedding[target_name](targets[target_name]).view(num_sequences_with_target,1,self.args.embed_dim))
        y = torch.cat(y, dim=-2) #concatenate across second to last dimension # N, num_targets, D
        assert y.shape == (num_sequences_with_target, self.num_targets_input, self.args.embed_dim), "Error in y shape: {}".format(y.shape)
        y = y.view(1, num_sequences_with_target, self.num_targets_input, self.args.embed_dim) # 1, N, num_targets, D
        
        #Concatenate AA tokens and targets
        x = torch.cat((x,y),dim=-2) # 1, N, (L+num_targets), D
        x = self.emb_layer_norm_before(x)
        x = self.dropout_module(x)

        if padding_mask is not None:
            B, N, L, D = x.shape
            padding_mask_with_targets = torch.zeros(B, N, L).to(x.device)
            padding_mask_with_targets[...,:seqlen] = padding_mask
            padding_mask = padding_mask_with_targets
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
            padding_mask = padding_mask.bool()
        
        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers: hidden_representations[0] = x
        if need_head_weights:
            row_attn_weights = []
            col_attn_weights = []

        # 1 x N x L x D -> N x L x 1 x D
        x = x.permute(1, 2, 0, 3)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if need_head_weights:
                x, col_attn, row_attn = x
                col_attn_weights.append(col_attn.permute(2, 0, 1, 3, 4).cpu())
                row_attn_weights.append(row_attn.permute(1, 0, 2, 3).cpu())
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.permute(2, 0, 1, 3)
        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # N x L x 1 x D -> 1 x N x L x D
        assert x.shape == (1, num_sequences_with_target, seqlen + self.num_targets_input, self.args.embed_dim), "Error with axial transformer"
        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers: hidden_representations[layer_idx + 1] = x
        
        # Loss over NPT MLM objective
        if self.aa_embedding_dim != self.args.embed_dim:
            logits_protein_sequence = self.lm_head(self.token_embedding_expansion(x[...,:seqlen,:]))
        else:
            logits_protein_sequence = self.lm_head(x[...,:seqlen,:]) #Remove dependency on targets for final AA predictions. logits size: (1, N, L, Vocab)
        
        x = x.view(num_sequences_with_target, seqlen + self.num_targets_input, self.args.embed_dim)
        x, y = x[:,:seqlen,:], x[:,seqlen:,:] # (N,L,D) and (N,num_targets,D)
        assert y.shape == (num_sequences_with_target, self.num_targets_input, self.args.embed_dim)
        if self.args.target_prediction_head == "Target_embeddings_and_AA_embeddings_mean_pooled": 
            if self.args.target_prediction_model == "CNN": 
                assert len(x.size())==3, "Size error input"
                N, L, D = x.size()
                x = x.permute(0,2,1) #N, D, L
                x = self.layer_pre_head(x)
                x = x.permute(0,2,1)
            elif self.args.target_prediction_model == "ConvBERT":
                x = self.layer_pre_head(x)[0]
            x = x.mean(dim=-2) # N, D
            y = y.view(num_sequences_with_target,self.num_targets_input * self.args.embed_dim)
            y = torch.cat((x,y),dim=-1) # N, (1+num_targets) * D
        
        target_predictions = {}
        for target_index, target_name in enumerate(self.target_names):
            if self.args.target_prediction_head == "Target_embeddings_and_AA_embeddings_mean_pooled": 
                target_predictions[target_name] = self.target_pred_head[target_name](y).view(-1,self.args.target_config[target_name]["dim"]).squeeze(dim=-1) #We use the concatenated X and target embeddings (all of them) to predict each target
            else:
                if self.args.target_prediction_model == "MLP": y[:,target_index,:] = self.layer_pre_head[target_name](y[:,target_index,:])
                target_predictions[target_name] = self.target_pred_head[target_name](y[:,target_index,:]).view(-1,self.args.target_config[target_name]["dim"]).squeeze(dim=-1) #input the embedding with the relevant target_index
            if self.args.augmentation=="zero_shot_fitness_predictions_covariate":
                target_predictions[target_name] += self.zero_shot_fitness_prediction_weight[target_name](zero_shot_fitness_predictions).view(-1,self.args.target_config[target_name]["dim"]).squeeze(dim=-1)
            
        result = {"logits_protein_sequence": logits_protein_sequence, "target_predictions": target_predictions, "representations": hidden_representations}
        
        if need_head_weights:
            col_attentions = torch.stack(col_attn_weights, 1)
            row_attentions = torch.stack(row_attn_weights, 1)
            result["col_attentions"] = col_attentions
            result["row_attentions"] = row_attentions

        return result

    def forward_with_uncertainty(self, tokens, targets, zero_shot_fitness_predictions=None, sequence_embeddings=None, num_MC_dropout_samples=10, number_of_mutated_seqs_to_score=None):
        """
        Performs MC dropout to compute predictions and the corresponding uncertainties.
        Assumes 1D predictions (eg., prediction of continuous output)
        """
        self.eval() 
        for m in self.modules(): #Move all dropout layers in train mode to support MC dropout. Keep everything else in eval mode.
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        with torch.no_grad():
            predictions_dict = defaultdict(list)
            for _ in range(num_MC_dropout_samples):
                target_predictions_sample = self.forward(tokens, targets, zero_shot_fitness_predictions=zero_shot_fitness_predictions, sequence_embeddings=sequence_embeddings)["target_predictions"]
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

    def protein_npt_loss(self, token_predictions_logits, token_labels, target_predictions, target_labels, MLM_reconstruction_loss_weight, label_smoothing=0.0):
        target_prediction_loss_weight = 1.0 - MLM_reconstruction_loss_weight
        total_loss = 0.0
        if (token_labels is not None) and (MLM_reconstruction_loss_weight > 0.0):
            if self.args.aa_embeddings == "MSA_Transformer" and self.args.sequence_embeddings_location is None: token_labels = token_labels[:,0,:] #Only keep the token labels for seq to score. Drops the token labels for MSA sequences
            reconstruction_loss = CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)(token_predictions_logits.reshape(-1, self.alphabet_size), token_labels.reshape(-1))
            total_loss += MLM_reconstruction_loss_weight * reconstruction_loss
        else:
            reconstruction_loss = torch.tensor(0.0)
        target_prediction_loss = {}
        for target_name in self.target_names:
            if self.args.target_config[target_name]["in_NPT_loss"]:
                loss_masked_targets = ~target_labels[target_name].eq(-100) #Masked items are the ones for which the label was not set to -100
                if loss_masked_targets.sum()==0 or torch.isnan(target_labels[target_name][loss_masked_targets]).sum() > 0: #First condition true if we dont mask anything (eg., all target missing at eval). Second condition true if we force-mask one value at train time (to satisfy min_num_labels_masked in mast_target()) and corresponding target value is missing
                    tgt_loss = torch.tensor(0.0)
                else:
                    if self.args.target_config[target_name]["type"]=="continuous":
                        tgt_loss = MSELoss(reduction="mean")(target_predictions[target_name][loss_masked_targets], target_labels[target_name][loss_masked_targets])
                    else:
                        tgt_loss = CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)(target_predictions[target_name][loss_masked_targets].view(-1, self.args.target_config[target_name]["dim"]), target_labels[target_name][loss_masked_targets].view(-1).long()) # Note: we dont add one to the # of categories in the CE loss here (we dont predict <mask>)
                if torch.isnan(tgt_loss).sum() > 0:
                    print("Detected nan loss")
                    print(target_predictions[target_name])
                target_prediction_loss[target_name] = tgt_loss
                
                total_loss += target_prediction_loss_weight * target_prediction_loss[target_name]
        return total_loss, reconstruction_loss, target_prediction_loss

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        Adapted from Huggingface Transformers library.
        """
        if self.optimizer is None:
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
    