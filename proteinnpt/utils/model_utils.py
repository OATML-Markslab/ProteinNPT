import math
import os
import time
import wandb
import random
import tqdm
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data_utils import collate_fn_protein_npt

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer. 
    Adapted from Huggingface Transformers library.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def get_learning_rate(training_step, num_warmup_steps=1000, num_total_training_steps=20000, max_learning_rate=3e-4, min_learning_rate=3e-5):
    """
    """
    if training_step <= num_warmup_steps:
        lr = (max_learning_rate * training_step) / num_warmup_steps
    elif training_step > num_total_training_steps:
        lr=min_learning_rate
    else:
        ratio_total_steps_post_warmup = (training_step - num_warmup_steps) / (num_total_training_steps - num_warmup_steps)
        cosine_scaler = 0.5 * (1.0 + math.cos(math.pi * ratio_total_steps_post_warmup))
        lr = min_learning_rate + cosine_scaler * (max_learning_rate - min_learning_rate)
    return lr

def learning_rate_scheduler(num_warmup_steps=1000, num_total_training_steps=20000, max_learning_rate=3e-4, min_learning_rate=3e-5):
    def get_lr(training_step):
        return get_learning_rate(training_step, num_warmup_steps, num_total_training_steps, max_learning_rate, min_learning_rate)
    return get_lr

def get_reconstruction_loss_coefficient(training_step, num_total_training_steps=20000, start_MLM_coefficient=0.5, end_MLM_coefficient=0.05):
    ratio_total_steps = training_step / num_total_training_steps
    cosine_scaler = 0.5 * (1.0 + math.cos(math.pi * ratio_total_steps))
    reconstruction_loss_coeff = end_MLM_coefficient + cosine_scaler * (start_MLM_coefficient - end_MLM_coefficient)
    return reconstruction_loss_coeff

def update_lr_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def collapse_triplets(s):
    triplets = s.split(":")
    positions = {}
    for triplet in triplets:
        pos = triplet[1:-1]
        aa1, aa2 = triplet[0], triplet[-1]
        if pos in positions:
            positions[pos] = positions[pos][:-1] + aa2
        else:
            positions[pos] = aa1 + pos + aa2
    s_new = ":".join([aa for aa in positions.values()])
    return s_new

class Trainer():
    def __init__(self, 
        model,
        args,
        train_data, 
        val_data=None,
        MSA_sequences=None, 
        MSA_weights=None,
        MSA_start_position=None,
        MSA_end_position=None,
        target_processing=None,
        distributed_training=False
        ):
        self.model = model
        self.args = args
        self.train_data = train_data
        self.val_data = val_data
        self.MSA_sequences = MSA_sequences
        self.MSA_weights = MSA_weights
        self.MSA_start_position = MSA_start_position
        self.MSA_end_position = MSA_end_position
        self.target_processing = target_processing
        self.distributed_training = distributed_training
            
    def train(self):
        """
        Returns the last value of training_step (useful in case of early stopping for isntance)
        """
        import proteinnpt
        self.model.train()
        self.model.cuda()
        self.model.set_device()

        if self.distributed_training:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_data)
        else:
            train_sampler = None
        
        #To ensure reproducibility with seed setting
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(0)
        train_loader = torch.utils.data.DataLoader(
                            dataset=self.train_data, 
                            batch_size=self.args.training_num_assay_sequences_per_batch_per_gpu, 
                            shuffle=(train_sampler is None),
                            num_workers=self.args.num_data_loaders_workers, 
                            pin_memory=True, 
                            sampler=train_sampler,
                            collate_fn=collate_fn_protein_npt,
                            worker_init_fn=seed_worker,
                            generator=g,
                        )
        optimizer = self.model.create_optimizer()
        scheduler = learning_rate_scheduler(
            num_warmup_steps=self.args.num_warmup_steps, 
            num_total_training_steps=self.args.num_total_training_steps, 
            max_learning_rate=self.args.max_learning_rate, 
            min_learning_rate=self.args.min_learning_rate
        )
        
        train_iterator = iter(train_loader)
        num_epochs = 0
        prior_log_time = time.time()
        total_train_time = 0
        log_train_total_loss = 0
        if self.model.model_type=="ProteinNPT":
            log_train_reconstruction_loss = 0
            log_train_num_masked_tokens = 0
            log_train_num_target_masked_tokens_dict = defaultdict(int)
        else:
            log_num_sequences_predicted = 0
        log_train_target_prediction_loss_dict = defaultdict(int)
        all_spearmans_eval_during_training = []
        max_average_spearman_across_targets = - math.inf
        if self.args.training_fp16: scaler = torch.cuda.amp.GradScaler()

        for training_step in tqdm.tqdm(range(1, self.args.num_total_training_steps+1)):
            optimizer.zero_grad(set_to_none=True)
            lr = scheduler(training_step)
            update_lr_optimizer(optimizer, lr)
            reconstruction_loss_coeff = get_reconstruction_loss_coefficient(training_step, num_total_training_steps=self.args.num_total_training_steps) if (self.model.model_type=="ProteinNPT" and not self.model.PNPT_no_reconstruction_error) else 0
            for gradient_accum_step in range(self.args.gradient_accumulation):
                try:
                    batch = next(train_iterator)
                except:
                    num_epochs +=1
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)
                if self.model.model_type=="ProteinNPT":
                    processed_batch = proteinnpt.proteinnpt.data_processing.process_batch(
                        batch = batch,
                        model = self.model,
                        alphabet = self.model.alphabet, 
                        args = self.args, 
                        MSA_sequences = self.MSA_sequences, 
                        MSA_weights = self.MSA_weights,
                        MSA_start_position = self.MSA_start_position, 
                        MSA_end_position = self.MSA_end_position,
                        target_processing = self.target_processing,
                        training_sequences = None,
                        proba_target_mask = 0.15,
                        proba_aa_mask = 0.15,
                        eval_mode = False,
                        device=self.model.device,
                        indel_mode=self.args.indel_mode
                    )
                else:
                    processed_batch = proteinnpt.baselines.data_processing.process_batch(
                        batch = batch,
                        model = self.model,
                        alphabet = self.model.alphabet, 
                        args = self.args, 
                        MSA_sequences = self.MSA_sequences, 
                        MSA_weights = self.MSA_weights,
                        MSA_start_position = self.MSA_start_position, 
                        MSA_end_position = self.MSA_end_position,
                        device=self.model.device,
                        eval_mode=False,
                        indel_mode=self.args.indel_mode
                    )

                if self.args.augmentation=="zero_shot_fitness_predictions_covariate":
                    zero_shot_fitness_predictions = processed_batch['target_labels']['zero_shot_fitness_predictions'].view(-1,1)
                    del processed_batch['target_labels']['zero_shot_fitness_predictions']
                else:
                    zero_shot_fitness_predictions = None
                
                if self.args.training_fp16:
                    with torch.cuda.amp.autocast():
                        if self.model.model_type=="ProteinNPT":
                            output = self.model(
                                tokens=processed_batch['masked_tokens'],
                                targets=processed_batch['masked_targets'],
                                zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                                sequence_embeddings=processed_batch['sequence_embeddings']
                            )
                            total_loss, reconstruction_loss, target_prediction_loss_dict = self.model.protein_npt_loss(
                                token_predictions_logits=output['logits_protein_sequence'], 
                                token_labels=processed_batch['token_labels'], 
                                target_predictions=output['target_predictions'], 
                                target_labels=processed_batch['target_labels'], 
                                MLM_reconstruction_loss_weight=reconstruction_loss_coeff, 
                                label_smoothing=self.args.label_smoothing
                            )
                        else:
                            output = self.model(
                                tokens=processed_batch['input_tokens'],
                                zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                                sequence_embeddings=processed_batch['sequence_embeddings']
                            )
                            total_loss, target_prediction_loss_dict = self.model.prediction_loss(
                                target_predictions=output["target_predictions"], 
                                target_labels=processed_batch['target_labels'],
                                label_smoothing=self.args.label_smoothing
                            )
                        scaler.scale(total_loss).backward()
                else:
                    if self.model.model_type=="ProteinNPT":
                        output = self.model(
                            tokens=processed_batch['masked_tokens'],
                            targets=processed_batch['masked_targets'],
                            zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                            sequence_embeddings=processed_batch['sequence_embeddings']
                        )
                        total_loss, reconstruction_loss, target_prediction_loss_dict = self.model.protein_npt_loss(
                            token_predictions_logits=output['logits_protein_sequence'], 
                            token_labels=processed_batch['token_labels'], 
                            target_predictions=output['target_predictions'], 
                            target_labels=processed_batch['target_labels'], 
                            MLM_reconstruction_loss_weight=reconstruction_loss_coeff, 
                            label_smoothing=self.args.label_smoothing
                        )
                        if total_loss.item() > 10.0 and training_step >= 100:
                            print("High training loss detected: {}".format(total_loss.item()))
                    else:
                        output = self.model(
                            tokens=processed_batch['input_tokens'],
                            zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                            sequence_embeddings=processed_batch['sequence_embeddings']
                        )
                        total_loss, target_prediction_loss_dict = self.model.prediction_loss(
                            target_predictions=output["target_predictions"], 
                            target_labels=processed_batch['target_labels'],
                            label_smoothing=self.args.label_smoothing
                        )
                    total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm_clip)
            # Taking optimizer update out of the inner loop to support gradient accumulation
            if self.args.training_fp16:
                with torch.cuda.amp.autocast():
                    scaler.step(optimizer)
                    scaler.update()
            else:
                optimizer.step()

            log_train_total_loss += total_loss
            if self.model.model_type=="ProteinNPT": 
                num_masked_tokens_in_batch = (~processed_batch['token_labels'].eq(-100)).sum().item()
                log_train_num_masked_tokens += num_masked_tokens_in_batch
                log_train_reconstruction_loss += reconstruction_loss * num_masked_tokens_in_batch
                for target_name in self.model.target_names:
                    if self.args.target_config[target_name]["type"]=="continuous":
                        num_masked_tokens_target_in_batch = processed_batch['masked_targets'][target_name][:,-1].eq(1.0).sum().item() # Masked targets are encoded by 1.0. Mask column is the very last one
                    else:
                        num_masked_tokens_target_in_batch = processed_batch['masked_targets'][target_name].eq(self.args.target_config[target_name]["dim"]).sum().item() # Index of mask is exactly self.args.target_config[target_name]["dim"] (largest value possible)
                    log_train_num_target_masked_tokens_dict[target_name] += num_masked_tokens_target_in_batch
                    log_train_target_prediction_loss_dict[target_name] += target_prediction_loss_dict[target_name] * num_masked_tokens_target_in_batch
            else:
                log_num_sequences_predicted += len(batch['mutant_mutated_seq_pairs'])
                for target_name in self.model.target_names:
                    log_train_target_prediction_loss_dict[target_name] += target_prediction_loss_dict[target_name] * len(batch['mutant_mutated_seq_pairs'])
            if training_step % self.args.num_logging_training_steps == 0 and self.args.use_wandb:
                time_end_step = time.time()
                delta_time_since_last_log = time_end_step - prior_log_time
                total_train_time += delta_time_since_last_log
                prior_log_time = time_end_step
                train_logs = {
                    "training_step": training_step, 
                    "step_time": delta_time_since_last_log / (self.args.num_logging_training_steps)
                }
                if self.model.model_type=="ProteinNPT": 
                    train_logs["train_total_loss_per_step"]: log_train_total_loss / self.args.num_logging_training_steps
                    train_logs["train_reconstruction_loss_per_masked_token"] = log_train_reconstruction_loss.item() / log_train_num_masked_tokens
                    for target_name in self.model.target_names:
                        train_logs["train_prediction_"+str(target_name)+"_loss_per_masked_token"] = log_train_target_prediction_loss_dict[target_name].item() / log_train_num_target_masked_tokens_dict[target_name]
                else:
                    train_logs["train_total_loss_per_seq"]: log_train_total_loss / log_num_sequences_predicted
                    for target_name in self.model.target_names:
                        train_logs["train_prediction_"+str(target_name)+"_loss_per_seq"] = log_train_target_prediction_loss_dict[target_name] / log_num_sequences_predicted
                wandb.log(train_logs)
                log_train_total_loss = 0
                log_train_target_prediction_loss_dict = defaultdict(int)
                if self.model.model_type=="ProteinNPT":
                    log_train_reconstruction_loss = 0
                    log_train_num_masked_tokens = 0
                    log_train_num_target_masked_tokens_dict = defaultdict(int)
                else:
                    log_num_sequences_predicted = 0 
                
            if self.args.save_model_checkpoint and (training_step % self.args.num_saving_training_steps) == 0:
                if not os.path.exists(self.args.model_location): os.mkdir(self.args.model_location)
                if not os.path.exists(self.args.model_location + os.sep + 'checkpoint-'+str(training_step)): os.mkdir(self.args.model_location + os.sep + 'checkpoint-'+str(training_step))
                torch.save({
                    'training_step': training_step,
                    'args': self.args,
                    'state_dict': self.model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, 
                    self.args.model_location + os.sep + 'checkpoint-'+str(training_step) + os.sep + 'checkpoint.t7'
                )
            
            if training_step % self.args.num_eval_steps == 0 and self.args.use_validation_set:
                if self.model.model_type=="ProteinNPT":
                    eval_results = self.eval(
                        test_data=self.val_data,
                        train_data=self.train_data,
                        reconstruction_loss_weight=0.0,
                        output_all_predictions=True
                    )
                else:
                    eval_results = self.eval(
                        test_data=self.val_data, 
                        output_all_predictions=True
                    )
                eval_logs = {"Training step": training_step} 
                eval_logs['Eval total loss per seq.'] = eval_results['eval_total_loss']
                average_spearman_across_targets = 0 #If early stopping based on validation spearman and multiple targets, we check that avg spearman is not decreasing for a certain # of times in a row
                for target_name in self.model.target_names:
                    eval_logs['Eval loss '+str(target_name)+' per seq.'] = eval_results['eval_target_prediction_loss_dict'][target_name]
                    if self.args.target_config[target_name]["dim"]==1:
                        eval_logs['Eval spearman '+target_name] = spearmanr(eval_results['output_scores']['predictions_'+target_name], eval_results['output_scores']['labels_'+target_name])[0]
                    else:
                        # In the categorical setting, we predict the spearman between the logits of the category with highest index, and target value indices. This is meaningul in the binary setting. Use with care if 3 categories or more.
                        eval_logs['Eval spearman '+target_name] = spearmanr(eval_results['output_scores']['predictions_'+target_name][:,-1], eval_results['output_scores']['labels_'+target_name])[0]
                    average_spearman_across_targets += eval_logs['Eval spearman '+target_name]
                average_spearman_across_targets /= len(self.model.target_names)
                print(" | ".join([key + ": "+str(round(eval_logs[key],5)) for key in eval_logs.keys()]))
                if self.args.use_wandb: wandb.log(eval_logs)
                # Early stopping
                all_spearmans_eval_during_training.append(average_spearman_across_targets)
                if average_spearman_across_targets > max_average_spearman_across_targets: max_average_spearman_across_targets = average_spearman_across_targets
                if (training_step >= 1000) and (self.args.early_stopping_patience is not None) and (np.array(all_spearmans_eval_during_training)[-self.args.early_stopping_patience:].max() < max_average_spearman_across_targets):
                    print("Early stopping. Training step: {}. Total eval loss: {}. Avg spearman: {}".format(training_step, eval_results['eval_total_loss'], average_spearman_across_targets))
                    break
                self.model.train() #Move back the model to train mode after eval loop
        trainer_final_status = {
            'total_training_steps': training_step,
            'total_train_time': total_train_time,
            'total_training_epochs': num_epochs
        }
        return trainer_final_status

    def eval(self, test_data, output_all_predictions=False, need_head_weights=False, train_data = None, reconstruction_loss_weight=0.5, selected_indices_seed=0):
        """
        total_eval_target_prediction_loss is the sum of all target prediction losses across all targets
        total_eval_target_prediction_loss contains the breakdown by target
        num_predicted_targets has the number of predicted items
        output_scores is a dict with sequences, predictions and labels
        """
        import proteinnpt
        self.model.eval()
        self.model.cuda()
        self.model.set_device()
        with torch.no_grad():
            eval_loader = torch.utils.data.DataLoader(
                                dataset=test_data, 
                                batch_size=self.args.eval_num_sequences_to_score_per_batch_per_gpu, 
                                shuffle=False,
                                num_workers=self.args.num_data_loaders_workers,
                                pin_memory=True,
                                collate_fn=collate_fn_protein_npt
                            )
            eval_iterator = iter(eval_loader)
            
            num_eval_batches = 0
            eval_total_loss = 0
            if self.model.model_type=="ProteinNPT": 
                eval_reconstruction_loss = 0
                eval_num_masked_tokens = 0
                eval_num_masked_targets = defaultdict(int)
            else:
                num_predicted_targets = 0
            eval_target_prediction_loss_dict = defaultdict(int)
            output_scores = defaultdict(list) if output_all_predictions else None

            if need_head_weights:
                col_attentions=[]
                row_attentions=[]

            for batch in tqdm.tqdm(eval_iterator):
                if output_all_predictions: 
                    output_scores['mutated_sequence'] += list(zip(*batch['mutant_mutated_seq_pairs']))[1]
                    output_scores['mutant'] += list(zip(*batch['mutant_mutated_seq_pairs']))[0]
                if self.model.model_type=="ProteinNPT":
                    processed_batch = proteinnpt.proteinnpt.data_processing.process_batch(
                        batch = batch,
                        model = self.model,
                        alphabet = self.model.alphabet, 
                        args = self.args, 
                        MSA_sequences = self.MSA_sequences, 
                        MSA_weights = self.MSA_weights,
                        MSA_start_position = self.MSA_start_position, 
                        MSA_end_position = self.MSA_end_position,
                        target_processing = self.target_processing,
                        training_sequences = train_data,
                        proba_target_mask = 1.0, 
                        proba_aa_mask = 0.0,
                        eval_mode = True,
                        device=self.model.device,
                        selected_indices_seed=selected_indices_seed,
                        indel_mode=self.args.indel_mode
                    )
                else:
                    processed_batch = proteinnpt.baselines.data_processing.process_batch(
                        batch = batch,
                        model = self.model,
                        alphabet = self.model.alphabet, 
                        args = self.args, 
                        MSA_sequences = self.MSA_sequences, 
                        MSA_weights = self.MSA_weights,
                        MSA_start_position = self.MSA_start_position, 
                        MSA_end_position = self.MSA_end_position,
                        device=self.model.device,
                        eval_mode=True,
                        indel_mode=self.args.indel_mode
                    )
                if self.args.augmentation=="zero_shot_fitness_predictions_covariate":
                    zero_shot_fitness_predictions = processed_batch['target_labels']['zero_shot_fitness_predictions'].view(-1,1)
                    del processed_batch['target_labels']['zero_shot_fitness_predictions']
                else:
                    zero_shot_fitness_predictions = None
                
                if self.model.model_type=="ProteinNPT":
                    output = self.model(
                        tokens=processed_batch['masked_tokens'],
                        targets=processed_batch['masked_targets'],
                        zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                        sequence_embeddings=processed_batch['sequence_embeddings'],
                        need_head_weights=need_head_weights
                    )
                    batch_loss, batch_reconstruction_loss, batch_target_prediction_loss_dict = self.model.protein_npt_loss(
                        token_predictions_logits=output['logits_protein_sequence'], 
                        token_labels=processed_batch['token_labels'], 
                        target_predictions=output['target_predictions'], 
                        target_labels=processed_batch['target_labels'], 
                        MLM_reconstruction_loss_weight=reconstruction_loss_weight, 
                        label_smoothing=self.args.label_smoothing
                    )
                    if batch_loss.item() > 10.0:
                        print("High eval loss detected: {}".format(batch_loss.item()))
                else:
                    output = self.model(
                        tokens=processed_batch['input_tokens'],
                        zero_shot_fitness_predictions=zero_shot_fitness_predictions,
                        sequence_embeddings=processed_batch['sequence_embeddings']
                    )
                    batch_loss, batch_target_prediction_loss_dict = self.model.prediction_loss(
                        target_predictions=output["target_predictions"], 
                        target_labels=processed_batch['target_labels'],
                        label_smoothing=self.args.label_smoothing
                    )
                
                num_eval_batches += 1
                eval_total_loss += batch_loss.item()
                if self.model.model_type=="ProteinNPT":
                    num_masked_tokens_in_batch = (~processed_batch['token_labels'].eq(100)).sum().item()  #processed_batch['masked_tokens'].eq(self.model.alphabet.mask_idx).sum().item()
                    eval_num_masked_tokens += num_masked_tokens_in_batch
                    eval_reconstruction_loss += batch_reconstruction_loss.item() * num_masked_tokens_in_batch
                    for target_name in self.model.target_names:
                        if self.args.target_config[target_name]["type"]=="continuous":
                            num_masked_tokens_target_in_batch = processed_batch['masked_targets'][target_name][:,-1].eq(1.0).sum().item() # Masked targets are encoded by 1.0. Mask column is the very last one
                        else:
                            num_masked_tokens_target_in_batch = processed_batch['masked_targets'][target_name].eq(self.args.target_config[target_name]["dim"]).sum().item() # Index of mask is exactly self.args.target_config[target_name]["dim"] (largest value possible)
                        eval_num_masked_targets[target_name] += num_masked_tokens_target_in_batch
                        eval_target_prediction_loss_dict[target_name] += batch_target_prediction_loss_dict[target_name].item() * num_masked_tokens_target_in_batch
                else:
                    num_predicted_targets += len(batch['mutant_mutated_seq_pairs'])
                    for target_name in self.model.target_names:
                        eval_target_prediction_loss_dict[target_name] += batch_target_prediction_loss_dict[target_name].item() * len(batch['mutant_mutated_seq_pairs'])
                if output_all_predictions:
                    num_of_mutated_seqs_to_score = processed_batch['num_of_mutated_seqs_to_score'] if self.model.model_type=="ProteinNPT" else len(processed_batch['mutant_mutated_seq_pairs'])
                    for target_name in self.model.target_names:
                        output_scores['predictions_'+target_name] += list(output["target_predictions"][target_name][:num_of_mutated_seqs_to_score].cpu().numpy())
                        output_scores['labels_'+target_name] += list(processed_batch['target_labels'][target_name][:num_of_mutated_seqs_to_score].cpu().numpy())
                if need_head_weights:
                    col_attentions.append(output["col_attentions"])
                    row_attentions.append(output["row_attentions"])

            output_scores = pd.DataFrame.from_dict(output_scores)
            output_scores_numeric_cols = [col_name for col_name in output_scores.columns if col_name not in ['mutated_sequence']]
            output_scores = output_scores[output_scores_numeric_cols]
            assert len(output_scores)==output_scores['mutant'].nunique()
            mutated_seqs_dict = {}
            mutant_mutated_seqs = list(zip(*test_data['mutant_mutated_seq_pairs']))
            mutated_seqs_dict['mutant'] = mutant_mutated_seqs[0]
            mutated_seqs_dict['mutated_sequence'] = mutant_mutated_seqs[1]
            mutated_seqs_df = pd.DataFrame.from_dict(mutated_seqs_dict)
            output_scores = pd.merge(output_scores, mutated_seqs_df, on='mutant', how='left')

        # Normalization
        for target_name in self.model.target_names:
            if self.model.model_type=="ProteinNPT":
                eval_target_prediction_loss_dict[target_name] /= eval_num_masked_targets[target_name] # We track exactly how many targets were masked across batches to account for potential discrepancies across batches (eg., last abtch may not have the same number of labels)
            else:
                eval_target_prediction_loss_dict[target_name] /= num_predicted_targets
        eval_results = {
            'eval_total_loss':eval_total_loss / num_eval_batches,
            'eval_target_prediction_loss_dict': eval_target_prediction_loss_dict,
            'output_scores': output_scores
        }
        if need_head_weights:
            print("dimension of first attention column {}".format(col_attentions[0].shape))
            eval_results['col_attentions'] = torch.stack(col_attentions, dim=0).cpu().numpy()
            eval_results['row_attentions'] = torch.stack(row_attentions, dim=0).cpu().numpy()
        
        if self.model.model_type=="ProteinNPT":
            eval_results['eval_reconstruction_loss'] = eval_reconstruction_loss / eval_num_masked_tokens
            eval_results['eval_num_masked_tokens'] = eval_num_masked_tokens
            eval_results['eval_num_masked_targets'] = eval_num_masked_targets
        else:
            eval_results['eval_num_predicted_targets'] = num_predicted_targets
        return eval_results