import os,gc
import json
import argparse
import random
import numpy as np
import pandas as pd
import wandb
import torch
from collections import defaultdict
from torch.nn import CrossEntropyLoss

from proteinnpt.proteinnpt.model import ProteinNPTModel
from proteinnpt.baselines.model import AugmentedPropertyPredictor
from proteinnpt.utils.esm.data import Alphabet
from proteinnpt.utils.tranception.model_pytorch import get_tranception_tokenizer
from proteinnpt.utils.data_utils import get_train_val_test_data, standardize, pnpt_count_non_nan, pnpt_spearmanr
from proteinnpt.utils.msa_utils import process_MSA
from proteinnpt.utils.model_utils import Trainer

def setup_config_and_paths(args):
    # All parameters that are not defined by end user are fetched from the config file
    if args.model_config_location is not None:
        args.main_config=json.load(open(args.model_config_location))
        args_setup_from_config=set([])
        for key in args.main_config:
            if args.__dict__[key] is None:
                args.__dict__[key] = args.main_config[key]
                args_setup_from_config.add(key)

    # File paths config
    for local_path in ['embedding_model_location','MSA_data_folder','MSA_weight_data_folder','path_to_hhfilter']:
        if getattr(args, local_path) and local_path in args_setup_from_config:
            setattr(args, local_path, args.data_location + os.sep + getattr(args, local_path))
    if not os.path.exists(args.data_location + os.sep + 'model_predictions'): os.mkdir(args.data_location + os.sep + 'model_predictions')
    if not os.path.exists(args.data_location + os.sep + 'checkpoint'): os.mkdir(args.data_location + os.sep + 'checkpoint')
    args.output_scores_location = args.data_location + os.sep + 'model_predictions' + os.sep + args.model_name_suffix
    if not os.path.exists(args.output_scores_location): os.mkdir(args.output_scores_location)
    args.model_location = args.data_location + os.sep + 'checkpoint' + os.sep + args.model_name_suffix
    if not os.path.exists(args.model_location): os.mkdir(args.model_location)
    if args.assay_data_location and not args.assay_data_folder: args.assay_data_folder = [ os.sep.join(args.assay_data_location.split(os.sep)[:-1]) ] # args.assay_data_folder is a list

    # Target config
    args.target_config=json.load(open(args.target_config_location))
    zero_shot_predictions_mapping={
            "MSA_Transformer_pred": "MSA_Transformer_ensemble",
            "ESM1v_pred": "ESM1v_ensemble",
            "ESM2_15B_pred": "ESM2_15B",
            "ESM2_3B_pred": "ESM2_3B",
            "ESM2_650M_pred": "ESM2_650M",
            "TranceptEVE_pred": "TranceptEVE_L",
            "Tranception_pred": "Tranception_L",
            "DeepSequence_pred": "DeepSequence_ensemble"
        }
    if args.model_type=="ProteinNPT": zero_shot_predictions_mapping["ProteinNPT"]=zero_shot_predictions_mapping[args.aa_embeddings+"_pred"]
    if args.augmentation=="zero_shot_fitness_predictions_auxiliary_labels": # Add auxiliary label to target_config
        assert args.zero_shot_fitness_predictions_location is not None, "Location of zero-shot fitness predictions to use as auxiliary labels not properly referenced"
        print("Using zero-shot fitness predictions as auxiliary labels")
        args.target_config["zero_shot_fitness_predictions"] = {
            "type": "continuous",
            "dim": 1,
            "var_name": zero_shot_predictions_mapping[args.model_type], #Select the relevant model for zero-shot fitness predictions
            "location": args.zero_shot_fitness_predictions_location,
            "in_NPT_loss": False,
            "main_target": False
        }
        args.augmentation_short="auxiliary"
    elif args.augmentation=="zero_shot_fitness_predictions_covariate": # Will use zero-shot fitness predictions as an additional model covariate
        assert args.zero_shot_fitness_predictions_location is not None, "Location of zero-shot fitness predictions to use as model covariate not properly referenced"
        print("Using zero-shot fitness predictions as covariate")
        args.augmentation_short="covariate"
        args.zero_shot_fitness_predictions_var_name = zero_shot_predictions_mapping[args.model_type]
    else:
        args.augmentation_short="none"
    
    for target_index,target in enumerate(args.target_config):
        if "location" not in args.target_config[target].keys(): # Note: the case of zero-shot fitness predictions is already handled above if present
            if args.assay_data_folder is not None: # We passed at least one path for the assay location
                num_targets = [x for x in args.target_config.keys() if args.target_config[x]["in_NPT_loss"]]
                if len(args.assay_data_folder) > 1:
                    assert len(args.assay_data_folder)==num_targets, "Trying to predict {} targets, but only referencing {} distinct paths for them.".format(num_targets,len(args.assay_location))
                    args.target_config[target]["location"] = args.assay_data_folder[target_index]
                    print("Location used for target {} is: {}".format(target,args.assay_data_folder[target_index]))
                else:
                    args.target_config[target]["location"] = args.assay_data_folder[0]
                    print("Location used for target {} is: {}".format(target,args.assay_data_folder[0]))
            else:
                print("Assay location not provided. Defaulting to location for single substitutions fitness assays: {}".format(args.data_location + os.sep + 'data/fitness/substitutions_singles'))
                args.target_config[target]["location"] = args.data_location + os.sep + 'data/fitness/substitutions_singles'
    
    return args

def log_performance_fold(args,target_names,test_eval_results,trainer_final_status,perf_list,logs_folder=None):
    test_logs = {'total_training_steps': trainer_final_status['total_training_steps'], 'total_training_epochs': trainer_final_status['total_training_epochs'], 'total_train_time': trainer_final_status['total_train_time']}
    if logs_folder is None:
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_folder = dir_path+os.sep+'output'
        if not os.path.exists(logs_folder): os.mkdir(logs_folder)
    test_logs['Test total loss per seq.'] = test_eval_results['eval_total_loss']
    spearmans = {}
    num_obs_spearmans = {}
    for target_name in target_names:
        if args.target_config[target_name]["dim"]==1:
            spearmans[target_name] = pnpt_spearmanr(test_eval_results['output_scores']['predictions_'+target_name], test_eval_results['output_scores']['labels_'+target_name])
        else:
            # In the categorical setting, we predict the spearman between the logits of the category with highest index, and target value indices. This is meaningul in the binary setting. Use with care if 3 categories or more.
            spearmans[target_name] = pnpt_spearmanr(test_eval_results['output_scores']['predictions_'+target_name].apply(lambda x: x[-1]), test_eval_results['output_scores']['labels_'+target_name])
        num_obs_spearmans[target_name] = pnpt_count_non_nan(test_eval_results['output_scores']['labels_'+target_name])
        print("Spearman {} target: {}".format(target_name,spearmans[target_name]))
        test_logs['Test Spearman '+target_name] = spearmans[target_name]
        test_logs['Test loss '+str(target_name)+' per seq.'] = test_eval_results['eval_target_prediction_loss_dict'][target_name]
    with open(logs_folder+os.sep+"test_performance_by_fold_"+args.model_name_suffix+".csv", "a") as perf_tracker:
        if os.path.getsize(logs_folder+os.sep+"test_performance_by_fold_"+args.model_name_suffix+".csv") == 0: 
            header="fold_index,model_type,model_name_suffix,targets,assay_id,UniProt_id,fold_variable_name,total_training_steps,total_training_epochs,aa_embeddings,target_prediction_model,target_prediction_head,augmentation,frozen_embedding_parameters,dropout,weight_decay,early_stopping_patience,use_validation_set,training_num_assay_sequences_per_batch_per_gpu,eval_num_sequences_to_score_per_batch_per_gpu,eval_num_training_sequences_per_batch_per_gpu,eval_training_sequences_sampling_method,num_MSA_sequences_per_training_instance,embed_dim,ffn_embed_dim,attention_heads,conv_kernel_size,num_protein_npt_layers,total_loss"
            for target_name in target_names: header += (",loss_" + target_name + ",Spearman_" + target_name + ",num_obs_Spearman_" + target_name)
            perf_tracker.write(header+"\n")
        perf = ",".join([str(x) for x in perf_list]) + "," + str(round(test_logs['Test total loss per seq.'],5)) 
        for target_name in target_names: perf += ("," + str(round(test_logs['Test loss '+str(target_name)+' per seq.'],5)) +","+str(spearmans[target_name])+","+str(num_obs_spearmans[target_name]))
        perf_tracker.write(perf+"\n")
    return test_logs, spearmans

def log_performance_all_folds(args,target_names,all_test_predictions_across_folds,spearmans_across_folds,perf_list,logs_folder=None):
    if not os.path.exists(args.output_scores_location + os.sep + 'all_aggregated_predictions'): os.mkdir(args.output_scores_location + os.sep + 'all_aggregated_predictions')
    all_test_predictions_across_folds = pd.DataFrame.from_dict(all_test_predictions_across_folds)
    all_test_predictions_across_folds.to_csv(args.output_scores_location + os.sep + 'all_aggregated_predictions' + os.sep + model_name_prefix + ".csv", index=False)
    if logs_folder is None:
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_folder = dir_path+os.sep+'output'
        if not os.path.exists(logs_folder): os.mkdir(logs_folder)
    with open(logs_folder+os.sep+"test_performance_overall_"+perf_list[2]+".csv", "a") as overall_perf:
        if os.path.getsize(logs_folder+os.sep+"test_performance_overall_"+perf_list[2]+".csv") == 0: 
            header = "model_type,model_name_suffix,targets,assay_id,UniProt_id,fold_variable_name,total_training_steps,total_training_epochs,aa_embeddings,target_prediction_model,target_prediction_head,augmentation,frozen_embedding_parameters,dropout,weight_decay,early_stopping_patience,use_validation_set,training_num_assay_sequences_per_batch_per_gpu,eval_num_sequences_to_score_per_batch_per_gpu,eval_num_training_sequences_per_batch_per_gpu,eval_training_sequences_sampling_method,num_MSA_sequences_per_training_instance,embed_dim,ffn_embed_dim,attention_heads,conv_kernel_size,num_protein_npt_layers,total_loss"
            for target_name in target_names: header += (",loss_" + target_name + ",Spearman_" + target_name + ",Std_dev_Spearman_" + target_name + ",num_obs_Spearman_" + target_name + ",standardized_loss_" + target_name + ",standardized_Spearman_" + target_name)
            overall_perf.write(header+"\n")
        perf = ",".join([str(x) for x in perf_list[1:]]) #Remove fold_index from perf_list
        for target_name in target_names:
            missing_mask = np.isnan(all_test_predictions_across_folds['labels_'+target_name]) | np.equal(all_test_predictions_across_folds['labels_'+target_name],-100)
            if args.target_config[target_name]["type"]=="continuous":
                loss = ((all_test_predictions_across_folds['predictions_'+target_name][~missing_mask] - all_test_predictions_across_folds['labels_'+target_name][~missing_mask])**2).mean()
                loss_standardized = ((all_test_predictions_across_folds['fold_standardized_predictions_'+target_name][~missing_mask] - all_test_predictions_across_folds['labels_'+target_name][~missing_mask])**2).mean()
                spearman = pnpt_spearmanr(all_test_predictions_across_folds['predictions_'+target_name], all_test_predictions_across_folds['labels_'+target_name])
                spearman_standardized = pnpt_spearmanr(all_test_predictions_across_folds['fold_standardized_predictions_'+target_name], all_test_predictions_across_folds['labels_'+target_name])
            else:
                predictions_np = np.array([np.array(pred, dtype=np.float32) for pred in all_test_predictions_across_folds['predictions_'+target_name][~missing_mask]])
                loss = CrossEntropyLoss(reduction="mean")(torch.tensor(predictions_np).view(-1, args.target_config[target_name]["dim"]), torch.tensor(all_test_predictions_across_folds['labels_'+target_name][~missing_mask]).view(-1).long()).item()
                loss_standardized = None
                spearman = pnpt_spearmanr(all_test_predictions_across_folds['predictions_'+target_name].apply(lambda x: x[-1]), all_test_predictions_across_folds['labels_'+target_name])
                spearman_standardized = None
            num_obs_spearman = pnpt_count_non_nan(all_test_predictions_across_folds['labels_'+target_name])
            spearman_std_dev = np.array(spearmans_across_folds[target_name]).std()
            perf += ("," + str(loss) +","+str(spearman) + ","+ str(spearman_std_dev) + "," + str(num_obs_spearman) + "," + str(loss_standardized) +","+str(spearman_standardized))
        overall_perf.write(perf+"\n")

def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # target_names are the true targets we want to predict. target_names_input also includes auxiliary labels (as used in ProteinNPT)
    target_names = [x for x in args.target_config.keys() if args.target_config[x]["in_NPT_loss"]]
    target_names_input = args.target_config.keys()
    num_targets = len(target_names)
    num_targets_input = len(target_names_input)
    print("We want to predict {} target(s): {}".format(num_targets, ' and '.join(target_names)))
    if num_targets_input > num_targets: print("We leverage {} target(s) and auxiliary labels: {}".format(num_targets_input, ' and '.join(target_names_input)))

    if args.assay_reference_file_location is not None:
        assay_reference_file = pd.read_csv(args.assay_reference_file_location)
        assert "DMS_id" in assay_reference_file.columns, "Reference file must include a DMS_id"
        assay_id=assay_reference_file["DMS_id"][args.assay_index]
        assay_file_name = assay_reference_file["DMS_filename"][assay_reference_file["DMS_id"]==assay_id].values[0] # File name of main assay used during training (if single property, this is also the only assay). Retrieved embeddings are always for this assay.
        target_seq = assay_reference_file["target_seq"][assay_reference_file["DMS_id"]==assay_id].values[0]
        args.seq_len = int(assay_reference_file["seq_len"][assay_reference_file["DMS_id"]==assay_id].values[0]) if "seq_len" in assay_reference_file.columns else len(target_seq)
        args.MSA_seq_len = int(assay_reference_file["MSA_len"][assay_reference_file["DMS_id"]==assay_id].values[0]) if "MSA_len" in assay_reference_file.columns else len(target_seq)
    else:
        assay_id = args.assay_data_location.split(".csv")[0].split(os.sep)[-1]
        assay_file_name = args.assay_data_location.split(os.sep)[-1]
        args.seq_len = len(args.target_seq)
        args.MSA_seq_len = args.MSA_end - args.MSA_start + 1
    print("Training model for assay: {}, where the test_fold index is: {}".format(assay_id, args.test_fold_index))
    args.save_model_checkpoint = not args.do_not_save_model_checkpoint
    args.frozen_embedding_parameters = not args.fine_tune_model_embedding_parameters
    if args.model_type=="MSA_Transformer_pred": assert args.num_MSA_sequences_per_training_instance==args.num_MSA_sequences_per_eval_instance, "MSA_Transformer_pred only supports same size of MSA for train and eval"
    
    effective_batch_size = args.gradient_accumulation * args.training_num_assay_sequences_per_batch_per_gpu
    print("Effective batch size is {}".format(effective_batch_size))

    model_hypers = [args.aa_embeddings,args.target_prediction_model,args.target_prediction_head,args.augmentation,args.frozen_embedding_parameters,args.dropout,args.weight_decay, \
                    args.early_stopping_patience, args.use_validation_set, args.training_num_assay_sequences_per_batch_per_gpu, args.eval_num_sequences_to_score_per_batch_per_gpu, args.eval_num_training_sequences_per_batch_per_gpu, \
                    args.eval_training_sequences_sampling_method, args.num_MSA_sequences_per_training_instance, args.embed_dim, args.ffn_embed_dim, args.attention_heads, args.conv_kernel_size, args.num_protein_npt_layers]
    model_hypers_str = ','.join([str(x) for x in model_hypers])
    model_name_prefix = '_'.join([str(x) for x in [args.model_type,assay_id,"_".join(target_names_input),args.fold_variable_name,'embed_'+args.aa_embeddings,'head_'+str(args.target_prediction_model),'aug_'+str(args.augmentation_short), \
                                    'froz_'+str(args.frozen_embedding_parameters),'drop_'+str(args.dropout),'val_'+str(args.use_validation_set),args.model_name_suffix]])
    model_name = model_name_prefix + "_fold-" + str(args.test_fold_index)
    if not os.path.exists(args.model_location+os.sep+model_name): os.mkdir(args.model_location+os.sep+model_name)
    with open(args.model_location+os.sep+model_name+os.sep+'training_arguments', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print("Model name: "+model_name)
    args.sequence_embeddings_location = args.sequence_embeddings_folder + os.sep + assay_file_name.split(".csv")[0] + '.h5' if args.sequence_embeddings_folder else None
    print("Sequence embeddings: {}".format(args.sequence_embeddings_location))
    
    if args.use_wandb: wandb.login()   
    
    # Create & initiate model   
    if args.aa_embeddings=="Tranception":
        alphabet = get_tranception_tokenizer() if args.aa_embeddings=="Tranception" else Alphabet.from_architecture("msa_transformer")
    elif args.aa_embeddings=="MSA_Transformer":
        alphabet = Alphabet.from_architecture("msa_transformer")
    else:
        alphabet = Alphabet.from_architecture("ESM-1b")
    
    if args.model_type=="ProteinNPT":
        model = ProteinNPTModel(args, alphabet)
    elif args.model_type in ["MSA_Transformer_pred", "ESM1v_pred", "Tranception_pred", "TranceptEVE_pred", "Linear_Embedding_pred", "DeepSequence_pred"] or args.model_type.startswith("ESM2"):
        model = AugmentedPropertyPredictor(args, alphabet)
    if args.frozen_embedding_parameters and args.aa_embeddings in ["MSA_Transformer", "ESM1v", "Tranception"]:
        for para in model.aa_embedding.parameters():
            para.requires_grad = False

    # List of assays involved in training
    if num_targets==1:
        # Single property prediction
        assay_file_names={
            target_names[0]: assay_file_name
        }
        if "zero_shot_fitness_predictions" in target_names_input: assay_file_names["zero_shot_fitness_predictions"] = assay_file_name
    else:
        # Multiple properties prediction
        assay_file_names={}
        for target in target_names_input:
            if target=="zero_shot_fitness_predictions":
                assay_file_names[target] = assay_file_name # The name of the zero-shot prediction file matches that of the main assay
            elif args.assay_reference_file_location is None: #Not using reference file
                print("If not using a reference file and predicting several targets simultaneously, we assume the different targets are all present in the same assay file")
                assay_file_names[target] = assay_file_name
            else:
                assay_file_names[target] = assay_reference_file[assay_reference_file["DMS_id"]==assay_id]['DMS_filename'].values[0]
            
    # Load training, val and test data
    if args.assay_reference_file_location is not None:
        UniProt_id = assay_reference_file["UniProt_ID"][assay_reference_file["DMS_id"]==assay_id].values[0] if "UniProt_ID" in assay_reference_file.columns else assay_id
        MSA_filename = assay_reference_file["MSA_filename"][assay_reference_file["DMS_id"]==assay_id].values[0] if "MSA_filename" in assay_reference_file.columns else None
        MSA_weights_filename = assay_reference_file["weight_file_name"][assay_reference_file["DMS_id"]==assay_id].values[0] if "weight_file_name" in assay_reference_file.columns else None
        MSA_start_position = int(assay_reference_file["MSA_start"][assay_reference_file["DMS_id"]==assay_id].values[0]) if "MSA_start" in assay_reference_file.columns else 1
        MSA_end_position = int(assay_reference_file["MSA_end"][assay_reference_file["DMS_id"]==assay_id].values[0]) if "MSA_end" in assay_reference_file.columns else args.seq_len
    else:
        UniProt_id = assay_id
        MSA_filename = args.MSA_location.split(os.sep)[-1] if args.MSA_location is not None else None
        MSA_weights_filename = args.MSA_sequence_weights_filename
        MSA_start_position = args.MSA_start
        MSA_end_position = args.MSA_end
    train_data, val_data, test_data, target_processing = get_train_val_test_data(args = args, assay_file_names = assay_file_names)
    if args.aa_embeddings == "MSA_Transformer":
        MSA_sequences, MSA_weights = process_MSA(
            MSA_data_folder=args.MSA_data_folder,
            MSA_weight_data_folder=args.MSA_weight_data_folder,
            MSA_filename=MSA_filename,
            MSA_weights_filename=MSA_weights_filename,
            path_to_hhfilter=args.path_to_hhfilter
        )
    else:
        MSA_sequences = None
        MSA_weights = None
    
    if args.use_wandb:
        combined_dict = {**vars(args), "parameter_count": sum(p.numel() for p in model.parameters()), "assay_id": assay_id, "UniProt_id": UniProt_id}
        wandb.init(project=os.getenv("WANDB_PROJECT"), config=combined_dict, name=model_name, dir=args.wandb_location, save_code=True)
    
    print("Starting training")
    # Define trainer
    trainer = Trainer(
            model= model,
            args=args,
            train_data=train_data, 
            val_data=val_data,
            MSA_sequences=MSA_sequences, 
            MSA_weights=MSA_weights,
            MSA_start_position=MSA_start_position,
            MSA_end_position=MSA_end_position,
            target_processing=target_processing
    )
    # Load model from checkpoint or train from scratch
    if args.load_model_checkpoint:
        checkpoint_location = args.model_location +os.sep + model_name + os.sep + 'final' + os.sep + 'checkpoint.t7'
        checkpoint = torch.load(checkpoint_location)
        # load the state dictionary into your model
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        trainer_final_status = {
            'total_training_steps': -1,
            'total_training_epochs': -1,
            'total_train_time': -1
        }
        model.cuda()
        model.set_device()
    else:
        trainer_final_status = trainer.train()
    print('Final training step: {} | Num training epochs: {} | Total train time: {} hrs'.format(trainer_final_status['total_training_steps'], trainer_final_status['total_training_epochs'], str(trainer_final_status['total_train_time'] / 3600)))

    # Eval performance on test set & log to wandb & persist predictions / performance to disk
    if args.model_type == "ProteinNPT":
        test_eval_results = trainer.eval(
            test_data=test_data,
            train_data=train_data,
            reconstruction_loss_weight=0.0,
            output_all_predictions=True
        )
    else:
        test_eval_results = trainer.eval(
            test_data=test_data, 
            output_all_predictions=True
        )
    
    # Log performance (by default, this will create an "output" directory in the parent dir where logs will be stored. Adjust logs_folder as needed)
    perf_list = [args.test_fold_index,args.model_type,args.model_name_suffix,"_".join(target_names_input),assay_id,UniProt_id,args.fold_variable_name,trainer_final_status['total_training_steps'],trainer_final_status['total_training_epochs'],model_hypers_str]
    test_logs, spearmans = log_performance_fold(args,target_names,test_eval_results,trainer_final_status,perf_list,logs_folder=None)
    perf_list.append(test_logs['Test total loss per seq.'])
    test_eval_results['output_scores'].to_csv(args.output_scores_location + os.sep + model_name + '.csv', index=False) # Store fold predictions
    if args.use_wandb: 
        wandb.log(test_logs)
        wandb.finish()

    # Save final model
    if args.save_model_checkpoint:
        if not os.path.exists(args.model_location +os.sep + model_name + os.sep + 'final'): os.mkdir(args.model_location +os.sep + model_name + os.sep + 'final')
        if hasattr(model, 'aa_embedding') and args.frozen_embedding_parameters: del model.aa_embedding #If embeddings are not fine tuned, we do not need to save them to save space
        torch.save({
            'args': args,
            'final_training_step': trainer_final_status['total_training_steps'],
            'state_dict': model.state_dict(),
            }, 
            args.model_location +os.sep + model_name + os.sep + 'final' + os.sep + 'checkpoint.t7'
        )

    # Freeing up GPU memory after training on a given fold split:
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return test_eval_results['output_scores'], perf_list, model_name_prefix, spearmans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ProteinNPT or baseline model')
    parser.add_argument('--data_location', default=None, type=str, help='Path to core ProteinNPT datasets (e.g., MSA files, DMS assays, pretrained model checkpoints). Training output will also be stored there (i.e., checkpoints and test set predictions).')
    parser.add_argument('--model_config_location', default=None, type=str, help='Path to main model config file that specifies all parameters as needed')
    #Data parameters
    parser.add_argument('--assay_reference_file_location', default=None, type=str, help='Path to reference file with list of assays to score')
    parser.add_argument('--assay_index', default=None, type=int, help='Index of main assay to train on/predict for in the ProteinGym reference file')
    parser.add_argument('--assay_data_folder', default=None, type=str, nargs='*', help='Location of assay file(s) (including CV splits variables). If predicting multiple assays in different directories, pass as many location as needed')
    parser.add_argument('--target_config_location', default=None, type=str, help='Config file for assays to be used for modelings')
    parser.add_argument('--augmentation', default=None, type=str, help='Type of augmentation used ["None","zero_shot_fitness_predictions_covariate" or "zero_shot_fitness_predictions_auxiliary_labels"]. Note that default value is set in each model config files')
    parser.add_argument('--zero_shot_fitness_predictions_location', default=None, type=str, help='Path to zero-shot fitness predictions used as additional covariates (baselines) or auxiliary labels (ProteinNPT)')
    parser.add_argument('--wandb_location', default="wandb", type=str, help='Wandb directory where metadata is stored')
    parser.add_argument('--fold_variable_name', default=None, type=str, help='Name of the fold variable in the processed assay files')
    parser.add_argument('--test_fold_index', default=-1, type=int, help='Index of fold to test performance on [If "-1" is provided, we will train on all seed splits sequentially]')
    parser.add_argument('--use_validation_set', action='store_true', help='Whether to use a validation set during training [If yes, we will stop training based on CV loss and patience param. Train until the end otherwise]')
    parser.add_argument('--num_data_loaders_workers', default=0, type=int, help='Number of workers to use to fetch and load data in memory')
    parser.add_argument('--MSA_data_folder', default=None, type=str, help='Folder all MSAs are stored for reference sequence of ProteinGym assays')
    parser.add_argument('--MSA_weight_data_folder', default=None, type=str, help='Folder where MSA sequence weights are stored (for diversity sampling of MSA)')
    parser.add_argument('--path_to_hhfilter', default=None, type=str, help='Path to hhfilter (for filtering MSA)')
    #Model parameters
    parser.add_argument('--model_type', default=None, type=str, help='Model type')
    parser.add_argument('--model_name_suffix', default=None, type=str, help='Suffix to reference model')
    parser.add_argument('--sequence_embeddings_folder', required=True, type=str, help='Location of stored embeddings on disk')
    parser.add_argument('--embedding_model_location', default=None, type=str, help='Location of model used to embed protein sequences')
    parser.add_argument('--aa_embeddings', default=None, type=str, help='Type of protein sequence embedding [MSA_Transformer|Tranception|ESM1v|ESM2|Linear_embedding]')
    parser.add_argument('--long_sequences_slicing_method', default='center', type=str, help='Method to slice long sequences [rolling, center, left]. We do not slice OHE input')
    parser.add_argument('--max_positions', default=None, type=int, help='Maximum context length')
    parser.add_argument('--embed_dim', default=None, type=int, help='Embedding dimension')
    parser.add_argument('--ffn_embed_dim', default=None, type=int, help='Feedforward embedding dimension')
    parser.add_argument('--attention_heads', default=None, type=int, help='Number of attention heads')
    parser.add_argument('--conv_kernel_size', default=None, type=int, help='Size of convolutional kernel')
    parser.add_argument('--weight_decay', default=None, type=float, help='Weight decay to apply to network weights during training')
    parser.add_argument('--dropout', default=None, type=float, help='Dropout')
    parser.add_argument('--attention_dropout', default=None, type=float, help='Attention dropout')
    parser.add_argument('--activation_dropout', default=None, type=float, help='Activation dropout')
    parser.add_argument('--num_protein_npt_layers', default=None, type=int, help='Number of ProteinNPT layers [ProteinNPT only]')
    parser.add_argument('--target_prediction_head', default=None, type=str, help='Target prediction head type [AA_embeddings_mean_pooled, One_hot_encoding]')
    parser.add_argument('--target_prediction_model', default=None, type=str, help='Target prediction head model type [linear | MLP | CNN]')
    #Training & Eval parameters
    parser.add_argument('--num_total_training_steps', default=None, type=int, help='Number of total training steps')
    parser.add_argument('--num_logging_training_steps', default=None, type=int, help='Number of steps between 2 consecutive training loss logging')
    parser.add_argument('--do_not_save_model_checkpoint', action='store_true', help='Whether to save model checkpoint')
    parser.add_argument('--load_model_checkpoint', action='store_true', help='Whether to load model checkpoint')
    parser.add_argument('--num_saving_training_steps', default=None, type=int, help='Number of steps between 2 consecutive model checkpoint saving')
    parser.add_argument('--num_eval_steps', default=None, type=int, help='Number of steps between 2 consecutive evaluations on validation set')
    parser.add_argument('--num_warmup_steps', default=None, type=int, help='Number of training steps for lr warmup')
    parser.add_argument('--gradient_accumulation', default=None, type=int, help='Number of gradient accumulation steps (ie., number of forward & bwd passes per gradient optim. step)')
    parser.add_argument('--training_num_assay_sequences_per_batch_per_gpu', default=None, type=int, help='Number of assay sequences (with labels) to be leveraged during training per device')
    parser.add_argument('--eval_num_sequences_to_score_per_batch_per_gpu', default=None, type=int, help='Number of sequences to score (no label) at inference time')
    parser.add_argument('--eval_num_training_sequences_per_batch_per_gpu', default=None, type=int, help='Number of sequences from training (with label) at inference time [ProteinNPT only]')
    parser.add_argument('--eval_training_sequences_sampling_method', default=None, type=str, help='How to sample training points (with label) at inference time [ProteinNPT only]')
    parser.add_argument('--indel_mode', action='store_true', help='indel mode')
    parser.add_argument('--seed', default=None, type=int, help='Random seed used during training')
    parser.add_argument('--num_MSA_sequences_per_training_instance', default=None, type=int, help='Number of MSA sequences to be leveraged during training')
    parser.add_argument('--num_MSA_sequences_per_eval_instance', default=None, type=int, help='Number of MSA sequences to be leveraged at evaluation time')
    parser.add_argument('--max_tokens_per_msa', default=2**14, type=int, help='Used during inference to batch attention computations in a single forward pass. This allows increased input sizes with less memory.')
    parser.add_argument('--early_stopping_patience', default=5, type=int, help='Number of consecutive evals for which the loss has to not go below the min value to call early stopping (if None, no early stopping)')
    parser.add_argument('--max_learning_rate', default=3e-4, type=float, help='Max learning rate after warmup')
    parser.add_argument('--min_learning_rate', default=1e-5, type=float, help='Min learning rate post warmup and cosine decline')
    parser.add_argument('--adam_beta1', default=0.9, type=float, help='Beta1 value in AdamW optimizer')
    parser.add_argument('--adam_beta2', default=0.999, type=float, help='Beta1 value in AdamW optimizer')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Term added to the denominator to improve numerical stability in AdamW')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='Label smoothing parameter in the MLM loss')
    parser.add_argument('--grad_norm_clip', default=1.0, type=float, help='Maximum gradient value above which we do gradient clipping')
    parser.add_argument('--fine_tune_model_embedding_parameters', action='store_true', help='Whether to fine tune the model providing protein sequence embeddings')
    parser.add_argument('--training_fp16', action='store_true', help='Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training.')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to log runs in wandb')
    #No reference file
    parser.add_argument('--assay_data_location', default=None, type=str, help='Path to assay data csv (expects a csv, with at least three columns: mutant or mutated_sequence | DMS_score | fold_variable_name)')
    parser.add_argument('--MSA_location', default=None, type=str, help='Path to MSA file (expects .a2m)')
    parser.add_argument('--MSA_sequence_weights_filename', default=None, type=str, help='Sequence weights in MSA')
    parser.add_argument('--target_seq', default=None, type=str, help='WT sequence mutated in the assay')
    parser.add_argument('--MSA_start', default=None, type=int, help='Index of first AA covered by the MSA relative to target_seq coordinates (1-indexing)')
    parser.add_argument('--MSA_end', default=None, type=int, help='Index of last AA covered by the MSA relative to target_seq coordinates (1-indexing)')
    args = parser.parse_args()
    
    setup_config_and_paths(args)
    print("Embeddings folder:", args.embedding_model_location)

    if (args.MSA_start is None) or (args.MSA_end is None):
        if args.MSA_location is not None: print("MSA start and end not provided -- Assuming the MSA is covering the full WT sequence")
        args.MSA_start = 1
        if args.target_seq: args.MSA_end = len(args.target_seq)
        
    if args.test_fold_index==-1:
        target_names = [x for x in args.target_config.keys() if args.target_config[x]["in_NPT_loss"]]
        if args.assay_reference_file_location is not None:
            assay_reference_file = pd.read_csv(args.assay_reference_file_location)
            assay_id=assay_reference_file["DMS_id"][args.assay_index]
            assay_file_name = assay_reference_file["DMS_filename"][assay_reference_file["DMS_id"]==assay_id].values[0]
            for target in target_names:
                if args.target_config[target]["main_target"]: main_target_name=target
            assay_df = pd.read_csv(args.target_config[main_target_name]["location"] + os.sep + assay_file_name)
        else:
            assert args.assay_data_location is not None, "Reference file nor assay data file not provided"
            assay_df = pd.read_csv(args.assay_data_location)
        all_folds = [int(x) for x in sorted(list(assay_df[args.fold_variable_name].unique()))]
        num_folds = len(all_folds)
        print("There are {} unique fold values in the assay data for the selected fold_variable_name: {}".format(num_folds,all_folds))
        
        all_test_predictions_by_fold = {}
        all_test_predictions_across_folds = defaultdict(list)
        spearmans_across_folds = defaultdict(list)
        for fold_index in all_folds:
            args.test_fold_index=fold_index
            all_test_predictions_by_fold[fold_index], perf_list, model_name_prefix, spearmans = main(args)
            all_test_predictions_across_folds['mutated_sequence'] += list(all_test_predictions_by_fold[fold_index]['mutated_sequence'])
            for target_name in target_names:
                if args.target_config[target_name]["dim"]==1: all_test_predictions_across_folds['fold_standardized_predictions_'+target_name] += list(standardize(all_test_predictions_by_fold[fold_index]['predictions_'+target_name]))
                all_test_predictions_across_folds['predictions_'+target_name] += list(all_test_predictions_by_fold[fold_index]['predictions_'+target_name])
                all_test_predictions_across_folds['labels_'+target_name] += list(all_test_predictions_by_fold[fold_index]['labels_'+target_name])
                spearmans_across_folds[target_name].append(spearmans[target_name])
        log_performance_all_folds(args,target_names,all_test_predictions_across_folds,spearmans_across_folds,perf_list)
    else:
        main(args)