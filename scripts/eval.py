import os
import argparse
import torch

from proteinnpt.proteinnpt.model import ProteinNPTModel
from proteinnpt.baselines.model import AugmentedPropertyPredictor
from proteinnpt.utils.esm.data import Alphabet
from proteinnpt.utils.tranception.model_pytorch import get_tranception_tokenizer
from proteinnpt.utils.data_utils import get_train_val_test_data, pnpt_spearmanr
from proteinnpt.utils.model_utils import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eval ProteinNPT or baseline model')
    parser.add_argument('--model_location', required=True, type=str, help='Path to core ProteinNPT datasets (e.g., MSA files, DMS assays, pretrained model checkpoints). Training output will also be stored there (i.e., checkpoints and test set predictions).')
    parser.add_argument('--assay_data_location', required=True, type=str, help='Path to data with mutated sequences to score (.csv file) [If multiple targets prediction, we assume all targets are present in this file as well]')
    parser.add_argument('--embeddings_location', required=True, type=str, help='Path to embeddings for train and test sequences [to be obtained for assay data via the embeddings.sh script]')
    parser.add_argument('--zero_shot_fitness_predictions_location', default=None, type=str, help='Path to zero-shot fitness predictions for train and test sequences [to be obtained for assay data via the zero_shot_fitness_subs.sh or zero_shot_fitness_indels.sh script]')
    parser.add_argument('--fold_variable_name', default=None, type=str, help='Name of the fold variable in the processed assay files')
    parser.add_argument('--test_fold_index', default=1, type=int, help='Index of test fold [By default, we assume all train sequences are in fold 0, all test sequences are in fold 1]')
    parser.add_argument('--output_scores_location', type=str, help='Path to file in which to file model predictions for test sequences')
    args = parser.parse_args()

    # Load model checkpoint
    main_checkpoint = torch.load(args.model_location)
    model_args = main_checkpoint['args']
    alphabet = get_tranception_tokenizer() if model_args.aa_embeddings=="Tranception" else Alphabet.from_architecture("msa_transformer")
    if model_args.model_type=="ProteinNPT":
        model = ProteinNPTModel(model_args, alphabet)
    elif model_args.model_type in ["MSA_Transformer_pred", "ESM1v_pred", "Tranception_pred", "TranceptEVE_pred", "Linear_Embedding_pred", "DeepSequence_pred"]:
        model = AugmentedPropertyPredictor(model_args, alphabet)
    model.load_state_dict(main_checkpoint['state_dict'], strict=False) #Set strict to False as we typically do not save aa_embedding in model checkpoint for disk space considerations (and not needed at inference as we rely on embeddings_location)
    if torch.cuda.is_available(): model.cuda()
    model.set_device()
    
    # Files setup
    model_args.fold_variable_name = args.fold_variable_name
    model_args.test_fold_index = args.test_fold_index
    model_args.embeddings_location = args.embeddings_location
    model_args.assay_data_folder = [ os.sep.join(args.assay_data_location.split(os.sep)[:-1]) ]
    assay_file_name = args.assay_data_location.split(os.sep)[-1]
    assert (model_args.augmentation_short=="none" or args.zero_shot_fitness_predictions_location is not None), "Zero-shot fitness predictions were not provided but were used at train time"
    zero_shot_fitness_predictions_folder = os.sep.join(args.zero_shot_fitness_predictions_location.split(os.sep)[:-1])
    zero_shot_fitness_predictions_filename = args.zero_shot_fitness_predictions_location.split(os.sep)[-1]
    assay_file_names={}
    print("Assay filename: {}".format(assay_file_name))
    for target_index,target_name in enumerate(model_args.target_config):
        if target_name!="zero_shot_fitness_predictions": 
            model_args.target_config[target_name]["location"] = model_args.assay_data_folder[0]
            assay_file_names[target_name] = assay_file_name
        elif model_args.augmentation=="zero_shot_fitness_predictions_auxiliary_labels":
            model_args.target_config["zero_shot_fitness_predictions"]["location"] = zero_shot_fitness_predictions_folder
            assay_file_names[target_name] = zero_shot_fitness_predictions_filename
    train_data, val_data, test_data, target_processing = get_train_val_test_data(args = model_args, assay_file_names = assay_file_names)

    # Inference
    trainer = Trainer(
            model=model,
            args=model_args,
            train_data=train_data,
            target_processing=target_processing
    )
    if model_args.model_type == "ProteinNPT":
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
    test_eval_results['output_scores'].to_csv(args.output_scores_location, index=False)
    
    for target_name in ["fitness"]:
        spearman_pho = pnpt_spearmanr(test_eval_results['output_scores']['predictions_'+target_name], test_eval_results['output_scores']['labels_'+target_name])
        print(spearman_pho)