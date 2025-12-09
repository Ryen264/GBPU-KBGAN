import torch
import os
import argparse
import sys
from config import config, overwrite_config_with_args, logger_init
from data_loader import index_entity_relation, graph_size, read_data
from datasets import sparse_heads_tails, inplace_shuffle
from kbgan import KBGAN

def main():
    _config = config()
    mode = sys.argv[1].split('=')[1] if len(sys.argv) > 1 else None
    args = sys.argv[2:]
    if args:
        overwrite_config_with_args(args)
        print("Running config:", _config)

    _config['KBGAN']['n_epoch'] = 2
    _config[_config.d_config]['n_epoch'] = 2
    _config[_config.g_config]['n_epoch'] = 2

    # Init logging now that config is prepared
    logger_init()

    # Load data
    task_dir = '.\\data\\' + _config.dataset
    kb_index = index_entity_relation(
        os.path.join(task_dir, 'train.txt'),
        os.path.join(task_dir, 'valid.txt'),
        os.path.join(task_dir, 'test.txt')
    )
    n_entity, n_relation = graph_size(kb_index)

    train_data = read_data(os.path.join(task_dir, 'train.txt'), kb_index)
    inplace_shuffle(*train_data)
    valid_data = read_data(os.path.join(task_dir, 'valid.txt'), kb_index)
    test_data = read_data(os.path.join(task_dir, 'test.txt'), kb_index)
    heads, tails = sparse_heads_tails(n_entity, train_data, valid_data, test_data)

    # For task triple-classification, we need to read data with labels
    if _config.task == 'triple-classification' or _config.task == 'all':
        valid_data_with_label   = read_data(os.path.join('.\\data\\' + _config.dataset + '_w_labels', 'valid.txt'), kb_index, with_label=True)
        test_data_with_label    = read_data(os.path.join('.\\data\\' + _config.dataset + '_w_labels', 'test.txt'), kb_index, with_label=True)

    # Convert to tensors
    train_data  = [torch.LongTensor(vec) for vec in train_data]
    valid_data  = [torch.LongTensor(vec) for vec in valid_data]
    test_data   = [torch.LongTensor(vec) for vec in test_data]

    print(f"Running mode: {mode}")
    model = KBGAN(discriminator_type="TransE", generator_type="DistMult")
    model.fit(n_entity, n_relation)
    if mode == 'full-train':
        # Train 2 components
        dis_best_perf, dis_model_path, gen_best_perf, gen_model_path = model.pretrain(heads, tails, train_data, valid_data_with_label,
                                                                                        use_early_stopping=_config['KBGAN']['early_stopping_pretrain'], patience=_config['KBGAN']['patience'], optimizer_name=_config['KBGAN']['optimizer_name'])
        # Test 2 components just be trained on link prediction
        dis_metrics = model.evaluate_component("discriminator", heads, tails, test_data)
        print(f"Discriminator metrics on Link Prediction:\n{dis_metrics}")

        gen_metrics = model.evaluate_component("generator", heads, tails, test_data)
        print(f"Generator metrics on Link Prediction:\n{gen_metrics}")
#
        # Train KBGAN
        best_perf, model_path = model.train(heads, tails, train_data, valid_data,
                                            use_early_stopping=_config['KBGAN']['early_stopping_train'], patience=_config['KBGAN']['patience'], optimizer_name=_config['KBGAN']['optimizer_name'])
        
        print(f"KBGAN model saved to: {model_path}")
        print(f"Best validation performance on link prediction while training: {best_perf}")

        
        # Test KBGAN on link prediction
        link_prediction_metrics = model.evaluate_on_link_prediction(heads, tails, test_data)
        print(f"Link prediction metrics:\n{link_prediction_metrics}")

        # Test KBGAN on triple classification
        triple_classification_metrics, threshold, predictions, scores_list = model.evaluate_on_triple_classification(test_data_with_label, valid_data_with_label, threshold=None, auto_threshold=True)
        print(f"Triple classification metrics:\n{triple_classification_metrics}")

    elif mode == 'gan-train':
        # Load 2 pretrained components
        dis_model_path = '.\\models\\' + _config.dataset + '\\' + _config.task + '\\components\\' + _config['d_config'] + '.mdl'
        model.load_component(component_role="discriminator", component_path=dis_model_path)
        gen_model_path = '.\\models\\' + _config.dataset + '\\' + _config.task + '\\components\\' + _config['g_config'] + '.mdl'
        model.load_component(component_role="generator", component_path=gen_model_path)

        # Train KBGAN
        best_perf, model_path = model.train(heads, tails, train_data, valid_data,
                                             use_early_stopping=_config['KBGAN']['early_stopping_train'], patience=_config['KBGAN']['patience'], optimizer_name=_config['KBGAN']['optimizer_name'])
        print(f"KBGAN model saved to: {model_path}")
        print(f"Best validation performance on link prediction while training: {best_perf}")

        # Test KBGAN on Link Prediction
        link_prediction_metrics = model.evaluate_on_link_prediction(heads, tails, test_data)
        print(f"Link prediction metrics:\n{link_prediction_metrics}")

        # Test KBGAN on Triple Classification
        triple_classification_metrics, threshold, predictions, scores_list = model.evaluate_on_triple_classification(test_data_with_label, valid_data_with_label, threshold=None, auto_threshold=True)
        print(f"Triple classification metrics:\n{triple_classification_metrics}")   
        
    elif mode == 'test-only':
        # Load pretrained KBGAN
        model.load("./output/" + _config.task.dir + "/models/" + _config['d_config'] + ".mdl")

        # Test KBGAN on link prediction
        link_prediction_metrics = model.evaluate_on_link_prediction(heads, tails, test_data)
        print(f"Link prediction metrics:\n{link_prediction_metrics}")

        # Test KBGAN on Triple Classification
        triple_classification_metrics, threshold, predictions, scores_list = model.evaluate_on_triple_classification(test_data_with_label, valid_data_with_label, threshold=None, auto_threshold=True)
        print(f"Triple classification metrics:\n{triple_classification_metrics}") 
    else: 
        print("Invalid mode. Please try again and specify a mode: full-train / gan-train / test-only") 


if __name__ == '__main__':
    main()
