import torch
import os
import argparse

from config import config, overwrite_config_with_args, logger_init
from data_loader import index_entity_relation, graph_size, read_data
from datasets import sparse_heads_tails, inplace_shuffle
from kbgan import KBGAN

def parse_args():
    p = argparse.ArgumentParser(description='Run KBGAN pipeline')
    p.add_argument('--mode', choices=['full', 'pretrain', 'train', 'evaluate_on_link_prediction', 'evaluate_on_triple_classification'], default='full',
                   help='Pipeline mode to run')
    p.add_argument('--config-file',
                   help='Path to YAML config file (overrides default)')
    p.add_argument('--override', action='append', default=[],
                   help='Config overrides like --override TransE.n_epoch=500 (can be repeated)')
    p.add_argument('--early-stopping-pretrain', action='store_true',
                   help='Enable early stopping for pretrain')
    p.add_argument('--early-stopping-train', action='store_true',
                   help='Enable early stopping for train')
    p.add_argument('--patience', type=int, default=10,
                   help='Number of validation checks to wait for improvement before stopping (default: 10)')
    p.add_argument('--optimizer-name', choices=['Adam', 'SGD', 'AdamW', 'RMSprop', 'Adagrad'], default='Adam',
                   help='Optimizer used for pretrain')
    return p.parse_args()

def main(argv=None, mode: str=None,
         early_stopping_pretrain: bool=None, early_stopping_train: bool=None, patience: int=None, optimizer_name: str=None):
    args = parse_args() if argv is None else parse_args()

    # Load configuration (default or provided file)
    if args.config_file:
        _config = config(args.config_file)
        print(f"Loaded config from {args.config_file}")
    else:
        _config = config()

    # Apply default hyperparameter overrides first
    default_overrides = [
        "--TransE.n_epoch=2",
        "--TransE.epoch_per_test=5",
        "--DistMult.n_epoch=2",
        "--DistMult.epoch_per_test=5",
        "--KBGAN.n_epoch=2",
        "--KBGAN.epoch_per_test=5"
    ]
    overwrite_config_with_args(default_overrides)

    # Apply user overrides (after defaults so they take precedence)
    user_overrides = []
    for o in args.override:
        if o.startswith('--'):
            user_overrides.append(o)
        else:
            user_overrides.append('--' + o)
    if user_overrides:
        overwrite_config_with_args(user_overrides)

    _config.log.to_file = True

    # Init logging now that config is prepared
    logger_init()

    # Load data
    task_dir = '.\\data\\' + _config.task.dir
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

    # For WN18RR, we need to read data with labels for triple classification
    if _config.task.dir == 'wn18rr':
        valid_data_with_label   = read_data(os.path.join('.\\data\\evaluation on TP\\wn18rr', 'valid.txt'), kb_index, with_label=True)
        test_data_with_label    = read_data(os.path.join('.\\data\\evaluation on TP\\wn18rr', 'test.txt'), kb_index, with_label=True)

    # Convert to tensors
    train_data  = [torch.LongTensor(vec) for vec in train_data]
    valid_data  = [torch.LongTensor(vec) for vec in valid_data]
    test_data   = [torch.LongTensor(vec) for vec in test_data]

    model = KBGAN(discriminator_type="TransE", generator_type="DistMult")
    model.fit(n_entity, n_relation)

    print(f"Running mode: {args.mode}")
    if mode is not None and mode in ('full', 'pretrain', 'train', 'evaluate_on_link_prediction', 'evaluate_on_triple_classification'):
        args.mode = mode
    if early_stopping_pretrain is not None:
        args.early_stopping_pretrain = early_stopping_pretrain
    if early_stopping_train is not None:
        args.early_stopping_train = early_stopping_train
    if patience is not None:
        args.patience = patience
    if optimizer_name is not None:
        args.optimizer_name = optimizer_name

    if args.mode == 'full':
        # Pretrain 2 components
        dis_best_perf, dis_model_path, gen_best_perf, gen_model_path = model.pretrain(heads, tails, train_data, valid_data_with_label,
                                                                                        use_early_stopping=args.early_stopping_pretrain, patience=args.patience, optimizer_name=args.optimizer_name)
        
        # Test 2 pretrained components
        dis_metrics = model.evaluate_component("discriminator", heads, tails, test_data)
        print(f"Discriminator metrics:\n{dis_metrics}")

        gen_metrics = model.evaluate_component("generator", heads, tails, test_data)
        print(f"Generator metrics:\n{gen_metrics}")
#
        # Train KBGAN
        best_perf, model_path = model.train(heads, tails, train_data, valid_data,
                                            use_early_stopping=args.early_stopping_train, patience=args.patience, optimizer_name=args.optimizer_name)
        
        # Test KBGAN on link prediction
        link_prediction_metrics = model.evaluate_on_link_prediction(heads, tails, test_data)
        print(f"Link prediction metrics:\n{link_prediction_metrics}")

        # Test KBGAN on triple classification
        triple_classification_metrics, threshold, predictions, scores_list = model.evaluate_on_triple_classification(test_data_with_label, valid_data_with_label, threshold=None, auto_threshold=True)
        print(f"Triple classification metrics:\n{triple_classification_metrics}")

    elif args.mode =='pretrain':
        dis_best_perf, dis_model_path, gen_best_perf, gen_model_path = model.pretrain(heads, tails, train_data, valid_data_with_label,
                                                                                        use_early_stopping=args.early_stopping_pretrain, patience=args.patience, optimizer_name=args.optimizer_name)
        
        # Test 2 pretrained components
        dis_metrics = model.evaluate_component("discriminator", heads, tails, test_data)
        print(f"Discriminator metrics:\n{dis_metrics}")

        gen_metrics = model.evaluate_component("generator", heads, tails, test_data)
        print(f"Generator metrics:\n{gen_metrics}")

    elif args.mode == 'train':
        # Load 2 pretrained components
        model.load_component(component_role="discriminator", component_path=dis_model_path)
        model.load_component(component_role="generator", component_path=gen_model_path)

        # Train KBGAN
        best_perf, model_path = model.train(heads, tails, train_data, valid_data,
                                            use_early_stopping=args.early_stopping_train, patience=args.patience, optimizer_name=args.optimizer_name)
        
    elif args.mode == 'evaluate_on_link_prediction':
        # Load pretrained KBGAN
        model.load()

        # Test KBGAN on link prediction
        link_prediction_metrics = model.evaluate_on_link_prediction(heads, tails, test_data)
        print(f"Link prediction metrics:\n{link_prediction_metrics}")

    elif args.mode == 'evaluate_on_triple_classification':
        # Load pretrained KBGAN
        model.load()

        # Test KBGAN on triple classification
        triple_classification_metrics, threshold, predictions, scores_list = model.evaluate_on_triple_classification(test_data_with_label, valid_data_with_label, threshold=None, auto_threshold=True)
        print(f"Triple classification metrics:\n{triple_classification_metrics}")

if __name__ == '__main__':
    main(mode='full', early_stopping_pretrain=True, early_stopping_train=True, patience=100, optimizer_name='Adagrad')