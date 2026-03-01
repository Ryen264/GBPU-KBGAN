import torch
import os
import sys
import time
from config import config, overwrite_config_with_args, logger_init, log_step
from data_loader import index_entity_relation, graph_size, read_data
from datasets import sparse_heads_tails, inplace_shuffle
from kbgan import KBGAN

MODE = 'full-train'  # full-train / gan-train / test-only
DATASET = 'wn18rr'   # wn18rr / wn18 / fb15k237

# ./main.py mode=<mode> [other optional args to overwrite config]

K_LIST = [1, 3, 10]                     # Default k values for ranking metrics
RANK_FILT = True                        # Whether to apply filtering in ranking metrics evaluation
OPTIMIZING_CLASS_METRIC = 'accuracy'    # Metric to optimize for triple classification (e.g., 'accuracy', 'f1', etc.)

def main():
    config_path = './config/config_' + DATASET + '.yaml'
    _config = config(config_path)
    working_task = _config.task # link-prediction / triple-classification / all (all for 'full-train' mode)

    global MODE
    if len(sys.argv) > 1:
        MODE = sys.argv[1].split('=')[1]
    args = sys.argv[2:]
    if args:
        overwrite_config_with_args(args)
        print("Running config: ", _config)

    dis_type, gen_type = _config.d_config, _config.g_config

    # Assign or construct pretrained components' paths for 'gan-train' mode
    pretrained_dis_path = os.path.join('.', 'models', DATASET, working_task, 'components', dis_type + '.mdl')
    pretrained_gen_path = os.path.join('.', 'models', DATASET, working_task, 'components', gen_type + '.mdl')

    # Assign or construct pretrained KBGAN's path for 'test-only' mode
    pretrained_kbgan_path = os.path.join('.', 'models', DATASET, working_task, 'kbgan.mdl')

    t_total = time.perf_counter()
    # Init logging now that config is prepared
    logger_init()
    t_step = time.perf_counter()

    # Load data
    task_dir = os.path.join('.', 'data', DATASET)
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
    t_step = log_step("Data load", t_step)

    # For task triple-classification, we need to read data with labels
    if _config.task == 'triple-classification' or _config.task == 'all':
        valid_data_with_label = read_data(os.path.join('.', 'data', DATASET + '_w_labels', 'valid.txt'), kb_index, with_label=True)
        test_data_with_label  = read_data(os.path.join('.', 'data', DATASET + '_w_labels', 'test.txt'), kb_index, with_label=True)
        t_step = log_step("Labelled data load", t_step)

    # Convert to tensors
    train_data  = [torch.LongTensor(vec) for vec in train_data]
    valid_data  = [torch.LongTensor(vec) for vec in valid_data]
    test_data   = [torch.LongTensor(vec) for vec in test_data]
    t_step = log_step("Tensor conversion", t_step)

    print(f"Running mode: {MODE}")
    model = KBGAN(discriminator_type=dis_type, generator_type=gen_type, n_entity=n_entity, n_relation=n_relation)

    optimizer_name = _config['KBGAN']['optimizer_name']
    rank_class_balance = _config['KBGAN']['rank_class_balance']
    early_stop_patience = _config['KBGAN']['early_stop_patience']
    if MODE == 'full-train':
        # Train 2 components
        dis_best_perf, gen_best_perf = model.train_components(heads, tails, train_data, valid_data_with_label,
                                                            optimizer_name=optimizer_name, rank_class_balance=rank_class_balance,
                                                            early_stop_patience=early_stop_patience)
        t_step = log_step("Pretrain components", t_step)
        print("----------------")

        # Test 2 components just be trained on link prediction
        dis_ranking_metrics = model.discriminator.evaluate_on_ranking(test_data, heads, tails,
                                                                    filt=RANK_FILT, k_list=K_LIST)
        print(f"Discriminator metrics on Link Prediction: {dis_ranking_metrics}")

        gen_ranking_metrics = model.generator.evaluate_on_ranking(test_data, heads, tails,
                                                                filt=RANK_FILT, k_list=K_LIST)
        print(f"Generator metrics on Link Prediction: {gen_ranking_metrics}")
        t_step = log_step("Component link prediction eval", t_step)
        print("----------------")

        # Test 2 components just be trained on triple classification
        dis_classification_metrics = model.discriminator.evaluate_on_classification(test_data_with_label,
                                                                                    optimizing_metric=OPTIMIZING_CLASS_METRIC)
        print(f"Discriminator metrics on Triple Classification: {dis_classification_metrics}")
        t_step = log_step("Component triple classification eval", t_step)

        gen_classification_metrics = model.generator.evaluate_on_classification(test_data_with_label,
                                                                                optimizing_metric=OPTIMIZING_CLASS_METRIC)
        print(f"Generator metrics on Triple Classification: {gen_classification_metrics}")
        print("----------------")

        # Train KBGAN
        best_perf = model.train_kbgan(heads, tails, train_data, valid_data_with_label,
                                    optimizer_name=optimizer_name, rank_class_balance=rank_class_balance,
                                    early_stop_patience=early_stop_patience)        
        print(f"Best validation performance while training: {best_perf}")
        t_step = log_step("Train KBGAN", t_step)
        print("----------------")
        
        # Test KBGAN on link prediction
        link_prediction_metrics = model.evaluate_on_link_prediction(heads, tails, test_data,
                                                                    filt=RANK_FILT, k_list=K_LIST)
        print(f"Link prediction metrics:\n{link_prediction_metrics}")
        t_step = log_step("KBGAN link prediction eval", t_step)
        print("----------------")

        # Test KBGAN on triple classification
        triple_classification_metrics = model.evaluate_on_triple_classification(test_data_with_label,
                                                                                optimizing_metric=OPTIMIZING_CLASS_METRIC)
        print(f"Triple classification metrics:\n{triple_classification_metrics}")
        t_step = log_step("KBGAN triple classification eval", t_step)
        print("----------------")
    elif MODE == 'gan-train':
        # Load 2 pretrained components
        model.load_discriminator(pretrained_dis_path)
        model.load_generator(pretrained_gen_path)
        print("----------------")

        # Train KBGAN
        best_perf = model.train_kbgan(heads, tails, train_data, valid_data_with_label,
                                    optimizer_name=optimizer_name, rank_class_balance=rank_class_balance,
                                    early_stop_patience=early_stop_patience)        
        print(f"Best validation performance while training: {best_perf}")
        t_step = log_step("Train KBGAN", t_step)
        print("----------------")

        # Test KBGAN on task
        if working_task == 'link-prediction' or working_task == 'all':
            link_prediction_metrics = model.evaluate_on_link_prediction(heads, tails, test_data,
                                                                        filt=RANK_FILT, k_list=K_LIST)
            print(f"Link prediction metrics:\n{link_prediction_metrics}")
            t_step = log_step("KBGAN link prediction eval", t_step)

        if working_task == 'triple-classification' or working_task == 'all':
            triple_classification_metrics = model.evaluate_on_triple_classification(test_data_with_label,
                                                                                    optimizing_metric=OPTIMIZING_CLASS_METRIC)
            print(f"Triple classification metrics:\n{triple_classification_metrics}") 
            t_step = log_step("KBGAN triple classification eval", t_step)
        print("----------------")
    elif MODE == 'test-only':
        # Load pretrained KBGAN
        model.load_kbgan(pretrained_kbgan_path)
        print("----------------")

        # Test KBGAN on task
        if working_task == 'link-prediction' or working_task == 'all':
            link_prediction_metrics = model.evaluate_on_link_prediction(heads, tails, test_data,
                                                                        filt=RANK_FILT, k_list=K_LIST)
            print(f"Link prediction metrics:\n{link_prediction_metrics}")

        if working_task == 'triple-classification' or working_task == 'all':
            triple_classification_metrics = model.evaluate_on_triple_classification(test_data_with_label,
                                                                                    optimizing_metric=OPTIMIZING_CLASS_METRIC)
            print(f"Triple classification metrics:\n{triple_classification_metrics}")
            t_step = log_step("KBGAN triple classification eval", t_step)
        print("----------------")
    else: 
        print("Invalid mode. Please try again and specify a mode: full-train / gan-train / test-only") 

    total_elapsed = time.perf_counter() - t_total
    print(f"[TIMER] Total runtime: {total_elapsed:.2f}s")

if __name__ == '__main__':
    main()