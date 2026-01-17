import torch
import os
import sys
import time
from config import config, overwrite_config_with_args, logger_init
from data_loader import index_entity_relation, graph_size, read_data
from datasets import sparse_heads_tails, inplace_shuffle
from kbgan import KBGAN

MODE = 'full-train'  # full-train / gan-train / test-only

# ./main.py mode=<mode> [other optional args to overwrite config]

def log_step(label: str, start_ts: float) -> float:
    """Print elapsed time for a pipeline step and return a new start timestamp."""
    elapsed = time.perf_counter() - start_ts
    print(f"[TIMER] {label}: {elapsed:.2f}s")
    return time.perf_counter()

def main():    
    _config = config()
    global MODE

    if len(sys.argv) > 1:
        MODE = sys.argv[1].split('=')[1]
    args = sys.argv[2:]
    if args:
        overwrite_config_with_args(args)
        print("Running config:", _config)

    t_total = time.perf_counter()
    _config.dataset = 'wn18rr'
    # _config['KBGAN']['n_epoch'] = 200
    # _config[_config.d_config]['n_epoch'] = 100
    # _config[_config.g_config]['n_epoch'] = 100
    # _config.task = 'all'


    # Init logging now that config is prepared
    logger_init()
    t_step = time.perf_counter()

    # Load data
    task_dir = './data/' + _config.dataset
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
        valid_data_with_label   = read_data(os.path.join('./data/' + _config.dataset + '_w_labels', 'valid.txt'), kb_index, with_label=True)
        test_data_with_label    = read_data(os.path.join('./data/' + _config.dataset + '_w_labels', 'test.txt'), kb_index, with_label=True)
        t_step = log_step("Labelled data load", t_step)

    # Convert to tensors
    train_data  = [torch.LongTensor(vec) for vec in train_data]
    valid_data  = [torch.LongTensor(vec) for vec in valid_data]
    test_data   = [torch.LongTensor(vec) for vec in test_data]
    t_step = log_step("Tensor conversion", t_step)

    print(f"Running mode: {MODE}")
    model = KBGAN(discriminator_type="TransE", generator_type="DistMult",
                  n_entity=n_entity, n_relation=n_relation)
    if MODE == 'full-train':
        # Train 2 components
        dis_best_perf, dis_path, gen_best_perf, gen_path = model.train_components(heads, tails, train_data, valid_data_with_label,
                                                                                    use_early_stopping=_config['KBGAN']['early_stopping_pretrain'],
                                                                                    patience=_config['KBGAN']['patience'],
                                                                                    optimizer_name=_config['KBGAN']['optimizer_name'],
                                                                                    is_save_components=True)
        t_step = log_step("Pretrain components", t_step)
        print("----------------")

        # Test 2 components just be trained on link prediction
        dis_ranking_metrics = model.evaluate_discriminator_on_link_prediction(heads, tails, test_data,
                                                                                filt=True, k_list=[1, 3, 10])
        print(f"Discriminator metrics on Link Prediction: {dis_ranking_metrics}")

        gen_ranking_metrics = model.evaluate_generator_on_link_prediction(heads, tails, test_data,
                                                                            filt=True, k_list=[1, 3, 10])
        print(f"Generator metrics on Link Prediction: {gen_ranking_metrics}")
        t_step = log_step("Component link prediction eval", t_step)
        print("----------------")

        # Test 2 components just be trained on triple classification
        dis_classification_metrics = model.evaluate_discriminator_on_triple_classification(test_data_with_label, optimizing_metric='accuracy')
        print(f"Discriminator metrics on Triple Classification: {dis_classification_metrics}")
        t_step = log_step("Component triple classification eval", t_step)

        gen_classification_metrics = model.evaluate_generator_on_triple_classification(test_data_with_label, optimizing_metric='accuracy')
        print(f"Generator metrics on Triple Classification: {gen_classification_metrics}")
        print("----------------")
#
        # Train KBGAN
        best_perf, kbgan_path = model.train_kbgan(heads, tails, train_data, valid_data,
                                                    use_early_stopping=_config['KBGAN']['early_stopping_train'],
                                                    patience=_config['KBGAN']['patience'],
                                                    optimizer_name=_config['KBGAN']['optimizer_name'],
                                                    is_save_kbgan=True)        
        print(f"Best validation performance while training: {best_perf}")
        t_step = log_step("Train KBGAN", t_step)
        print("----------------")
        
        # Test KBGAN on link prediction
        link_prediction_metrics = model.evaluate_kbgan_on_link_prediction(heads, tails, test_data,
                                                                          filt=True, k_list=[1, 3, 10])
        print(f"Link prediction metrics:\n{link_prediction_metrics}")
        t_step = log_step("KBGAN link prediction eval", t_step)
        print("----------------")

        # Test KBGAN on triple classification
        triple_classification_metrics = model.evaluate_kbgan_on_triple_classification(test_data_with_label, optimizing_metric='accuracy')
        print(f"Triple classification metrics:\n{triple_classification_metrics}")
        t_step = log_step("KBGAN triple classification eval", t_step)
        print("----------------")

    elif MODE == 'gan-train':
        # Load 2 pretrained components
        dis_model_path = './models/' + _config.dataset + '/' + _config.task + '/components/' + _config['d_config'] + '.mdl'
        model.load_discriminator(component_path=dis_model_path)

        gen_model_path = './models/' + _config.dataset + '/' + _config.task + '/components/' + _config['g_config'] + '.mdl'
        model.load_generator(component_path=gen_model_path)
        print("----------------")

        # Train KBGAN
        best_perf, kbgan_path = model.train_kbgan(heads, tails, train_data, valid_data,
                                                    use_early_stopping=_config['KBGAN']['early_stopping_train'],
                                                    patience=_config['KBGAN']['patience'],
                                                    optimizer_name=_config['KBGAN']['optimizer_name'],
                                                    is_save_kbgan=True)        
        print(f"Best validation performance while training: {best_perf}")
        t_step = log_step("Train KBGAN", t_step)
        print("----------------")

        # Test KBGAN on task
        if _config.task == 'link-prediction' or _config.task == 'all':
            link_prediction_metrics = model.evaluate_kbgan_on_link_prediction(heads, tails, test_data,
                                                                              filt=True, k_list=[1, 3, 10])
            print(f"Link prediction metrics:\n{link_prediction_metrics}")
            t_step = log_step("KBGAN link prediction eval", t_step)

        if _config.task == 'triple-classification' or _config.task == 'all':
            triple_classification_metrics = model.evaluate_kbgan_on_triple_classification(test_data_with_label, optimizing_metric='accuracy')
            print(f"Triple classification metrics:\n{triple_classification_metrics}") 
            t_step = log_step("KBGAN triple classification eval", t_step)
        print("----------------")
        
    elif MODE == 'test-only':
        # Load pretrained KBGAN
        kbgan_path = './models/' + _config.dataset + '/' + _config.task + 'kbgan_' + 'dis-' + _config['d_config'] + '_gen-' + _config['g_config'] + '.mdl'
        model.load_kbgan(kbgan_path)
        print("----------------")

        # Test KBGAN on task
        if _config.task == 'link-prediction' or _config.task == 'all':
            link_prediction_metrics = model.evaluate_kbgan_on_link_prediction(heads, tails, test_data,
                                                                              filt=True, k_list=[1, 3, 10])
            print(f"Link prediction metrics:\n{link_prediction_metrics}")

        if _config.task == 'triple-classification' or _config.task == 'all':
            triple_classification_metrics = model.evaluate_kbgan_on_triple_classification(test_data_with_label, optimizing_metric='accuracy')
            print(f"Triple classification metrics:\n{triple_classification_metrics}")
            t_step = log_step("KBGAN triple classification eval", t_step)
        print("----------------")
    else: 
        print("Invalid mode. Please try again and specify a mode: full-train / gan-train / test-only") 

    total_elapsed = time.perf_counter() - t_total
    print(f"[TIMER] Total runtime: {total_elapsed:.2f}s")

if __name__ == '__main__':
    main()