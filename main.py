import torch
import os
import sys
import time
from config import config, overwrite_config_with_args, logger_init, log_step
from data_loader import index_entity_relation, graph_size, read_data
from datasets import convert_data_to_no_label, sparse_heads_tails, inplace_shuffle
from kbgan import KBGAN

# ./main.py mode=<mode> [other optional args to overwrite config]
MODE = 'full-train'  # full-train / gan-train / test-only
DATASET = 'wn18rr'   # wn18rr / wn18 / fb15k237
RANK_OPTIMIZING_METRIC = 'mrr'              # Metric to optimize for ranking evaluation (e.g., 'mrr', 'hits@1', etc.)
RANK_FILT = True                            # Whether to apply filtering in ranking metrics evaluation
RANK_K_LIST = [1, 3, 10]                    # Default k values for ranking metrics
CLASS_OPTIMIZING_METRIC = 'accuracy'        # Metric to optimize for triple classification (e.g., 'accuracy', 'f1', etc.)
CLASS_USE_MAXGOOD_MINBAD_THRESHOLD = True   # Whether to use dynamic threshold based on max_d_good and min_d_bad for classification

def main():
    #config_path = './config/config_' + DATASET + '.yaml'
    config_path = './config/config_' + DATASET + '_test.yaml' # Use the test config with smaller epochs for quick testing

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

    class_rank_balance = _config['KBGAN']['class_rank_balance']
    early_stop_patience = _config['KBGAN']['early_stop_patience']
    temperature = _config['KBGAN']['temperature']
    n_sample = _config['KBGAN']['n_sample']
    # For hard negative mining with topk, use larger pool by default (100 candidates, select 20)
    # If negative_sampling_strategy is 'multinomial', pool size doesn't matter (all equal probability)
    negative_sampling_strategy = _config['KBGAN'].get('negative_sampling_strategy', 'topk')
    if negative_sampling_strategy == 'topk':
        # Hard mining: default pool size = 100 to select top n_sample=20
        n_candidate = _config['KBGAN'].get('n_candidate', 100)
    else:
        # Multinomial: pool size equals sample size (no hard mining)
        n_candidate = _config['KBGAN'].get('n_candidate', n_sample)
    n_epoch = _config['KBGAN']['n_epoch']
    n_batch = _config['KBGAN']['n_batch']
    epoch_per_test = _config['KBGAN']['epoch_per_test']
    kbg_cfg = _config['KBGAN']
    n_generated_valid_negative = kbg_cfg.get('n_generated_valid_negative', 5)
    join_loss_method = kbg_cfg.get('join_loss_method', 'adaptive_norm')
    loss_ema_beta = kbg_cfg.get('loss_ema_beta', 0.98)
    class_rank_balance_start = kbg_cfg.get('class_rank_balance_start', 0.2)
    class_rank_balance_warmup_epochs = kbg_cfg.get('class_rank_balance_warmup_epochs', 10)

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
    valid_data_with_labels = read_data(os.path.join('.', 'data', DATASET + '_w_labels', 'valid.txt'), kb_index, with_label=True)
    test_data_with_labels  = read_data(os.path.join('.', 'data', DATASET + '_w_labels', 'test.txt'), kb_index, with_label=True)

    test_data_no_label = convert_data_to_no_label(test_data_with_labels)
    valid_data_no_label = convert_data_to_no_label(valid_data_with_labels)
    heads, tails = sparse_heads_tails(n_entity, train_data, valid_data_no_label, test_data_no_label)
    t_step = log_step("Data load", t_step)

    # Convert to tensors
    train_data  = [torch.LongTensor(vec) for vec in train_data]
    valid_data_with_labels  = [torch.LongTensor(vec) for vec in valid_data_with_labels]
    test_data_with_labels   = [torch.LongTensor(vec) for vec in test_data_with_labels]
    t_step = log_step("Tensor conversion", t_step)

    print(f"Running mode: {MODE}")
    model = KBGAN(discriminator_type=dis_type, generator_type=gen_type,
                  n_entity=n_entity, n_relation=n_relation)

    if MODE == 'full-train':
        # Train 2 components
        print(f"Training {dis_type} discriminator and {gen_type} generator components with paramenters:\n\tclass_rank_balance={class_rank_balance}\n\tearly_stop_patience={early_stop_patience}\n"
            + f"\trank_optimizing_metric={RANK_OPTIMIZING_METRIC}\n\trank_filt={RANK_FILT}\n\trank_k_list={RANK_K_LIST}\n"
            + f"\tclass_optimizing_metric={CLASS_OPTIMIZING_METRIC}\n\tclass_threshold=None")
        dis_best_perf, gen_best_perf = model.train_components(heads, tails, train_data, valid_data_with_labels,
                                                            class_rank_balance=class_rank_balance, early_stop_patience=early_stop_patience,
                                                            rank_optimizing_metric=RANK_OPTIMIZING_METRIC, rank_filt=RANK_FILT, rank_k_list=RANK_K_LIST,
                                                            class_optimizing_metric=CLASS_OPTIMIZING_METRIC)
        t_step = log_step("Pretrain components", t_step)
        print("----------------")

        # Test 2 trained components on link prediction
        print(f"Testing component on Link Prediction: {dis_type} discriminator")
        dis_ranking_metrics = model.discriminator.evaluate_on_ranking(test_data_no_label, heads, tails,
                                                                    filt=RANK_FILT, k_list=RANK_K_LIST)
        
        print(f"Testing component on Link Prediction: {gen_type} generator")
        gen_ranking_metrics = model.generator.evaluate_on_ranking(test_data_no_label, heads, tails,
                                                                filt=RANK_FILT, k_list=RANK_K_LIST)

        t_step = log_step("Component link prediction eval", t_step)
        print("----------------")

        # Test 2 trained components on triple classification
        print(f"Testing component on Triple Classification: {dis_type} discriminator")
        dis_classification_metrics = model.discriminator.evaluate_on_classification(test_data_with_labels,
                                                                                    optimizing_metric=CLASS_OPTIMIZING_METRIC, is_threshold_tunning=False)
        print(f"Classification threshold for Discriminator: {model.discriminator.classification_threshold}")

        print(f"Testing component on Triple Classification: {gen_type} generator")
        gen_classification_metrics = model.generator.evaluate_on_classification(test_data_with_labels,
                                                                                optimizing_metric=CLASS_OPTIMIZING_METRIC, is_threshold_tunning=False)
        print(f"Classification threshold for Generator: {model.generator.classification_threshold}")

        t_step = log_step("Component triple classification eval", t_step)
        print("----------------")

        # Train KBGAN
        print(f"Training KBGAN with paramenters:\n\tclass_rank_balance={class_rank_balance}\n\tearly_stop_patience={early_stop_patience}\n"
            + f"\ttemperature={temperature}\n\tn_sample={n_sample}\n\tn_candidate={n_candidate}\n\tnegative_sampling_strategy={negative_sampling_strategy}\n"
            + f"\tn_epoch={n_epoch}\n\tn_batch={n_batch}\n\tepoch_per_test={epoch_per_test}\n"
            + f"\trank_optimizing_metric={RANK_OPTIMIZING_METRIC}\n\trank_filt={RANK_FILT}\n\trank_k_list={RANK_K_LIST}\n"
            + f"\tclass_optimizing_metric={CLASS_OPTIMIZING_METRIC}\n\tclass_use_maxgood_minbad_threshold={CLASS_USE_MAXGOOD_MINBAD_THRESHOLD}\n"
            + f"\tn_generated_valid_negative={n_generated_valid_negative}\n\tjoin_loss_method={join_loss_method}\n\tloss_ema_beta={loss_ema_beta}\n"
            + f"\tclass_rank_balance_start={class_rank_balance_start}\n\tclass_rank_balance_warmup_epochs={class_rank_balance_warmup_epochs}")
        best_perf = model.train_kbgan(heads, tails, train_data, valid_data_with_labels,
                                    class_rank_balance=class_rank_balance, early_stop_patience=early_stop_patience,
                                    temperature=temperature, n_sample=n_sample, n_candidate=n_candidate,
                                    n_epoch=n_epoch, n_batch=n_batch, epoch_per_test=epoch_per_test,
                                    rank_optimizing_metric=RANK_OPTIMIZING_METRIC, rank_filt=RANK_FILT, rank_k_list=RANK_K_LIST,
                                    class_optimizing_metric=CLASS_OPTIMIZING_METRIC, class_use_maxgood_minbad_threshold=CLASS_USE_MAXGOOD_MINBAD_THRESHOLD,
                                    n_generated_valid_negative=n_generated_valid_negative,
                                    class_rank_balance_start=class_rank_balance_start,
                                    class_rank_balance_warmup_epochs=class_rank_balance_warmup_epochs,
                                    negative_sampling_strategy=negative_sampling_strategy,
                                    join_loss_method=join_loss_method, loss_ema_beta=loss_ema_beta)
        print(f"Best validation performance while training: {best_perf}")
        t_step = log_step("Train KBGAN", t_step)
        print("----------------")
        
        # Test KBGAN on link prediction
        print("Testing KBGAN on Link Prediction...")
        link_prediction_metrics = model.evaluate_on_link_prediction(heads, tails, test_data_no_label,
                                                                    filt=RANK_FILT, k_list=RANK_K_LIST)
        t_step = log_step("KBGAN link prediction eval", t_step)
        print("----------------")

        # Test KBGAN on triple classification
        print("Testing KBGAN on Triple Classification...")
        triple_classification_metrics = model.evaluate_on_triple_classification(test_data_with_labels,
                                                                                optimizing_metric=CLASS_OPTIMIZING_METRIC, use_maxgood_minbad_threshold=CLASS_USE_MAXGOOD_MINBAD_THRESHOLD)
        t_step = log_step("KBGAN triple classification eval", t_step)
        print("----------------")
    elif MODE == 'gan-train':
        # Load 2 pretrained components
        print(f"Loading pretrained component: {dis_type} discriminator...")
        model.load_discriminator(pretrained_dis_path)
        print(f"Loading pretrained component: {gen_type} generator...")
        model.load_generator(pretrained_gen_path)
        print("----------------")

        # Train KBGAN
        print(f"Training KBGAN with paramenters:\n\tclass_rank_balance={class_rank_balance}\n\tearly_stop_patience={early_stop_patience}\n"
            + f"\ttemperature={temperature}\n\tn_sample={n_sample}\n\tn_candidate={n_candidate}\n\tnegative_sampling_strategy={negative_sampling_strategy}\n"
            + f"\tn_epoch={n_epoch}\n\tn_batch={n_batch}\n\tepoch_per_test={epoch_per_test}\n"
            + f"\trank_optimizing_metric={RANK_OPTIMIZING_METRIC}\n\trank_filt={RANK_FILT}\n\trank_k_list={RANK_K_LIST}\n"
            + f"\tclass_optimizing_metric={CLASS_OPTIMIZING_METRIC}\n\tclass_use_maxgood_minbad_threshold={CLASS_USE_MAXGOOD_MINBAD_THRESHOLD}\n"
            + f"\tn_generated_valid_negative={n_generated_valid_negative}\n\tjoin_loss_method={join_loss_method}\n\tloss_ema_beta={loss_ema_beta}\n"
            + f"\tclass_rank_balance_start={class_rank_balance_start}\n\tclass_rank_balance_warmup_epochs={class_rank_balance_warmup_epochs}")
        best_perf = model.train_kbgan(heads, tails, train_data, valid_data_with_labels,
                                    class_rank_balance=class_rank_balance, early_stop_patience=early_stop_patience,
                                    temperature=temperature, n_sample=n_sample, n_candidate=n_candidate,
                                    n_epoch=n_epoch, n_batch=n_batch, epoch_per_test=epoch_per_test,
                                    rank_optimizing_metric=RANK_OPTIMIZING_METRIC, rank_filt=RANK_FILT, rank_k_list=RANK_K_LIST,
                                    class_optimizing_metric=CLASS_OPTIMIZING_METRIC, class_use_maxgood_minbad_threshold=CLASS_USE_MAXGOOD_MINBAD_THRESHOLD,
                                    n_generated_valid_negative=n_generated_valid_negative,
                                    class_rank_balance_start=class_rank_balance_start,
                                    class_rank_balance_warmup_epochs=class_rank_balance_warmup_epochs,
                                    negative_sampling_strategy=negative_sampling_strategy,
                                    join_loss_method=join_loss_method, loss_ema_beta=loss_ema_beta)
        print(f"Best validation performance while training: {best_perf}")
        t_step = log_step("Train KBGAN", t_step)
        print("----------------")

        # Test KBGAN on task
        if working_task == 'link-prediction' or working_task == 'all':
            link_prediction_metrics = model.evaluate_on_link_prediction(heads, tails, test_data_no_label,
                                                                        filt=RANK_FILT, k_list=RANK_K_LIST)
            t_step = log_step("KBGAN link prediction eval", t_step)

        if working_task == 'triple-classification' or working_task == 'all':
            triple_classification_metrics = model.evaluate_on_triple_classification(test_data_with_labels,
                                                                                    optimizing_metric=CLASS_OPTIMIZING_METRIC, use_maxgood_minbad_threshold=CLASS_USE_MAXGOOD_MINBAD_THRESHOLD)
            t_step = log_step("KBGAN triple classification eval", t_step)
        print("----------------")
    elif MODE == 'test-only':
        # Load pretrained KBGAN
        model.load_kbgan(pretrained_kbgan_path)
        print("----------------")

        # Test KBGAN on task
        if working_task == 'link-prediction' or working_task == 'all':
            link_prediction_metrics = model.evaluate_on_link_prediction(heads, tails, test_data_no_label,
                                                                        filt=RANK_FILT, k_list=RANK_K_LIST)

        if working_task == 'triple-classification' or working_task == 'all':
            triple_classification_metrics = model.evaluate_on_triple_classification(test_data_with_labels,
                                                                                    optimizing_metric=CLASS_OPTIMIZING_METRIC, use_maxgood_minbad_threshold=CLASS_USE_MAXGOOD_MINBAD_THRESHOLD)
            t_step = log_step("KBGAN triple classification eval", t_step)
        print("----------------")
    else: 
        print("Invalid mode. Please try again and specify a mode: full-train / gan-train / test-only") 
    total_elapsed = time.perf_counter() - t_total
    print(f"[TIMER] Total runtime: {total_elapsed:.2f}s")

if __name__ == '__main__':
    main()