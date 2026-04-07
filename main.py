import torch
import os
import sys
import time
import logging
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


def _log_metrics(title: str, metrics: dict) -> None:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    message = f"{title}: " + ", ".join(parts)
    print(message)
    logging.info(message)


def _normalized_mode(mode: str) -> str:
    return mode.replace('-', '_')


def _activate_logger(prefix: str) -> None:
    overwrite_config_with_args([f"--log.prefix={prefix}"])
    logger_init()


def _format_summary_value(value):
    if value is None:
        return 'N/A'
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_summary_report(summary: dict) -> str:
    output_dir = os.path.join('.', 'output', summary['dataset'])
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime('%y%m%d-%H%M%S')
    report_path = os.path.join(output_dir, f"summary_{timestamp}.txt")

    lines = [
        'RUN SUMMARY REPORT',
        f"timestamp: {timestamp}",
        f"dataset: {summary['dataset']}",
        f"task: {summary['task']}",
        f"mode: {summary['mode']}",
        f"config_path: {summary.get('config_path', 'N/A')}",
        f"discriminator: {summary.get('dis_type', 'N/A')}",
        f"generator: {summary.get('gen_type', 'N/A')}",
        f"checkpoint: {summary.get('checkpoint_path', 'N/A')}",
        f"component_training_time_s: {_format_summary_value(summary.get('component_training_time_s', 'N/A'))}",
        f"kbgan_training_time_s: {_format_summary_value(summary.get('kbgan_training_time_s', 'N/A'))}",
        f"best_training_perf: {_format_summary_value(summary.get('best_training_perf', 'N/A'))}",
        f"best_validation_perf: {_format_summary_value(summary.get('best_validation_perf', 'N/A'))}",
        f"final_validation_perf: {_format_summary_value(summary.get('final_validation_perf', 'N/A'))}",
        f"total_runtime_s: {_format_summary_value(summary.get('total_runtime_s', 'N/A'))}",
        '',
    ]

    component_metrics = summary.get('component_metrics')
    if component_metrics:
        lines.append('COMPONENT TRAINING')
        for name, value in component_metrics.items():
            lines.append(f"  {name}: {_format_summary_value(value)}")
        lines.append('')

    test_metrics = summary.get('test_metrics', {})
    if test_metrics:
        lines.append('TEST RESULTS')
        for task_name, metrics in test_metrics.items():
            lines.append(f"{task_name}:")
            for key, value in metrics.items():
                lines.append(f"  {key}: {_format_summary_value(value)}")
        lines.append('')

    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write('\n'.join(lines).rstrip() + '\n')

    logging.info('Wrote run summary report to %s', report_path)
    print(f"Wrote run summary report to: {report_path}")
    return report_path

def main():
    # Check if config path is provided as first argument
    global MODE
    config_path = None
    args_start_index = 1
    
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        # If first argument looks like a config file path (contains .yaml), use it
        if first_arg.endswith('.yaml'):
            config_path = first_arg
            args_start_index = 2
    
    # Use provided config path or fall back to default
    if config_path is None:
        config_path = './config/config_' + DATASET + '.yaml'
    
    _config = config(config_path)
    working_task = _config.task # link-prediction / triple-classification / all (all for 'full-train' mode)

    # Parse remaining arguments for mode and config overrides
    if len(sys.argv) > args_start_index:
        first_remaining = sys.argv[args_start_index]
        # Check if it's a mode specification (mode=...)
        if '=' in first_remaining and first_remaining.startswith('mode='):
            MODE = first_remaining.split('=')[1]
            args = sys.argv[args_start_index + 1:]
        else:
            args = sys.argv[args_start_index:]
    else:
        args = []
    normalized_mode = _normalized_mode(MODE)
    if args:
        overwrite_config_with_args(args)
        print("Running config: ", _config)

    # Capture startup and data-loading messages in a run-scoped log before switching to stage logs.
    _activate_logger('run_')

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
    # v4 Unaugmented Validation: no generator-created negatives injected into validation set
    emb_uniform_scale = _config['KBGAN'].get('emb_uniform_scale', 2.0)
    entity_uniform_max_ids = _config['KBGAN'].get('entity_uniform_max_ids', 2048)
    uniform_gamma = _config['KBGAN'].get('uniform_lambda', _config['KBGAN'].get('uniform_gamma', 1.0))
    true_align_gamma = _config['KBGAN'].get('true_align_gamma', _config['KBGAN'].get('emb_loss_gamma', 1.0))
    fake_align_gamma = _config['KBGAN'].get('alpha', _config['KBGAN'].get('fake_align_gamma', 1.0))
    safe_margin = _config['KBGAN'].get('safe_margin', _config['KBGAN'].get('mu', 1.0))
    emb_align_balance = _config['KBGAN'].get('emb_align_balance', 0.7)
    emb_align_op = _config['KBGAN'].get('emb_align_op', 'add')
    
    # Assign or construct pretrained components' paths for 'gan-train' mode
    pretrained_dis_path = os.path.join('.', 'models', DATASET, working_task, 'components', dis_type + '.mdl')
    pretrained_gen_path = os.path.join('.', 'models', DATASET, working_task, 'components', gen_type + '.mdl')

    # Assign or construct pretrained KBGAN's path for test-only mode
    pretrained_kbgan_path = os.path.join('.', 'models', DATASET, working_task, f'kbgan_dis-{dis_type}_gen-{gen_type}.mdl')

    run_summary = {
        'dataset': DATASET,
        'task': working_task,
        'mode': MODE,
        'config_path': config_path,
        'dis_type': dis_type,
        'gen_type': gen_type,
        'checkpoint_path': pretrained_kbgan_path,
        'component_metrics': None,
        'test_metrics': {},
        'component_training_time_s': None,
        'kbgan_training_time_s': None,
        'best_training_perf': None,
        'best_validation_perf': None,
        'final_validation_perf': None,
        'total_runtime_s': None,
    }

    t_total = time.perf_counter()
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

    if normalized_mode == 'full_train':
        # Train 2 components
        print(f"Training {dis_type} discriminator and {gen_type} generator components with paramenters:\n\tclass_rank_balance={class_rank_balance}\n\tearly_stop_patience={early_stop_patience}\n"
            + f"\trank_optimizing_metric={RANK_OPTIMIZING_METRIC}\n\trank_filt={RANK_FILT}\n\trank_k_list={RANK_K_LIST}\n"
            + f"\tclass_optimizing_metric={CLASS_OPTIMIZING_METRIC}\n\tclass_threshold=None")
        component_train_start = time.perf_counter()
        dis_best_perf, gen_best_perf = model.train_components(heads, tails, train_data, valid_data_with_labels,
                                                            class_rank_balance=class_rank_balance, early_stop_patience=early_stop_patience,
                                                            rank_optimizing_metric=RANK_OPTIMIZING_METRIC, rank_filt=RANK_FILT, rank_k_list=RANK_K_LIST,
                                                            class_optimizing_metric=CLASS_OPTIMIZING_METRIC)
        run_summary['component_metrics'] = {
            f'{dis_type}_best_validation_perf': dis_best_perf,
            f'{gen_type}_best_validation_perf': gen_best_perf,
        }
        run_summary['component_training_time_s'] = time.perf_counter() - component_train_start
        t_step = log_step("Pretrain components", t_step)
        print("----------------")

        # Test 2 trained components on link prediction
        print(f"Testing component on Link Prediction: {dis_type} discriminator")
        dis_ranking_metrics = model.discriminator.evaluate_on_ranking(test_data_no_label, heads, tails,
                                                                    filt=RANK_FILT, k_list=RANK_K_LIST)
        _log_metrics("Discriminator link prediction", dis_ranking_metrics)
        
        print(f"Testing component on Link Prediction: {gen_type} generator")
        gen_ranking_metrics = model.generator.evaluate_on_ranking(test_data_no_label, heads, tails,
                                                                filt=RANK_FILT, k_list=RANK_K_LIST)
        _log_metrics("Generator link prediction", gen_ranking_metrics)

        t_step = log_step("Component link prediction eval", t_step)
        print("----------------")

        # Test 2 trained components on triple classification
        print(f"Testing component on Triple Classification: {dis_type} discriminator")
        dis_classification_metrics = model.discriminator.evaluate_on_classification(test_data_with_labels,
                                                                                    optimizing_metric=CLASS_OPTIMIZING_METRIC, is_threshold_tunning=False)
        _log_metrics("Discriminator triple classification", dis_classification_metrics)
        print(f"Classification threshold for Discriminator: {model.discriminator.classification_threshold}")

        print(f"Testing component on Triple Classification: {gen_type} generator")
        gen_classification_metrics = model.generator.evaluate_on_classification(test_data_with_labels,
                                                                                optimizing_metric=CLASS_OPTIMIZING_METRIC, is_threshold_tunning=False)
        _log_metrics("Generator triple classification", gen_classification_metrics)
        print(f"Classification threshold for Generator: {model.generator.classification_threshold}")

        t_step = log_step("Component triple classification eval", t_step)
        print("----------------")

        _activate_logger(f'KBGAN_{dis_type}_{gen_type}_')

        # Train KBGAN (v4 Bounded Hybrid)
        print(f"Training KBGAN (v4 Bounded Hybrid) with parameters:\n\tclass_rank_balance={class_rank_balance}\n\tearly_stop_patience={early_stop_patience}\n"
            f"\ttemperature={temperature}\n\tn_sample={n_sample}\n\tn_candidate={n_candidate}\n\tnegative_sampling_strategy={negative_sampling_strategy}\n"
            f"\tn_epoch={n_epoch}\n\tn_batch={n_batch}\n\tepoch_per_test={epoch_per_test}\n"
            f"\trank_optimizing_metric={RANK_OPTIMIZING_METRIC}\n\trank_filt={RANK_FILT}\n\trank_k_list={RANK_K_LIST}\n"
            f"\tclass_optimizing_metric={CLASS_OPTIMIZING_METRIC}\n"
            f"\ttrue_align_gamma={true_align_gamma}\n\temb_uniform_scale={emb_uniform_scale}\n\tentity_uniform_max_ids={entity_uniform_max_ids}\n\tuniformity_weight(lambda)={uniform_gamma}\n\temb_align_op={emb_align_op}\n\temb_align_balance={emb_align_balance}\n\tfake_alignment_weight(alpha)={fake_align_gamma}\n\tsafe_margin(mu)={safe_margin}")
        kbgan_train_start = time.perf_counter()
        best_perf = model.train_kbgan(heads, tails, train_data, valid_data_with_labels,
                                    class_rank_balance=class_rank_balance,
                                    early_stop_patience=early_stop_patience,
                                    temperature=temperature,
                                    n_sample=n_sample,
                                    n_candidate=n_candidate,
                                    n_epoch=n_epoch,
                                    n_batch=n_batch,
                                    epoch_per_test=epoch_per_test,
                                    negative_sampling_strategy=negative_sampling_strategy,
                                    emb_uniform_scale=emb_uniform_scale,
                                    entity_uniform_max_ids=entity_uniform_max_ids,
                                    uniform_gamma=uniform_gamma,
                                    true_align_gamma=true_align_gamma,
                                    fake_align_gamma=fake_align_gamma,
                                    safe_margin=safe_margin,
                                    emb_align_op=emb_align_op,
                                    emb_align_balance=emb_align_balance,
                                    alpha=fake_align_gamma,
                                    uniform_lambda=uniform_gamma,
                                    rank_optimizing_metric=RANK_OPTIMIZING_METRIC,
                                    rank_filt=RANK_FILT,
                                    rank_k_list=RANK_K_LIST,
                                    class_optimizing_metric=CLASS_OPTIMIZING_METRIC
                                    )
        run_summary['kbgan_training_time_s'] = time.perf_counter() - kbgan_train_start
        run_summary['best_training_perf'] = best_perf
        run_summary['best_validation_perf'] = getattr(model, 'best_validation_perf', best_perf)
        run_summary['final_validation_perf'] = getattr(model, 'final_validation_perf', best_perf)
        print(f"Best validation performance while training: {best_perf}")
        t_step = log_step("Train KBGAN", t_step)
        print("----------------")
        
        # Test KBGAN on link prediction
        print("Testing KBGAN on Link Prediction...")
        link_prediction_metrics = model.evaluate_on_link_prediction(heads, tails, test_data_no_label,
                                                                    filt=RANK_FILT, k_list=RANK_K_LIST)
        t_step = log_step("KBGAN link prediction eval", t_step)
        run_summary['test_metrics']['link_prediction'] = link_prediction_metrics
        print("----------------")

        # Test KBGAN on triple classification
        print("Testing KBGAN on Triple Classification...")
        triple_classification_metrics = model.evaluate_on_triple_classification(test_data_with_labels,
                                                                                optimizing_metric=CLASS_OPTIMIZING_METRIC)
        t_step = log_step("KBGAN triple classification eval", t_step)
        run_summary['test_metrics']['triple_classification'] = triple_classification_metrics
        print("----------------")
    elif normalized_mode == 'gan_train':
        # Load 2 pretrained components
        print(f"Loading pretrained component: {dis_type} discriminator...")
        model.load_discriminator(pretrained_dis_path)
        print(f"Loading pretrained component: {gen_type} generator...")
        model.load_generator(pretrained_gen_path)
        print("----------------")

        _activate_logger(f'KBGAN_{dis_type}_{gen_type}_')

        # Train KBGAN (v4 Bounded Hybrid)
        print(f"Training KBGAN (v4 Bounded Hybrid) with parameters:\n\tclass_rank_balance={class_rank_balance}\n\tearly_stop_patience={early_stop_patience}\n"
            f"\ttemperature={temperature}\n\tn_sample={n_sample}\n\tn_candidate={n_candidate}\n\tnegative_sampling_strategy={negative_sampling_strategy}\n"
            f"\tn_epoch={n_epoch}\n\tn_batch={n_batch}\n\tepoch_per_test={epoch_per_test}\n"
            f"\trank_optimizing_metric={RANK_OPTIMIZING_METRIC}\n\trank_filt={RANK_FILT}\n\trank_k_list={RANK_K_LIST}\n"
            f"\tclass_optimizing_metric={CLASS_OPTIMIZING_METRIC}\n"
            f"\ttrue_align_gamma={true_align_gamma}\n\temb_uniform_scale={emb_uniform_scale}\n\tentity_uniform_max_ids={entity_uniform_max_ids}\n\tuniformity_weight(lambda)={uniform_gamma}\n\temb_align_op={emb_align_op}\n\temb_align_balance={emb_align_balance}\n\tfake_alignment_weight(alpha)={fake_align_gamma}\n\tsafe_margin(mu)={safe_margin}")
        kbgan_train_start = time.perf_counter()
        best_perf = model.train_kbgan(heads, tails, train_data, valid_data_with_labels,
                                    class_rank_balance=class_rank_balance,
                                    early_stop_patience=early_stop_patience,
                                    temperature=temperature,
                                    n_sample=n_sample,
                                    n_candidate=n_candidate,
                                    n_epoch=n_epoch,
                                    n_batch=n_batch,
                                    epoch_per_test=epoch_per_test,
                                    negative_sampling_strategy=negative_sampling_strategy,
                                    emb_uniform_scale=emb_uniform_scale,
                                    entity_uniform_max_ids=entity_uniform_max_ids,
                                    uniform_gamma=uniform_gamma,
                                    true_align_gamma=true_align_gamma,
                                    fake_align_gamma=fake_align_gamma,
                                    safe_margin=safe_margin,
                                    emb_align_op=emb_align_op,
                                    emb_align_balance=emb_align_balance,
                                    alpha=fake_align_gamma,
                                    uniform_lambda=uniform_gamma,
                                    rank_optimizing_metric=RANK_OPTIMIZING_METRIC,
                                    rank_filt=RANK_FILT,
                                    rank_k_list=RANK_K_LIST,
                                    class_optimizing_metric=CLASS_OPTIMIZING_METRIC
                                    )
        run_summary['kbgan_training_time_s'] = time.perf_counter() - kbgan_train_start
        run_summary['best_training_perf'] = best_perf
        run_summary['best_validation_perf'] = getattr(model, 'best_validation_perf', best_perf)
        run_summary['final_validation_perf'] = getattr(model, 'final_validation_perf', best_perf)
        print(f"Best validation performance while training: {best_perf}")
        t_step = log_step("Train KBGAN", t_step)
        print("----------------")

        # Test KBGAN on task
        if working_task == 'link-prediction' or working_task == 'all':
            link_prediction_metrics = model.evaluate_on_link_prediction(heads, tails, test_data_no_label,
                                                                        filt=RANK_FILT, k_list=RANK_K_LIST)
            _log_metrics("KBGAN link prediction", link_prediction_metrics)
            run_summary['test_metrics']['link_prediction'] = link_prediction_metrics
            t_step = log_step("KBGAN link prediction eval", t_step)

        if working_task == 'triple-classification' or working_task == 'all':
            triple_classification_metrics = model.evaluate_on_triple_classification(test_data_with_labels,
                                                                                    optimizing_metric=CLASS_OPTIMIZING_METRIC)
            _log_metrics("KBGAN triple classification", triple_classification_metrics)
            run_summary['test_metrics']['triple_classification'] = triple_classification_metrics
            t_step = log_step("KBGAN triple classification eval", t_step)
        print("----------------")
    elif normalized_mode == 'test_only':
        # Load pretrained KBGAN checkpoint before evaluating both tasks
        _activate_logger(f'KBGAN_{dis_type}_{gen_type}_')
        pretrained_kbgan_path = run_summary['checkpoint_path']
        print(f"Loading pretrained KBGAN from: {pretrained_kbgan_path}")
        model.load_kbgan(pretrained_kbgan_path)
        print("----------------")

        if working_task in ('triple-classification', 'all'):
            if model.discriminator.classification_threshold is None and model.discriminator.global_threshold is None:
                print("Tuning classification threshold on validation set...")
                model.discriminator.evaluate_on_classification(
                    valid_data_with_labels,
                    optimizing_metric=CLASS_OPTIMIZING_METRIC,
                    is_threshold_tunning=True,
                )
                model.save_kbgan(pretrained_kbgan_path)
                print(f"Saved tuned threshold back to: {pretrained_kbgan_path}")
            else:
                print("Using classification threshold restored from checkpoint.")
            print(f"Classification threshold for Discriminator: {model.discriminator.classification_threshold}")
            print("----------------")

        # Test KBGAN on task
        if working_task == 'link-prediction' or working_task == 'all':
            link_prediction_metrics = model.evaluate_on_link_prediction(heads, tails, test_data_no_label,
                                                                        filt=RANK_FILT, k_list=RANK_K_LIST)
            _log_metrics("KBGAN link prediction", link_prediction_metrics)
            run_summary['test_metrics']['link_prediction'] = link_prediction_metrics

        if working_task == 'triple-classification' or working_task == 'all':
            triple_classification_metrics = model.evaluate_on_triple_classification(test_data_with_labels,
                                                                                    optimizing_metric=CLASS_OPTIMIZING_METRIC)
            _log_metrics("KBGAN triple classification", triple_classification_metrics)
            run_summary['test_metrics']['triple_classification'] = triple_classification_metrics
            t_step = log_step("KBGAN triple classification eval", t_step)
        print("----------------")
    else: 
        print("Invalid mode. Please try again and specify a mode: full-train / gan-train / test-only") 
    total_elapsed = time.perf_counter() - t_total
    print(f"[TIMER] Total runtime: {total_elapsed:.2f}s")
    run_summary['total_runtime_s'] = total_elapsed
    if run_summary['kbgan_training_time_s'] is None:
        run_summary['kbgan_training_time_s'] = 0.0
    _write_summary_report(run_summary)

if __name__ == '__main__':
    main()