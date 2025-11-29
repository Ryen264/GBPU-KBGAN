import os
import logging
import torch
import torch.nn.functional as nnf
from torch.autograd import Variable
from torch.optim import Adam, SGD, AdamW, RMSprop, Adagrad
import numpy as np
from typing import Generator, Tuple

from config import config, overwrite_config_with_args, logger_init
from datasets import batch_by_num, BernCorrupterMulti, BernCorrupter, sparse_heads_tails
from data_loader import index_entity_relation, graph_size, read_data
from models import TransE, TransD, DistMult, ComplEx

def acc_pre_rec_f1(predictions: list, true_labels: list) -> Tuple[float, float, float, float]:
    try:
        y_pred = np.asarray(predictions)
        y_true = np.asarray(true_labels)
    except Exception as e:
        print(f"Error converting inputs to numpy arrays: {e}")
        return None, None, None, None

    if y_pred.shape != y_true.shape:
        raise ValueError("Predictions and true labels must have the same shape.")

    # TP: y_pred == 1 AND y_true == 1
    TP = np.sum((y_pred == 1) & (y_true == 1))
    # TN: y_pred == 0 AND y_true == 0
    TN = np.sum((y_pred == 0) & (y_true == 0))
    # FP: y_pred == 1 AND y_true == 0
    FP = np.sum((y_pred == 1) & (y_true == 0))
    # FN: y_pred == 0 AND y_true == 1
    FN = np.sum((y_pred == 0) & (y_true == 1))

    # (TP + TN) / (TP + TN + FP + FN)
    total_samples = TP + TN + FP + FN
    accuracy = (TP + TN) / total_samples if total_samples > 0 else 0.0

    # TP / (TP + FP)
    precision_denominator = TP + FP
    precision = TP / precision_denominator if precision_denominator > 0 else 0.0

    # TP / (TP + FN)
    recall_denominator = TP + FN
    recall = TP / recall_denominator if recall_denominator > 0 else 0.0

    # 2 * (Precision * Recall) / (Precision + Recall)
    f1_denominator = precision + recall
    f1_score = 2 * (precision * recall) / f1_denominator if f1_denominator > 0 else 0.0
    return accuracy, precision, recall, f1_score

class Component():
    def __init__(self, role: str, model_type: str):
        """
        role = ["discriminator", "generator"]
        model_type = ["TransE", "TransD", "DistMult", "ComplEx"]
        """
        if role in ["discriminator", "generator"]:
            print(f"Initialized a new component with role {role}.")
            self.role = role
        else:
            print(f"Input role should be in list [\"discriminator\", \"generator\"]! Default role: \"discriminator\".")
            self.role = "discriminator"
        
        if model_type in ["TransE", "TransD", "DistMult", "ComplEx"]:
            print(f'Initialized component: {model_type} model.')
            self.model_type = model_type
        else:
            print(f"Input model type should be in list [\"TransE\", \"TransD\", \"DistMult\", \"ComplEx\"]! Default model type: \"TransE\".")
            self.model_type = "TransE"

        self.model_config = None
        self.model = None
        self.model_path = None
        self.n_entity = None
        self.n_relation = None

    def fit(self, n_entity: int, n_relation: int) -> None:
        self.n_entity = n_entity
        self.n_relation = n_relation

    def load(self, model_path: str=None) -> None:
        if (self.n_entity is None or self.n_relation is None):
            print(f"Component must be fitted before being loaded!")
            return
        
        self.model_config = config()[self.model_type]
        if self.model_type == 'TransE':
            self.model = TransE(self.n_entity, self.n_relation, self.model_config)
        elif self.model_type == 'TransD':
            self.model = TransD(self.n_entity, self.n_relation, self.model_config)
        elif self.model_type == 'DistMult':
            self.model = DistMult(self.n_entity, self.n_relation, self.model_config)
        elif self.model_type == 'ComplEx':
            self.model = ComplEx(self.n_entity, self.n_relation, self.model_config)

        output_dir = './output/' + config().task.dir + '/models'
        self.model_path = model_path if model_path is not None else os.path.join(output_dir, self.model_config.model_file)
    
        print(f"Loading component: {self.model_type} model.")
        self.model.load(self.model_path)
        print(f"Loaded component by path: {self.model_path}")

    def get_score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.model.get_score(head, relation, tail)

    def train(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data: tuple,
              use_early_stopping: bool=False, patience: int=10, optimizer_name: str='Adam') -> Tuple[float, str]:    
        if (self.n_entity is None or self.n_relation is None):
            print(f"Component must be fitted before being trained!")
            return None, None
        
        overwrite_config_with_args(["--pretrain_config=" + self.model_type])
        overwrite_config_with_args(["--log.prefix=" + self.model_type + '_'])
        logger_init()

        if self.model_type == 'TransE':
            corrupter = BernCorrupter(train_data, self.n_entity, self.n_relation)
            self.model = TransE(self.n_entity, self.n_relation, self.model_config)
        elif self.model_type == 'TransD':
            corrupter = BernCorrupter(train_data, self.n_entity, self.n_relation)
            self.model = TransD(self.n_entity, self.n_relation, self.model_config)
        elif self.model_type == 'DistMult':
            corrupter = BernCorrupterMulti(train_data, self.n_entity, self.n_relation, self.model_config.n_sample)
            self.model = DistMult(self.n_entity, self.n_relation, self.model_config)
        elif self.model_type == 'ComplEx':
            corrupter = BernCorrupterMulti(train_data, self.n_entity, self.n_relation, self.model_config.n_sample)
            self.model = ComplEx(self.n_entity, self.n_relation, self.model_config)    
        tester = lambda: self.model.evaluate(valid_data, self.n_entity, heads, tails)

        print(f'Training component: {self.model_type} model.')
        best_perf, model_path = self.model.train(train_data, corrupter, tester,
                                                use_early_stopping=use_early_stopping, patience=patience, optimizer_name=optimizer_name)
        print(f'Trained component: {self.model_type} model.')
        self.model_path = model_path
        return best_perf, model_path

    def step(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor, **kwargs):
        """
        Unified step function that handles both generator and discriminator logic.
        
        For generator:
            kwargs: n_sample=1, temperature=1.0, train=True
            Returns generator coroutine (yields samples, receives rewards)
            
        For discriminator:
            kwargs: head_fake, tail_fake, train=True
            Returns (losses, rewards)
        """
        if self.role == 'generator':
            return self._generator_step(head, relation, tail, **kwargs)
        elif self.role == 'discriminator':
            return self._discriminator_step(head, relation, tail, **kwargs)
        else:
            raise ValueError("Role must be either 'generator' or 'discriminator'")
    
    def _generator_step(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor,
                        n_sample: int=1, temperature: float=1.0, train: bool=True) -> Generator[torch.Tensor, torch.Tensor, None]:
        """Generator step: sample fake triples and update with REINFORCE"""
        # Forward pass: generate samples
        n, m = tail.size()
        relation_var = Variable(relation.cuda())
        head_var = Variable(head.cuda())
        tail_var = Variable(tail.cuda())

        logits = self.model.model.prob_logit(head_var, relation_var, tail_var) / temperature
        probs = nnf.softmax(logits, dim=-1)
        row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
        sample_idx = torch.multinomial(probs, n_sample, replacement=True)
        sample_heads = head[row_idx, sample_idx.data.cpu()]
        sample_tails = tail[row_idx, sample_idx.data.cpu()]
        
        # Yield samples to get rewards from discriminator
        rewards = yield sample_heads, sample_tails
        
        # Backward pass: update generator with REINFORCE
        if train:            
            self.model._ensure_optimizer()
            self.model.model.zero_grad()
            log_probs = nnf.log_softmax(logits, dim=-1)
            reinforce_loss = -torch.sum(Variable(rewards) * log_probs[row_idx.cuda(), sample_idx.data])
            reinforce_loss.backward()
            self.model.opt.step()
            self.model.model.constraint()
        yield None

    def _discriminator_step(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor,
                            head_fake: torch.Tensor=None, tail_fake: torch.Tensor=None, train: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discriminator step: distinguish real from fake triples"""
        if head_fake is None or tail_fake is None:
            raise ValueError("head_fake and tail_fake must be provided for discriminator step")
        
        # Forward pass: compute losses and scores
        head_var = Variable(head.cuda())
        relation_var = Variable(relation.cuda())
        tail_var = Variable(tail.cuda())
        head_fake_var = Variable(head_fake.cuda())
        tail_fake_var = Variable(tail_fake.cuda())
        
        losses = self.model.model.pair_loss(head_var, relation_var, tail_var, head_fake_var, tail_fake_var)
        fake_scores = self.model.model.score(head_fake_var, relation_var, tail_fake_var)
                
        # Backward pass: update discriminator
        if train:
            self.model._ensure_optimizer()
            self.model.model.zero_grad()
            torch.sum(losses).backward()
            self.model.opt.step()
            self.model.model.constraint()
        return losses.data, -fake_scores.data
        
    def evaluate(self, test_data: tuple, heads: list, tails: list) -> dict:
        """
        Evaluate the model on Link Prediction task.
        """
        if (self.n_entity is None or self.n_relation is None):
            print(f"Component must be fitted before being tested!")
            return None
        if (self.model_path is None):
            print(f"Component must be trained before being tested!")
            return None
        
        print(f"Testing component: {self.model_type} model.")
        metrics = self.model.evaluate(test_data, self.n_entity, heads, tails)
        return metrics
        
    def find_optimal_threshold(self, valid_data: tuple, labels: list, n_thresholds: int=100) -> float:
        """
        Find the optimal threshold for triple classification using validation data.

        Args:
            valid_data: Tuple of (heads, relations, tails)
            labels: Ground truth labels for validation data
            n_thresholds: Number of threshold values to try

        Returns:
            Optimal threshold value that maximizes F1 score
        """
        heads, relations, tails = valid_data

        # Compute scores for all validation samples
        scores_list = []
        with torch.no_grad():
            for i in range(len(heads)):
                head = torch.LongTensor([heads[i]]).cuda()
                relation = torch.LongTensor([relations[i]]).cuda()
                tail = torch.LongTensor([tails[i]]).cuda()

                score = self.get_score(head, relation, tail)
                scores_list.append(score)

        # Try different threshold values
        min_score = min(scores_list)
        max_score = max(scores_list)
        threshold_values = np.linspace(min_score, max_score, n_thresholds)

        best_f1 = 0.0
        best_threshold = 0.0

        # Determine if model is distance-based or similarity-based
        is_distance_based = self.model_type in ['TransE', 'TransD']

        for threshold in threshold_values:
            predictions = []
            for score in scores_list:
                if is_distance_based:
                    predictions.append(1 if score < threshold else 0)
                else:
                    predictions.append(1 if score > threshold else 0)
            
            _, _, _, f1 = acc_pre_rec_f1(predictions, labels)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        logging.info(f"Optimal threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
        return best_threshold

class KBGAN():
    def __init__(self, discriminator_type="TransE", generator_type="DistMult"):
        """
        discriminator_type = ["TransE", "TransD"]
        generator_type = ["DistMult", "ComplEx"]
        """
        self.discriminator_type = discriminator_type
        self.discriminator = Component(role="discriminator", model_type=discriminator_type)

        self.generator_type = generator_type
        self.generator = Component(role="generator", model_type=generator_type)

        self.discriminator_path = None
        self.generator_path = None
        self.kbgan_path = None 
        self.n_entity = None
        self.n_relation = None
    
    def fit(self, n_entity: int, n_relation: int) -> None:
        self.n_entity = n_entity
        self.n_relation = n_relation

    def load_component(self, component_role: str, component_path: str=None) -> None:
        """
        component_role = ["discriminator", "generator"]
        """
        if (self.n_entity is None or self.n_relation is None):
            print(f"Component must be fitted before being loaded!")
            return
        
        if component_role == "discriminator":
            print(f"Loading discriminator: {self.discriminator_type} model.")
            self.discriminator.load_model(component_path)
            print(f"Loaded discriminator by path: {component_path}")
        elif component_role == "generator":
            print(f"Loading generator: {self.generator_type} model.")
            self.generator.load_model(component_path)
            print(f"Loaded generator by path: {component_path}")

    def pretrain_component(self, component_role: str, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data: tuple,
                           use_early_stopping: bool=False, patience: int=10, optimizer_name: str='Adam') -> Tuple[float, str]:
        """
        component_role = ["discriminator", "generator"]
        """
        if (self.n_entity is None or self.n_relation is None):
            print(f"Component must be fitted before being pretrained!")
            return None, None
        
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data[0], torch.Tensor):
            valid_data = [torch.LongTensor(vec) for vec in valid_data]
        
        best_perf = 0.0
        path = None
        if component_role == "discriminator":
            print(f"Pretraining discriminator: {self.discriminator_type} model.")
            best_perf, path = self.discriminator.train(heads, tails, train_data, valid_data,
                                                    use_early_stopping=use_early_stopping, patience=patience, optimizer_name=optimizer_name)
            self.discriminator_path = path
            print(f"Pretrained discriminator saved in {path}.")
        elif component_role == "generator":
            print(f"Pretraining generator: {self.generator_type} model.")
            best_perf, path = self.generator.train(heads, tails, train_data, valid_data,
                                                    use_early_stopping=use_early_stopping, patience=patience, optimizer_name=optimizer_name)
            self.generator_path = path
            print(f"Pretrained generator saved in {path}.")
        return best_perf, path

    def evaluate_component(self, component_role: str, heads, tails, test_data) -> dict:
        """
        component_role = ["discriminator", "generator"]
        """
        if (self.n_entity is None or self.n_relation is None):
            print(f"Component must be fitted before being tested!")
            return None
        
        if not isinstance(test_data[0], torch.Tensor):
            test_data = [torch.LongTensor(vec) for vec in test_data]

        metrics = None
        if component_role == "discriminator":
            if (self.discriminator_path is None):
                print(f"Component must be pretrained before being tested!")
                return None
            
            print(f"Testing component: {self.discriminator_type} model.")
            metrics = self.discriminator.evaluate(test_data, self.n_entity, heads, tails)
        elif component_role == "generator":
            if (self.generator_path is None):
                print(f"Component must be pretrained before being tested!")
                return None
            print(f"Testing component: {self.generator_type} model.")
            metrics = self.generator.evaluate(test_data, self.n_entity, heads, tails)
        return metrics

    def load(self, kbgan_path: str=None) -> None:
        if (self.n_entity is None or self.n_relation is None):
            print(f"Model must be fitted before being loaded!")
            return
        
        print(f"Loading KBGAN: {self.discriminator_type} discriminator, {self.generator_type} generator.")
        self.discriminator.load_model(kbgan_path)
        print(f"Loaded KBGAN discriminator by path: {kbgan_path}")

    def pretrain(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data: tuple,
                use_early_stopping: bool=False, patience: int=10, optimizer_name: str='Adam') -> Tuple[float, str, float, str]:
        if (self.n_entity is None or self.n_relation is None):
            print(f"Model must be fitted before being pretrained!")
            return None, None, None, None
        
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data[0], torch.Tensor):
            valid_data = [torch.LongTensor(vec) for vec in valid_data]
        
        overwrite_config_with_args(["--log.prefix=" + self.discriminator_type + '-' + self.generator_type + "_"])
        logger_init()

        print(f"Pretraining KBGAN: {self.generator_type} generator, {self.discriminator_type} discriminator.")
        best_perf_d, path_d = self.pretrain_component("discriminator", heads, tails, train_data, valid_data,
                                                      use_early_stopping=use_early_stopping, patience=patience, optimizer_name=optimizer_name)
        best_perf_g, path_g = self.pretrain_component("generator", heads, tails, train_data, valid_data,
                                                      use_early_stopping=use_early_stopping, patience=patience, optimizer_name=optimizer_name)
        print(f"Pretrained KBGAN: {self.generator_type} generator, {self.discriminator_type} discriminator.")

        self.kbgan_path = os.path.join('./output/', config().task.dir, 'kbgan', f'kbgan_dis-{self.discriminator_type}_gen-{self.generator_type}.mdl')
        os.makedirs(os.path.dirname(self.kbgan_path), exist_ok=True)
        logging.info(f"KBGAN model saved to {self.kbgan_path}")
        return best_perf_d, path_d, best_perf_g, path_g
           
    def train(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data: tuple,
                use_early_stopping: bool=False, patience: int=10, optimizer_name: str = 'Adam') -> Tuple[float, str]:
        if (self.n_entity is None or self.n_relation is None):
            print(f"Model must be fitted before being trained!")
            return None
        
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data[0], torch.Tensor):
            valid_data = [torch.LongTensor(vec) for vec in valid_data]
        
        overwrite_config_with_args(["--log.prefix=" + self.discriminator_type + '-' + self.generator_type + "_"])
        logger_init()

        generator_config = config()[config().g_config]
        discriminator_config = config()[config().d_config]

        models = {'TransE': TransE, 'TransD': TransD, 'DistMult': DistMult, 'ComplEx': ComplEx}
        # Load pretrained models into the Component instances
        model_dir = './output/' + config().task.dir + '/models'
        
        # Initialize generator model if not already done
        if self.generator.model is None:
            self.generator.model_config = generator_config
            self.generator.model = models[config().g_config](self.n_entity, self.n_relation, generator_config)
        self.generator.load(os.path.join(model_dir, generator_config.model_file))
        
        # Initialize discriminator model if not already done
        if self.discriminator.model is None:
            self.discriminator.model_config = discriminator_config
            self.discriminator.model = models[config().d_config](self.n_entity, self.n_relation, discriminator_config)
        self.discriminator.load(os.path.join(model_dir, discriminator_config.model_file))

        # Initialize optimizers according to optimizer_name for both models
        opt_map = {
            'Adam': Adam,
            'SGD': SGD,
            'AdamW': AdamW,
            'RMSprop': RMSprop,
            'Adagrad': Adagrad,
        }
        opt_cls = opt_map.get(optimizer_name, Adam)
        try:
            self.generator.model.opt = opt_cls(self.generator.model.parameters())
            self.discriminator.model.opt = opt_cls(self.discriminator.model.parameters())
        except Exception:
            pass

        corrupter = BernCorrupterMulti(train_data, self.n_entity, self.n_relation, config().KBGAN.n_sample)
        head, relation, tail = train_data
        n_train = len(head)
        n_epoch = config().KBGAN.n_epoch
        n_batch = config().KBGAN.n_batch

        model_name = 'kbgan_' + 'dis-' + self.discriminator_type + '_gen-' + self.generator_type + '.mdl'
        best_perf = 0
        avg_reward = 0
        patience_counter = 0

        print(f'Training KBGAN: {self.generator_type} generator, {self.discriminator_type} discriminator.')
        for epoch in range(n_epoch):
            epoch_d_loss = 0
            epoch_reward = 0

            head_cand, relation_cand, tail_cand = corrupter.corrupt(head, relation, tail, keep_truth=False)
            for h, r, t, hs, rs, ts in batch_by_num(n_batch, head, relation, tail, head_cand, relation_cand, tail_cand, n_sample=n_train):
                gen_step = self.generator.step(hs, rs, ts, temperature=config().KBGAN.temperature)
                head_smpl, tail_smpl = next(gen_step)
                
                losses, rewards = self.discriminator.step(h, r, t, head_fake=head_smpl.squeeze(), tail_fake=tail_smpl.squeeze())
                epoch_reward += torch.sum(rewards)

                rewards = rewards - avg_reward
                # Update generator with rewards
                try:
                    gen_step.send(rewards.unsqueeze(1))
                except StopIteration:
                    pass
                epoch_d_loss += torch.sum(losses)
                
            avg_loss = epoch_d_loss / n_train
            avg_reward = epoch_reward / n_train

            logging.info('Epoch %d/%d, D_loss=%f, reward=%f', epoch + 1, n_epoch, avg_loss, avg_reward)
            save_dir = './output/' + config().task.dir + '/kbgan/'
            os.makedirs(save_dir, exist_ok=True)
            kbgan_path = os.path.join(save_dir, model_name)

            if (epoch + 1) % config().KBGAN.epoch_per_test == 0:
                metrics = self.discriminator.model.evaluate(valid_data, self.n_entity, heads, tails, filt=True)
                perf = metrics['MRR']
                if perf > best_perf:
                    best_perf = perf
                    patience_counter = 0
                    self.discriminator.model.save(kbgan_path)
                else:
                    patience_counter += 1
                    
                if use_early_stopping and patience_counter >= patience:
                    logging.info('Early stopping triggered at epoch %d (patience=%d)', epoch + 1, patience)
                    break
        print(f'Trained KBGAN: {self.generator_type} generator, {self.discriminator_type} discriminator.')
        self.kbgan_path = kbgan_path
        return best_perf, kbgan_path

    def evaluate_on_link_prediction(self, heads: torch.Tensor, tails: torch.Tensor, test_data: tuple) -> dict:
        if (self.n_entity is None or self.n_relation is None):
            print(f"Model must be fitted before being tested!")
            return None
        if (self.kbgan_path is None):
            print(f"Model must be trained before being tested!")
            return None
        
        if not isinstance(test_data[0], torch.Tensor):
            test_data = [torch.LongTensor(vec) for vec in test_data]

        print("Evaluating KBGAN discriminator on Link Prediction...")
        metrics = self.discriminator.model.evaluate(test_data, self.n_entity, heads, tails, filt=True)
        return metrics

    def evaluate_on_triple_classification(self, test_data_with_labels: tuple, valid_data_with_labels: tuple = None,
                                          threshold: float = None, auto_threshold: bool = True) -> Tuple[dict, float, list, list]:
        if (self.n_entity is None or self.n_relation is None):
            print(f"Model must be fitted before being tested!")
            return None
        if (self.kbgan_path is None):
            print(f"Model must be trained before being tested!")
            return None

        heads_test, relations_test, tails_test, labels = test_data_with_labels
        print("Evaluating KBGAN discriminator on Triple Classification...")

        scores_list = []
        with torch.no_grad():
            for i in range(len(heads_test)):
                head = torch.LongTensor([heads_test[i]]).cuda()
                relation = torch.LongTensor([relations_test[i]]).cuda()
                tail = torch.LongTensor([tails_test[i]]).cuda()

                score = self.discriminator.get_score(head, relation, tail)
                scores_list.append(score)

        # Determine threshold
        if threshold is None and auto_threshold and valid_data_with_labels is not None:
            # expect valid_data_with_labels to be (heads, relations, tails, labels)
            try:
                valid_data = valid_data_with_labels[:3]
                if not isinstance(valid_data[0], torch.Tensor):
                    valid_data = [torch.LongTensor(vec) for vec in valid_data]

                valid_labels = valid_data_with_labels[3]
                threshold = self.discriminator.find_optimal_threshold(valid_data, valid_labels, n_thresholds=100)
                logging.info(f"Auto-computed threshold: {threshold:.4f}")
            except Exception:
                logging.info("Could not auto-compute threshold from provided validation data; using default 0.0")
                threshold = 0.0
        elif threshold is None:
            threshold = 0.0
            logging.info(f"Using default threshold: {threshold:.4f}")

        is_distance_based = self.discriminator_type in ['TransE', 'TransD']
        predictions = []
        for score in scores_list:
            if is_distance_based:
                predictions.append(1 if score < threshold else 0)
            else:
                predictions.append(1 if score > threshold else 0)

        accuracy, precision, recall, f1 = acc_pre_rec_f1(predictions, list(labels) if labels is not None else [] )

        metrics = {}
        metrics['Accuracy'] = accuracy
        metrics['Precision'] = precision
        metrics['Recall'] = recall
        metrics['F1'] = f1
        metrics_str = f"Accuracy = {accuracy:.4f}\n"
        metrics_str += f"Precision = {precision:.4f}\n"
        metrics_str += f"Recall = {recall:.4f}\n"
        metrics_str += f"F1 Score = {f1:.4f}\n"        
        logging.info(metrics_str)
        return metrics, threshold, predictions, scores_list