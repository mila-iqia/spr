import torch
from torch import nn
from .utils import EarlyStopping, appendabledict, calculate_multiclass_accuracy, calculate_multiclass_f1_score
from copy import deepcopy
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from atariari.benchmark.categorization import summary_key_dict
from src.memory import blank_trans
import torch.functional as F

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=255):
        super().__init__()
        self.model = nn.Linear(in_features=input_dim, out_features=num_classes)

    def forward(self, feature_vectors):
        return self.model(feature_vectors)


class FullySupervisedLinearProbe(nn.Module):
    def __init__(self, encoder, num_classes=255):
        super().__init__()
        self.encoder = deepcopy(encoder)
        self.probe = LinearProbe(input_dim=self.encoder.hidden_size,
                                 num_classes=num_classes)

    def forward(self, x):
        feature_vec = self.encoder(x)
        return self.probe(feature_vec)


class ProbeTrainer():
    def __init__(self,
                 encoder=None,
                 forward=None,
                 method_name="my_method",
                 wandb=None,
                 patience=15,
                 num_classes=256,
                 fully_supervised=False,
                 save_dir=".models",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 lr=5e-4,
                 epochs=100,
                 batch_size=64,
                 representation_len=256,):

        self.encoder = encoder
        self.forward = forward
        self.wandb = wandb
        self.device = device
        self.fully_supervised = fully_supervised
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.method = method_name
        self.device = device
        self.feature_size = representation_len
        self.loss_fn = nn.CrossEntropyLoss()

        # bad convention, but these get set in "create_probes"
        self.probes = self.early_stoppers = self.optimizers = self.schedulers = None

    def create_probes(self, sample_label):
        if self.fully_supervised:
            assert self.encoder != None, "for fully supervised you must provide an encoder!"
            self.probes = {k: FullySupervisedLinearProbe(encoder=self.encoder,
                                                         num_classes=self.num_classes).to(self.device) for k in
                           sample_label.keys()}
        else:
            self.probes = {k: LinearProbe(input_dim=self.feature_size,
                                          num_classes=self.num_classes).to(self.device) for k in sample_label.keys()}

        self.early_stoppers = {
            k: EarlyStopping(patience=self.patience,
                             verbose=False,
                             name=k + "_probe",
                             wandb=self.wandb)
            for k in sample_label.keys()}

        self.optimizers = {k: torch.optim.Adam(list(self.probes[k].parameters()),
                                               eps=1e-5, lr=self.lr) for k in sample_label.keys()}
        self.schedulers = {
            k: torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizers[k], patience=5, factor=0.2, verbose=True,
                                                          mode='max', min_lr=1e-5) for k in sample_label.keys()}

    def generate_multistep_batch(self, transitions, labels, n=30):
        total_steps = len(transitions)
        print('Total Steps: {}'.format(len(transitions)))
        for idx in range(total_steps // self.batch_size):
            indices = np.random.randint(0, total_steps, size=self.batch_size)
            gap=n
            t1 = indices - gap
            underflow = np.clip(t1, a_max=0, a_min=None)
            indices -= underflow
            t1 -= underflow
            all_states = []
            all_actions = []
            all_labels = [appendabledict() for _ in range(n)]
            for t1, t2 in zip(t1, indices):
                # Get one sample from this episode
                while transitions[t2].timestep - gap < 0:
                    t2 = np.random.randint(0, total_steps)
                    t1 = t2 - gap
                    # don't allow negative indices.
                    underflow = np.clip(t1, a_max=0, a_min=None)
                    t1 -= underflow
                    t2 -= underflow

                for i, l_t in enumerate(labels[t1:t2]):
                    for k, v in l_t.items():
                        all_labels[i][k].append(v)
                all_states.append(torch.stack([t.state for t in transitions[t1:t2]], 0))
                all_actions.append([t.action for t in transitions[t1:t2]])

            all_states = torch.stack(all_states).to(self.device).float()/255.

            yield all_states,\
                  torch.tensor(all_actions, device=self.device).long(),\
                  all_labels,

    def run_multistep(self, transitions, labels, n=30, train=False):
        sample_label = labels[0]
        epoch_loss, accuracy = {k + "_loss": np.zeros(n) for k in sample_label.keys()},\
                               {k + "_acc": np.zeros(n) for k in sample_label.keys()}
        iterator = self.generate_multistep_batch(transitions, labels, n+3)
        steps = 0
        for step, (x, actions, labels_batch) in enumerate(iterator):
            steps += 1
            initial_shape = x.shape
            initial_x = x[:, :4].reshape(-1, *initial_shape[2:])
            stack = self.encoder(initial_x)
            representations = [stack.view(initial_shape[0], 4, -1)[:, -1]]

            current_stack = stack.view(initial_shape[0], -1)
            f_t_current = representations[0]
            for i in range(n-1):
                a_i = actions[:, i]
                f_t_current = self.forward(current_stack, a_i) + f_t_current

                current_stack = torch.cat([current_stack[:, self.encoder.hidden_size:],
                                           f_t_current], -1)
                representations.append(f_t_current)

            for jump, (step_representation, step_labels) in \
                    enumerate(zip(representations, labels_batch[3:])):
                for k, label in step_labels.items():
                    if self.early_stoppers[k].early_stop:
                        continue
                    optim = self.optimizers[k]
                    optim.zero_grad()

                    label = torch.tensor(label).long().to(self.device)
                    preds = self.probe(None, k, vector=step_representation)
                    loss = self.loss_fn(preds, label)

                    epoch_loss[k + "_loss"][jump] += (loss.detach().item())
                    accuracy[k + "_acc"][jump] += (calculate_multiclass_accuracy(preds, label))
                    if self.probes[k].training and train:
                        loss.backward()
                        optim.step()

        epoch_loss = {k: loss/steps for k, loss in epoch_loss.items()}
        accuracy = {k: acc/steps for k, acc in accuracy.items()}

        return epoch_loss, accuracy


    def generate_batch(self, transitions, labels):
        total_steps = len(transitions)
        print('Total Steps: {}'.format(len(transitions)))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        for idx in range(total_steps // self.batch_size):
            indices = np.random.randint(0, total_steps, size=self.batch_size)
            x = []
            y = appendabledict()
            for t1 in indices:

                x.append(transitions[t1].state)
                for k, v in labels[t1].items():
                    y[k].append(v)
            yield torch.stack(x).to(self.device).float() / 255., y

    def probe(self, batch, k, vector=None):
        probe = self.probes[k]
        probe.to(self.device)
        if vector is not None:
            preds = probe(vector.detach())
        elif self.fully_supervised:
            # if method is supervised batch is a batch of frames and probe is a full encoder + linear or nonlinear probe
            preds = probe(batch)

        elif not self.encoder:
            # if encoder is None then inputs are vectors
            f = batch.detach()
            assert len(f.squeeze().shape) == 2, "if input is a batch of vectors you must specify an encoder!"
            preds = probe(f)

        else:
            with torch.no_grad():
                self.encoder.to(self.device)
                f = self.encoder(batch).detach()
            preds = probe(f)
        return preds

    def do_one_epoch(self, episodes, label_dicts):
        sample_label = label_dicts[0]
        epoch_loss, accuracy = {k + "_loss": [] for k in sample_label.keys() if
                                not self.early_stoppers[k].early_stop}, \
                               {k + "_acc": [] for k in sample_label.keys() if
                                not self.early_stoppers[k].early_stop}

        data_generator = self.generate_batch(episodes, label_dicts)
        for step, (x, labels_batch) in enumerate(data_generator):
            for k, label in labels_batch.items():
                if self.early_stoppers[k].early_stop:
                    continue
                optim = self.optimizers[k]
                optim.zero_grad()

                label = torch.tensor(label).long().to(self.device)
                preds = self.probe(x, k)
                loss = self.loss_fn(preds, label)

                epoch_loss[k + "_loss"].append(loss.detach().item())
                accuracy[k + "_acc"].append(calculate_multiclass_accuracy(preds, label))
                if self.probes[k].training:
                    loss.backward()
                    optim.step()

        epoch_loss = {k: np.mean(loss) for k, loss in epoch_loss.items()}
        accuracy = {k: np.mean(acc) for k, acc in accuracy.items()}

        return epoch_loss, accuracy

    def do_test_epoch(self, episodes, label_dicts):
        sample_label = label_dicts[0]
        accuracy_dict, f1_score_dict = {}, {}
        pred_dict, all_label_dict = {k: [] for k in sample_label.keys()}, \
                                    {k: [] for k in sample_label.keys()}

        # collect all predictions first
        data_generator = self.generate_batch(episodes, label_dicts)
        for step, (x, labels_batch) in enumerate(data_generator):
            for k, label in labels_batch.items():
                label = torch.tensor(label).long().cpu()
                all_label_dict[k].append(label)
                preds = self.probe(x, k).detach().cpu()
                pred_dict[k].append(preds)

        for k in all_label_dict.keys():
            preds, labels = torch.cat(pred_dict[k]), torch.cat(all_label_dict[k])

            accuracy = calculate_multiclass_accuracy(preds, labels)
            f1score = calculate_multiclass_f1_score(preds, labels)
            accuracy_dict[k] = accuracy
            f1_score_dict[k] = f1score

        return accuracy_dict, f1_score_dict

    def train(self, tr_eps, val_eps, tr_labels, val_labels):
        sample_label = tr_labels[0]
        self.create_probes(sample_label)
        e = 0
        all_probes_stopped = np.all([early_stopper.early_stop for early_stopper in self.early_stoppers.values()])
        while (not all_probes_stopped) and e < self.epochs:
            epoch_loss, accuracy = self.do_one_epoch(tr_eps, tr_labels)
            self.log_results(e, epoch_loss, accuracy)

            val_loss, val_accuracy = self.evaluate(val_eps, val_labels, epoch=e)
            # update all early stoppers
            for k in sample_label.keys():
                if not self.early_stoppers[k].early_stop:
                    self.early_stoppers[k](val_accuracy["val_" + k + "_acc"], self.probes[k])

            for k, scheduler in self.schedulers.items():
                if not self.early_stoppers[k].early_stop:
                    scheduler.step(val_accuracy['val_' + k + '_acc'])
            e += 1
            all_probes_stopped = np.all([early_stopper.early_stop for early_stopper in self.early_stoppers.values()])
        print("All probes early stopped!")

    def evaluate(self, val_episodes, val_label_dicts, epoch=None):
        for k, probe in self.probes.items():
            probe.eval()
        epoch_loss, accuracy = self.do_one_epoch(val_episodes, val_label_dicts)
        epoch_loss = {"val_" + k: v for k, v in epoch_loss.items()}
        accuracy = {"val_" + k: v for k, v in accuracy.items()}
        self.log_results(epoch, epoch_loss, accuracy)
        for k, probe in self.probes.items():
            probe.train()
        return epoch_loss, accuracy

    def test(self, test_episodes, test_label_dicts, epoch=None):
        for k in self.early_stoppers.keys():
            self.early_stoppers[k].early_stop = False
        for k, probe in self.probes.items():
            probe.eval()
        acc_dict, f1_dict = self.do_test_epoch(test_episodes, test_label_dicts)

        acc_dict, f1_dict = postprocess_raw_metrics(acc_dict, f1_dict)

        print("""In our paper, we report F1 scores and accuracies averaged across each category. 
              That is, we take a mean across all state variables in a category to get the average score for that category.
              Then we average all the category averages to get the final score that we report per game for each method. 
              These scores are called \'across_categories_avg_acc\' and \'across_categories_avg_f1\' respectively
              We do this to prevent categories with large number of state variables dominating the mean F1 score.
              """)
        self.log_results("Test", acc_dict, f1_dict)
        return acc_dict, f1_dict

    def log_results(self, epoch_idx, *dictionaries):
        print("Epoch: {}".format(epoch_idx))
        for dictionary in dictionaries:
            for k, v in dictionary.items():
                print("\t {}: {:8.4f}".format(k, v))
            print("\t --")


def postprocess_raw_metrics(acc_dict, f1_dict):
    acc_overall_avg, f1_overall_avg = compute_dict_average(acc_dict), \
                                      compute_dict_average(f1_dict)
    acc_category_avgs_dict, f1_category_avgs_dict = compute_category_avgs(acc_dict), \
                                                    compute_category_avgs(f1_dict)
    acc_avg_across_categories, f1_avg_across_categories = compute_dict_average(acc_category_avgs_dict), \
                                                          compute_dict_average(f1_category_avgs_dict)
    acc_dict.update(acc_category_avgs_dict)
    f1_dict.update(f1_category_avgs_dict)

    acc_dict["overall_avg"], f1_dict["overall_avg"] = acc_overall_avg, f1_overall_avg
    acc_dict["across_categories_avg"], f1_dict["across_categories_avg"] = [acc_avg_across_categories,
                                                                           f1_avg_across_categories]

    acc_dict = append_suffix(acc_dict, "_acc")
    f1_dict = append_suffix(f1_dict, "_f1")

    return acc_dict, f1_dict


def compute_dict_average(metric_dict):
    return np.mean(list(metric_dict.values()))


def compute_category_avgs(metric_dict):
    category_dict = {}
    for category_name, category_keys in summary_key_dict.items():
        category_values = [v for k, v in metric_dict.items() if k in category_keys]
        if len(category_values) < 1:
            continue
        category_mean = np.mean(category_values)
        category_dict[category_name + "_avg"] = category_mean
    return category_dict


def append_suffix(dictionary, suffix):
    new_dict = {}
    for k, v in dictionary.items():
        new_dict[k + suffix] = v
    return new_dict
