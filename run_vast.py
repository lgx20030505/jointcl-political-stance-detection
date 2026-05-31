import argparse
from tqdm import tqdm
import faiss
import os
import torch
from torch import optim
import random
import numpy as np
from utils.criterion import TraditionCriterion, Stance_loss
from torch.utils.data import RandomSampler, DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from utils.data_utils import Tokenizer4Bert, ZeroshotDataset
from transformers import BertModel
from models.bert_scl_prototype_graph import BERT_SCL_Proto_Graph
import pickle
from time import strftime, localtime

# Optional GPU setup
gpu_id = 0
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    except Exception:
        pass


class Instructor(object):
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert_proto = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(opt, bert_proto).to(opt.device)

        print("using model:", opt.model_name)
        print("running dataset:", opt.dataset)
        print("output_dir:", opt.output_dir)
        print("device:", opt.device)

        train_file_name = './vast_train1.dat'
        dev_file_name = './vast_dev1.dat'
        test_file_name = './vast_test1.dat'

        try:
            self.trainset = pickle.load(open(train_file_name, 'rb'))
            self.valset = pickle.load(open(dev_file_name, 'rb'))
            self.testset = pickle.load(open(test_file_name, 'rb'))
            print("Loaded cached dataset pickles.")
        except Exception:
            self.trainset = ZeroshotDataset(
                data_dir=self.opt.train_dir,
                tokenizer=tokenizer,
                opt=self.opt,
                data_type='train'
            )
            self.valset = ZeroshotDataset(
                data_dir=self.opt.dev_dir,
                tokenizer=tokenizer,
                opt=self.opt,
                data_type='dev'
            )
            self.testset = ZeroshotDataset(
                data_dir=self.opt.test_dir,
                tokenizer=tokenizer,
                opt=self.opt,
                data_type='test'
            )
            pickle.dump(self.trainset, open(train_file_name, 'wb'))
            pickle.dump(self.valset, open(dev_file_name, 'wb'))
            pickle.dump(self.testset, open(test_file_name, 'wb'))
            print("Built and cached dataset pickles.")

        if 'scl' in self.opt.model_name:
            self.stance_criterion = Stance_loss(opt.temperature).to(opt.device)
            self.target_criterion = Stance_loss(opt.temperature).to(opt.device)
            self.logits_criterion = TraditionCriterion(opt)
            params = [p for p in self.model.parameters()] + [p for p in self.target_criterion.parameters()]
        else:
            self.criterion = TraditionCriterion(opt)
            params = [p for p in self.model.parameters()]

        self.optimizer = self.opt.optim_class(params, lr=self.opt.lr)
        self.cluster_result = None

    def save_txt_result(self, tag, acc, f1, y_true, y_pred):
        report_path = os.path.join(self.opt.output_dir, f"{tag}_result.txt")
        report = classification_report(y_true, y_pred, digits=6, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Tag: {tag}\n")
            f.write(f"Accuracy: {acc}\n")
            f.write(f"Macro F1: {f1}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm))
            f.write("\n")

        print(f"Saved TXT result to: {report_path}")

    def run_tradition(self):
        best_acc, best_f1 = self.train_traditon()

        state_dict_dir = os.path.join(self.opt.output_dir, "state_dict")

        final_acc, final_f1 = 0, 0

        print(f"\nReload the best model with best acc {best_acc} from path {state_dict_dir}\n")
        ckpt_acc = os.path.join(state_dict_dir, "best_acc_model.bin")
        if os.path.exists(ckpt_acc):
            ckpt = torch.load(ckpt_acc, map_location=self.opt.device)
            self.model.load_state_dict(ckpt)
            acc, f1, y_true, y_pred = self.test_tradition()
            self.save_txt_result("best_acc", acc, f1, y_true, y_pred)
            final_acc, final_f1 = acc, f1
        else:
            print("best_acc_model.bin not found.")

        print(f"\nReload the best model with best f1 {best_f1} from path {state_dict_dir}\n")
        ckpt_f1 = os.path.join(state_dict_dir, "best_f1_model.bin")
        if os.path.exists(ckpt_f1):
            ckpt = torch.load(ckpt_f1, map_location=self.opt.device)
            self.model.load_state_dict(ckpt)
            acc, f1, y_true, y_pred = self.test_tradition()
            self.save_txt_result("best_f1", acc, f1, y_true, y_pred)
            final_acc, final_f1 = acc, f1
        else:
            print("best_f1_model.bin not found.")

        summary_path = os.path.join(self.opt.output_dir, "final_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Final Acc: {final_acc}\n")
            f.write(f"Final F1: {final_f1}\n")
            f.write(f"Output dir: {self.opt.output_dir}\n")
            f.write(f"Train file: {self.opt.train_dir}\n")
            f.write(f"Dev file: {self.opt.dev_dir}\n")
            f.write(f"Test file: {self.opt.test_dir}\n")
        print(f"Saved summary TXT to: {summary_path}")

        return final_acc, final_f1

    def compute_features(self, train_loader):
        print('Computing features...')
        self.model.eval()
        features = torch.zeros(len(train_loader.dataset), self.opt.bert_dim).to(self.opt.device)

        for batch in tqdm(train_loader):
            input_features = [
                torch.as_tensor(batch[feat_name], dtype=torch.long, device=self.opt.device)
                for feat_name in self.opt.input_features
            ]
            index = batch['index']
            with torch.no_grad():
                feature = self.model.prototype_encode(input_features)
                feature = feature.squeeze(dim=1)
                features[index] = feature

        return features.detach().cpu()

    def run_kmeans(self, x):
        print('performing kmeans clustering')
        results = {'im2cluster': [], 'centroids': [], 'density': []}

        for seed, num_cluster in enumerate(self.opt.num_cluster):
            d = x.shape[1]
            k = int(num_cluster)

            clus = faiss.Clustering(d, k)
            clus.verbose = True
            clus.niter = 20
            clus.nredo = 5
            clus.seed = seed
            clus.max_points_per_centroid = 1000
            clus.min_points_per_centroid = 1

            index = faiss.IndexFlatL2(d)
            clus.train(x, index)

            D, I = index.search(x, 1)
            im2cluster = [int(n[0]) for n in I]
            centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

            Dcluster = [[] for _ in range(k)]
            for im, i in enumerate(im2cluster):
                Dcluster[i].append(D[im][0])

            density = np.zeros(k)
            for i, dist in enumerate(Dcluster):
                if len(dist) > 1:
                    density[i] = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)

            dmax = density.max() if density.max() > 0 else 1.0
            for i, dist in enumerate(Dcluster):
                if len(dist) <= 1:
                    density[i] = dmax

            density = density.clip(np.percentile(density, 10), np.percentile(density, 90))
            density = self.opt.temperature * density / density.mean()

            centroids = torch.tensor(centroids, device=self.opt.device, dtype=torch.float)
            centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
            im2cluster = torch.tensor(im2cluster, device=self.opt.device, dtype=torch.long)
            density = torch.tensor(density, device=self.opt.device, dtype=torch.float)

            results['centroids'].append(centroids)
            results['density'].append(density)
            results['im2cluster'].append(im2cluster)

        return results

    def run_prototype(self, train_loader):
        self.opt.warmup_epoch = 0
        self.opt.num_cluster = [3]

        features = self.compute_features(train_loader)

        cluster_result = {'im2cluster': [], 'centroids': [], 'density': []}
        for num_cluster in self.opt.num_cluster:
            cluster_result['im2cluster'].append(
                torch.zeros(len(train_loader.dataset), dtype=torch.long, device=self.opt.device)
            )
            cluster_result['centroids'].append(
                torch.zeros(int(num_cluster), self.opt.bert_dim, device=self.opt.device)
            )
            cluster_result['density'].append(
                torch.zeros(int(num_cluster), device=self.opt.device)
            )

        features = features.numpy()
        cluster_result = self.run_kmeans(features)
        return cluster_result

    def train_traditon(self):
        sampler = RandomSampler(self.trainset)
        train_loader = DataLoader(self.trainset, batch_size=self.opt.batch_size, sampler=sampler)
        train_loader_prototype = DataLoader(self.trainset, batch_size=self.opt.batch_size, sampler=sampler)

        print("Train loader length: {}".format(len(train_loader)))

        optimizer = self.optimizer
        best_acc = 0
        best_f1 = 0
        cnt = 0

        for i_epoch in range(self.opt.epochs):
            print('>' * 20, f'epoch:{i_epoch}', '<' * 20)

            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()

            for i_batch, batch in enumerate(train_loader):
                cluster_every = max(1, int(len(train_loader) / self.opt.cluster_times))
                if i_batch % cluster_every == 0:
                    cluster_result = self.run_prototype(train_loader_prototype)

                input_features = [
                    torch.as_tensor(batch[feat_name], dtype=torch.long, device=self.opt.device)
                    for feat_name in self.opt.input_features
                ]
                true_stance = torch.as_tensor(batch['polarity'], dtype=torch.long, device=self.opt.device)

                if 'scl' in self.opt.model_name:
                    true_targets = torch.as_tensor(batch['topic_index'], dtype=torch.long, device=self.opt.device)
                    s_t_list = [str(i.item() + j.item()) for i, j in zip(true_stance, true_targets)]
                    s_t_list_drop = list(set(s_t_list))
                    polarity2label = {polarity: idx for idx, polarity in enumerate(s_t_list_drop)}
                    s_t = torch.tensor([polarity2label[i] for i in s_t_list], device=self.opt.device)

                    feature = self.model.prototype_encode(input_features)
                    logits, node_for_con = self.model(input_features + [cluster_result['centroids']])
                    self.cluster_result = [cluster_result['centroids']]

                    if cluster_result is not None:
                        prototype_loss = self.target_criterion(node_for_con, s_t)
                        stance_loss = self.stance_criterion(feature, true_stance)
                    else:
                        prototype_loss = torch.tensor(0.0, device=self.opt.device)
                        stance_loss = torch.tensor(0.0, device=self.opt.device)

                    logits_loss = self.logits_criterion(logits, true_stance)
                    loss = (
                        logits_loss
                        + stance_loss * self.opt.stance_loss_weight
                        + prototype_loss * self.opt.prototype_loss_weight
                    )
                else:
                    logits = self.model(input_features)
                    loss = self.criterion(logits, true_stance)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                n_correct += (torch.argmax(logits, -1) == true_stance).sum().item()
                n_total += len(logits)
                loss_total += loss.item() * len(logits)

                if cnt % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    if 'scl' in self.opt.model_name:
                        print(
                            "Train step: {} acc:{:.5f} total_loss:{:.5f} loss:{:.5f}, stance loss:{:.5f}, prototype loss:{:.5f}".format(
                                cnt,
                                train_acc,
                                train_loss,
                                float(loss.detach().cpu()),
                                float(stance_loss.detach().cpu()),
                                float(prototype_loss.detach().cpu())
                            )
                        )
                    else:
                        print("Train step: {} acc:{} loss: {}".format(cnt, train_acc, train_loss))

                if cnt != 0 and cnt % self.opt.eval_steps == 0 and i_epoch > 0:
                    eval_acc, eval_f1 = self.dev_tradition()

                    if eval_acc > best_acc:
                        print('Better ACC! Saving model!')
                        best_acc = eval_acc
                        state_dict_dir = os.path.join(opt.output_dir, "state_dict")
                        os.makedirs(state_dict_dir, exist_ok=True)
                        torch.save(self.model.state_dict(), os.path.join(state_dict_dir, "best_acc_model.bin"))

                    if eval_f1 > best_f1:
                        print('Better F1! Saving model!')
                        best_f1 = eval_f1
                        state_dict_dir = os.path.join(opt.output_dir, "state_dict")
                        os.makedirs(state_dict_dir, exist_ok=True)
                        torch.save(self.model.state_dict(), os.path.join(state_dict_dir, "best_f1_model.bin"))

                cnt += 1

        print("Training finished.")
        return best_acc, best_f1

    def dev_tradition(self):
        self.model.eval()
        sampler = RandomSampler(self.valset)
        dev_loader = DataLoader(dataset=self.valset, batch_size=self.opt.eval_batch_size, sampler=sampler)

        all_labels = []
        all_logits = []

        for batch in dev_loader:
            input_features = [
                torch.as_tensor(batch[feat_name], dtype=torch.long, device=self.opt.device)
                for feat_name in self.opt.input_features
            ]
            true_stance = torch.as_tensor(batch['polarity'], dtype=torch.long, device=self.opt.device)

            with torch.no_grad():
                if 'scl' in self.opt.model_name:
                    if self.cluster_result is None:
                        raise RuntimeError("cluster_result is None during validation.")
                    logits, _ = self.model(input_features + self.cluster_result)
                else:
                    logits = self.model(input_features)

            labels = true_stance.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            all_labels.append(labels)
            all_logits.append(logits)

        all_labels = np.concatenate(all_labels, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        preds = all_logits.argmax(axis=1)

        acc = accuracy_score(y_true=all_labels, y_pred=preds)
        f1 = f1_score(all_labels, preds, average='macro')
        self.model.train()
        return acc, f1

    def test_tradition(self):
        self.model.eval()
        sampler = RandomSampler(self.testset)
        test_loader = DataLoader(dataset=self.testset, batch_size=self.opt.eval_batch_size, sampler=sampler)

        all_labels = []
        all_logits = []

        for batch in test_loader:
            input_features = [
                torch.as_tensor(batch[feat_name], dtype=torch.long, device=self.opt.device)
                for feat_name in self.opt.input_features
            ]
            true_stance = torch.as_tensor(batch['polarity'], dtype=torch.long, device=self.opt.device)

            with torch.no_grad():
                if 'scl' in self.opt.model_name:
                    if self.cluster_result is None:
                        raise RuntimeError("cluster_result is None during test.")
                    logits, _ = self.model(input_features + self.cluster_result)
                else:
                    logits = self.model(input_features)

            labels = true_stance.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            all_labels.append(labels)
            all_logits.append(logits)

        all_labels = np.concatenate(all_labels, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        preds = all_logits.argmax(axis=1)

        acc = accuracy_score(y_true=all_labels, y_pred=preds)
        f1 = f1_score(all_labels, preds, average='macro')
        print(classification_report(all_labels, preds, digits=6, zero_division=0))
        print("Test Acc: {} F1:{}".format(acc, f1))
        self.model.train()
        return acc, f1, all_labels, preds


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='bert-scl-prototype-graph', type=str, required=False)
    parser.add_argument('--type', default=2, type=int, required=False)
    parser.add_argument('--dataset', default='custom', type=str, required=False)
    parser.add_argument('--output_par_dir',default='./outputs',type=str)
    parser.add_argument('--polarities', default=["left", "right", "neutral"], nargs='+', required=False)
    parser.add_argument('--optimizer', default='adam', type=str, required=False)
    parser.add_argument('--temperature', default=0.07, type=float, required=False)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, required=False)
    parser.add_argument('--lr', default=5e-6, type=float, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, required=False)
    parser.add_argument('--l2reg', default=0.001, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False)
    parser.add_argument('--log_path', default="./log", type=str, required=False)
    parser.add_argument('--embed_dim', default=300, type=int, required=False)
    parser.add_argument('--hidden_dim', default=128, type=int, required=False)
    parser.add_argument('--feature_dim', default=2 * 128, type=int, required=False)
    parser.add_argument('--output_dim', default=64, type=int, required=False)
    parser.add_argument('--relation_dim', default=100, type=int, required=False)
    parser.add_argument('--bert_dim', default=768, type=int, required=False)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str, required=False)
    parser.add_argument('--max_seq_len', default=200, type=int, required=False)
    parser.add_argument('--train_dir', default='./custom_jointcl_data/train.csv', type=str, required=False)
    parser.add_argument('--dev_dir', default='./custom_jointcl_data/dev.csv', type=str, required=False)
    parser.add_argument('--test_dir', default='./custom_jointcl_data/test.csv', type=str, required=False)
    parser.add_argument('--stance_loss_weight', default=1, type=float, required=False)
    parser.add_argument('--prototype_loss_weight', default=0.01, type=float, required=False)
    parser.add_argument('--alpha', default=0.8, type=float, required=False)
    parser.add_argument('--beta', default=1.2, type=float, required=False)
    parser.add_argument('--device', default=None, type=str, required=False)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--batch_size", default=16, type=int, required=False)
    parser.add_argument("--eval_batch_size", default=16, type=int, required=False)
    parser.add_argument("--epochs", default=15, type=int, required=False)
    parser.add_argument("--eval_steps", default=50, type=int, required=False)
    parser.add_argument("--cluster_times", default=5, type=int, required=False)
    parser.add_argument('--gnn_dims', default='192,192', type=str, required=False)
    parser.add_argument('--att_heads', default='4,4', type=str, required=False)
    parser.add_argument('--dp', default=0.1, type=float)

    opt = parser.parse_args()

    # OPTION 2: built-in project settings
    opt.type = 2
    opt.dataset = 'custom'
    opt.output_par_dir = r'/mnt/c/Users/16248/Desktop/大学课程/541 Practical Code'
    opt.polarities = ["left", "right", "neutral"]

    opt.train_dir = './custom_jointcl_data/train.csv'
    opt.dev_dir = './custom_jointcl_data/dev.csv'
    opt.test_dir = './custom_jointcl_data/test.csv'
    opt.epochs = 5
    opt.batch_size = 2
    opt.eval_batch_size = 4
    opt.eval_steps = 10
    opt.cluster_times = 1
    opt.lr = 5e-6

    if opt.seed is not None:
        set_seed(opt.seed)

    model_classes = {
        'bert-scl-prototype-graph': BERT_SCL_Proto_Graph,
    }
    input_features = {
        'bert-scl-prototype-graph': ['concat_bert_indices', 'concat_segments_indices'],
    }
    optimizers = {
        'adam': optim.Adam,
    }

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.n_gpus = torch.cuda.device_count()
    opt.model_class = model_classes[opt.model_name]
    opt.optim_class = optimizers[opt.optimizer]
    opt.input_features = input_features[opt.model_name]
    opt.output_dir = os.path.join(
        opt.output_par_dir,
        opt.model_name,
        opt.dataset,
        strftime("%Y-%m-%d %H-%M-%S", localtime())
    )

    opt.num_labels = len(opt.polarities)
    os.makedirs(opt.output_dir, exist_ok=True)

    writer = SummaryWriter(opt.log_path)
    print(opt)

    ins = Instructor(opt)
    acc, f1 = ins.run_tradition()

    print('#' * 20, 'result : Acc {}, F1 {}'.format(acc, f1))
    writer.close()
