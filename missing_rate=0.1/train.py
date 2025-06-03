import torch
from network import Network
from metric import valid, interpolate_incomplete_samples
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def generate_incomplete_samples(dataset, missing_modality_rate=0.1):
    new_dataset = []
    for sample in dataset:
        xs, y, _ = sample
        new_xs = []
        for v in range(len(xs)):
            if np.random.rand() < missing_modality_rate:
                # 创建一个与原张量形状相同的全零张量来替代None
                if isinstance(xs[v], torch.Tensor):
                    # 添加随机噪声以增强数据
                    zero_tensor = torch.randn_like(xs[v]) * 0.01
                    new_xs.append(zero_tensor)
                else:
                    # 如果不是张量类型，可以根据实际情况进行处理，这里假设是numpy数组并创建全零数组
                    if isinstance(xs[v], np.ndarray):
                        zero_array = np.random.randn(*xs[v].shape) * 0.01
                        new_xs.append(zero_array)
            else:
                new_xs.append(xs[v])
        new_dataset.append((new_xs, y, _))
    return new_dataset
# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
Dataname = 'BDGP'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
if args.dataset == "MNIST-USPS":
    args.con_epochs = 50
    seed = 10
if args.dataset == "BDGP":
    args.con_epochs = 10
    seed = 10
if args.dataset == "CCV":
    args.con_epochs = 50
    seed = 3
if args.dataset == "Fashion":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Caltech-2V":
    args.con_epochs = 50
    seed = 10
if args.dataset == "Caltech-3V":
    args.con_epochs = 50
    seed = 10
if args.dataset == "Caltech-4V":
    args.con_epochs = 50
    seed = 10
if args.dataset == "Caltech-5V":
    args.con_epochs = 50
    seed = 5


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(seed)


dataset, dims, view, data_size, class_num = load_data(args.dataset)

# 生成不完整模态样本
incomplete_dataset = generate_incomplete_samples(dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def contrastive_train(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(criterion.forward_label(qs[v], qs[w]))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))


def make_pseudo_label(model, device):
    loader = torch.utils.data.DataLoader(
        incomplete_dataset,
        batch_size=data_size,
        shuffle=False,
    )
    model.eval()
    scaler = MinMaxScaler()
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            if xs[v] is not None:
                xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, _, _, _ = model.forward(xs)
        for v in range(view):
            if hs[v] is not None:
                hs[v] = hs[v].cpu().detach().numpy()
                hs[v] = scaler.fit_transform(hs[v])

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    new_pseudo_label = []
    for v in range(view):
        if hs[v] is not None:
            Pseudo_label = kmeans.fit_predict(hs[v])
            Pseudo_label = Pseudo_label.reshape(data_size, 1)
            Pseudo_label = torch.from_numpy(Pseudo_label)
            new_pseudo_label.append(Pseudo_label)
        else:
            new_pseudo_label.append(None)
    return new_pseudo_label


def match(y_true, y_pred):
    y_true = np.array(y_true).astype(np.int64).flatten()
    y_pred = np.array(y_pred).astype(np.int64).flatten()
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = dict(zip(col_ind, row_ind))
    new_y = np.array([mapping[y] for y in y_true])
    return torch.from_numpy(new_y).long().to(device)



def fine_tuning(epoch, new_pseudo_label):
    loader = torch.utils.data.DataLoader(
        incomplete_dataset,
        batch_size=data_size,
        shuffle=False,
    )
    tot_loss = 0.
    cross_entropy = torch.nn.CrossEntropyLoss()
    for batch_idx, (xs, _, idx) in enumerate(loader):
        for v in range(view):
            if xs[v] is not None:
                xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, qs, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            if new_pseudo_label[v] is not None:
                p = new_pseudo_label[v].numpy().T
                with torch.no_grad():
                    q = qs[v].detach().cpu()
                    q = torch.argmax(q, dim=1).numpy()
                    p_hat = match(p, q)
                loss_list.append(cross_entropy(qs[v], p_hat))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1




for i in range(T):
    print("ROUND:{}".format(i + 1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    epoch = 1
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
    while epoch <= args.mse_epochs + args.con_epochs:
        contrastive_train(epoch)
        if epoch == args.mse_epochs + args.con_epochs:
            acc, nmi, pur = valid(model, device, incomplete_dataset, view, data_size, class_num, eval_h=False)
        epoch += 1
    new_pseudo_label = make_pseudo_label(model, device)

    # 新增步骤：对不完整样本进行插值处理
    complete_dataset = data_loader.dataset
    interpolated_dataset = interpolate_incomplete_samples(model, device, incomplete_dataset, view, complete_dataset,
                                                           alpha=0.1)

    while epoch <= args.mse_epochs + args.con_epochs + args.tune_epochs:
        fine_tuning(epoch, new_pseudo_label)
        if epoch == args.mse_epochs + args.con_epochs + args.tune_epochs:
            # 使用插值后的数据集进行验证
            acc, nmi, pur = valid(model, device, interpolated_dataset, view, data_size, class_num, eval_h=False)
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
            print('Saving..')
            accs.append(acc)
            nmis.append(nmi)
            purs.append(pur)
        epoch += 1