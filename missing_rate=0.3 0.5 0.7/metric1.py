from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur


def inference(loader, model, device, view, data_size):
    """
    :return:
    total_pred: prediction among all modalities
    pred_vectors: predictions of each modality, list
    labels_vector: true label
    Hs: high - level features
    Zs: low - level features
    """
    model.eval()
    soft_vector = []
    pred_vectors = []
    Hs = []
    Zs = []
    for v in range(view):
        pred_vectors.append([])
        Hs.append([])
        Zs.append([])
    labels_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            qs, preds = model.forward_cluster(xs)
            hs, _, _, zs = model.forward(xs)
            q = sum(qs) / view
        for v in range(view):
            hs[v] = hs[v].detach()
            zs[v] = zs[v].detach()
            preds[v] = preds[v].detach()
            pred_vectors[v].extend(preds[v].cpu().detach().numpy())
            Hs[v].extend(hs[v].cpu().detach().numpy())
            Zs[v].extend(zs[v].cpu().detach().numpy())
        q = q.detach()
        soft_vector.extend(q.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)
    total_pred = np.argmax(np.array(soft_vector), axis=1)
    for v in range(view):
        Hs[v] = np.array(Hs[v])
        Zs[v] = np.array(Zs[v])
        pred_vectors[v] = np.array(pred_vectors[v])
    return total_pred, pred_vectors, Hs, labels_vector, Zs


def valid(model, device, dataset, view, data_size, class_num, eval_h=False):
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
    # 这里假设inference函数已经被修改为返回与填补过程相关的结果
    total_pred, pred_vectors, high_level_vectors, labels_vector, low_level_vectors = inference(test_loader, model, device, view, data_size)
    if eval_h:
        print("Clustering results on low - level features of each view:")

        for v in range(view):
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            y_pred = kmeans.fit_predict(low_level_vectors[v])
            # 假设evaluate函数被修改为评估填补相关指标
            nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
            print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, acc,
                                                                                     v + 1, nmi,
                                                                                     v + 1, ari,
                                                                                     v + 1, pur))

        print("Clustering results on high - level features of each view:")

        for v in range(view):
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            y_pred = kmeans.fit_predict(high_level_vectors[v])
            nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
            print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, acc,
                                                                                     v + 1, nmi,
                                                                                     v + 1, ari,
                                                                                     v + 1, pur))
        print("Clustering results on cluster assignments of each view:")
        for v in range(view):
            nmi, ari, acc, pur = evaluate(labels_vector, pred_vectors[v])
            print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, acc,
                                                                                     v + 1, nmi,
                                                                                     v + 1, ari,
                                                                                     v + 1, pur))


    # 假设这里的labels_vector与填补过程相关
    print("Clustering results on semantic labels: " + str(labels_vector.shape))
    nmi, ari, acc, pur = evaluate(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
    return acc, nmi, pur


def interpolate_incomplete_samples(model, device, incomplete_dataset, view, complete_dataset, alpha=0.3, top_k=5):
    interpolated_dataset = []
    confidence_scores = []

    model.eval()
    scaler = MinMaxScaler()

    # 提取完整样本特征
    with torch.no_grad():
        complete_features = []
        complete_xs_list = []
        for complete_sample in complete_dataset:
            xs, y, idx = complete_sample
            feature_concat = []
            for v in range(view):
                x = xs[v].to(device).unsqueeze(0)
                z = model.encoders[v](x)
                h = F.normalize(model.feature_contrastive_module(z), dim=1).squeeze(0)
                feature_concat.append(h)
            feature_all = torch.cat(feature_concat).cpu()
            complete_features.append(feature_all)
            complete_xs_list.append(xs)

    complete_features = torch.stack(complete_features)

    # 对每个不完整样本进行插补
    for incomplete_sample in incomplete_dataset:
        xs, y, idx = incomplete_sample
        mask = [x is not None for x in xs]  # None 标识缺失
        missing_views = [v for v, present in enumerate(mask) if not present]

        # 提取当前样本特征
        feature_concat = []
        with torch.no_grad():
            for v in range(view):
                x = xs[v].to(device).unsqueeze(0)
                z = model.encoders[v](x)
                h = F.normalize(model.feature_contrastive_module(z), dim=1).squeeze(0)
                feature_concat.append(h)
        feature_current = torch.cat(feature_concat).cpu()

        # 相似度计算
        sims = F.cosine_similarity(feature_current.unsqueeze(0), complete_features).numpy()
        top_idx = np.argsort(sims)[-top_k:]
        weights = sims[top_idx]
        weights = scaler.fit_transform(weights.reshape(-1, 1)).reshape(-1)
        weights = weights / weights.sum()

        # 对每个缺失视图插值
        new_xs = []
        for v in range(view):
            if mask[v]:
                new_xs.append(xs[v])
            else:
                # 插补来自 top-k 加权平均
                candidates = [complete_xs_list[i][v] for i in top_idx]
                interpolated = sum(w * candidates[i] for i, w in enumerate(weights))
                new_xs.append(interpolated)

        interpolated_dataset.append((new_xs, y, idx))
        confidence_scores.append(weights.mean())  # 简单平均作为置信度

    return interpolated_dataset, confidence_scores