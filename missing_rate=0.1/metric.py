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
   
    total_pred, pred_vectors, high_level_vectors, labels_vector, low_level_vectors = inference(test_loader, model, device, view, data_size)
    if eval_h:
        print("Clustering results on low - level features of each view:")

        for v in range(view):
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            y_pred = kmeans.fit_predict(low_level_vectors[v])
           
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


   
    print("Clustering results on semantic labels: " + str(labels_vector.shape))
    nmi, ari, acc, pur = evaluate(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
    return acc, nmi, pur


def interpolate_incomplete_samples(model, device, incomplete_dataset, view, complete_dataset, alpha):
    interpolated_dataset = []
    scaler = MinMaxScaler()

    for incomplete_sample, complete_sample in zip(incomplete_dataset, complete_dataset):
        incomplete_xs, incomplete_y, _ = incomplete_sample
        complete_xs, complete_y, _ = complete_sample

      
        hs_missing = []
        hs_complete = []
        with torch.no_grad():
            for v in range(view):
               
                z_miss = model.encoders[v](incomplete_xs[v].to(device))
                z_comp = model.encoders[v](complete_xs[v].to(device))

               
                h_miss = F.normalize(model.feature_contrastive_module(z_miss), p=2, dim=-1)
                h_comp = F.normalize(model.feature_contrastive_module(z_comp).view(1, -1), p=2, dim=1)

             
                h_miss = h_miss + 0.1 * torch.randn_like(h_miss)
                h_comp = h_comp + 0.1 * torch.randn_like(h_comp)

                hs_missing.append(h_miss.cpu())
                hs_complete.append(h_comp.cpu())

       
        similarities = []
        for v in range(view):
            if torch.all(incomplete_xs[v] == 0): 
                
                euclidean_sim = 1 / (1 + torch.norm(hs_missing[v] - hs_complete[v], p=2))

               
                cosine_sim = F.cosine_similarity(hs_missing[v], hs_complete[v], dim=0)

               
                joint = torch.histc(hs_missing[v] * hs_complete[v], bins=10)
                marginal = torch.histc(hs_missing[v], bins=10) * torch.histc(hs_complete[v], bins=10)
                mi = torch.sum(joint * torch.log(joint / (marginal + 1e-10) + 1e-10))

               
                combined_sim = 0.4 * euclidean_sim + 0.4 * cosine_sim + 0.2 * mi
                similarities.append(combined_sim)

        
        if similarities:
            weights = F.softmax(torch.stack(similarities) * 10, dim=0)  
            aggregated = sum(w * c for w, c in zip(weights, complete_xs))

            
            gate = torch.sigmoid(aggregated)  
            residual = F.gelu(aggregated)  
            interpolated = (1 - alpha) * incomplete_xs[v] + alpha * (gate * aggregated + (1 - gate) * residual)
        else:
            interpolated = incomplete_xs[v]

        
        new_xs = []
        for v in range(view):
            if torch.all(incomplete_xs[v] == 0):
                
                new_xs.append(F.layer_norm(interpolated, interpolated.shape[-1:]))
            else:
                new_xs.append(incomplete_xs[v])

        interpolated_dataset.append((new_xs, incomplete_y, _))

    return interpolated_dataset
