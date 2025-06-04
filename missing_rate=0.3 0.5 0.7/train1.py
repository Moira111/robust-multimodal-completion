import torch
import argparse
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from network1 import Network
from loss1 import Loss
from metric1 import valid
from dataloader1 import load_data
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Multi-modal Clustering')
parser.add_argument('--dataset', default='BDGP')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--feature_dim', type=int, default=512)
parser.add_argument('--high_feature_dim', type=int, default=128)
parser.add_argument('--temperature_f', type=float, default=0.5)
parser.add_argument('--temperature_l', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--mse_epochs', type=int, default=10)
parser.add_argument('--con_epochs', type=int, default=5)
parser.add_argument('--consistency_weight', type=float, default=0.2)
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)


def generate_incomplete_samples(dataset, view_num, missing_rate):
    new_dataset = []
    for sample in dataset:
        xs, y, idx = sample
        mask = np.random.rand(view_num) > missing_rate
        new_xs = []
        for v in range(view_num):
            if mask[v] or (v == 0 and not any(mask)):
                new_xs.append(xs[v])
            else:
                noise = torch.randn_like(xs[v]) * 0.05 + xs[v].mean()
                new_xs.append(noise)
        new_dataset.append((new_xs, y, idx, mask.astype(np.float32)))
    return new_dataset


def cross_view_consistency_loss(qs):
    total_loss = 0.0
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            total_loss += F.mse_loss(qs[i], qs[j])
    return total_loss / (len(qs) * (len(qs) - 1) / 2)


def feature_alignment_loss(model, dataloader, device, view):
    model.train()
    total_loss = 0.0
    for xs, _, _, _ in dataloader:
        xs = [x.to(device) for x in xs]
        hs, _, _, _ = model(xs)
        align_loss = 0.0
        for i in range(view):
            for j in range(i + 1, view):
                cos_sim = F.cosine_similarity(hs[i], hs[j], dim=1)
                align_loss += (1 - cos_sim).mean()
        align_loss /= (view * (view - 1) / 2)
        model.zero_grad()
        align_loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.data -= args.lr * p.grad
        total_loss += align_loss.item()
    return total_loss / len(dataloader)


def main():
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device).to(device)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    missing_stages = [0.1, 0.3, 0.5, 0.7]
    stage_results = []

    for stage_rate in missing_stages:
        print(f"\n=== Stage with Missing Rate: {stage_rate} ===")
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        train_dataset = generate_incomplete_samples(dataset, view, stage_rate)

        def collate_fn(batch):
            xs, ys, idxs, masks = zip(*batch)
            xs = [torch.stack([x[v] for x in xs]) for v in range(view)]
            masks = torch.tensor(np.stack(masks))
            return xs, torch.tensor(ys), torch.tensor(idxs), masks

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)


        model.train()
        for epoch in range(args.mse_epochs):
            total_loss = 0.0
            for xs, _, _, _ in train_loader:
                xs = [x.to(device) for x in xs]
                optimizer.zero_grad()
                _, _, xrs, _ = model(xs)
                loss = sum([F.mse_loss(x, xr) for x, xr in zip(xs, xrs)])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[MSE] Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")


        for epoch in range(args.con_epochs):
            model.train()
            total_loss = 0.0
            for xs, _, _, masks in train_loader:
                xs = [x.to(device) for x in xs]
                masks = masks.to(device)
                optimizer.zero_grad()
                hs, qs, _, _ = model(xs)
                confidence = masks.sum(dim=1) / view
                loss_main = criterion(hs, qs, confidence=confidence)
                loss_consist = cross_view_consistency_loss(qs)
                loss = loss_main + args.consistency_weight * loss_consist
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Contrastive+Consist] Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

        print("[Align] Feature Alignment...")
        align_loss = feature_alignment_loss(model, train_loader, device, view)
        print(f"[Align] Feature Align Loss: {align_loss:.4f}")


        acc, nmi, pur = valid(model, device, [(x, y, i) for (x, y, i, m) in train_dataset], view, data_size, class_num)
        stage_results.append((stage_rate, acc, nmi, pur))
        torch.cuda.synchronize()
        end_time = time.time()
        max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        print(f"\n[统计] Total training time: {end_time - start_time:.2f}s")
        print(f"[统计] Peak GPU memory usage: {max_mem:.2f} MB")

    print("\n=== Final Evaluation per Missing Rate Stage ===")
    for r in stage_results:
        print(f"[Final Eval @ Missing {r[0]:.1f}] ACC: {r[1]:.4f}, NMI: {r[2]:.4f}, PUR: {r[3]:.4f}")

if __name__ == "__main__":
    main()
