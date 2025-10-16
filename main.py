import os
#we have imported the filesystem paths and folders
import shutil
#this above line helps in compying or removing directories
import argparse
#is used for the parsing of command line flags
from typing import Tuple, Iterable


import torch
#the above line is used for the pytorch library that helps in tensors , device ,autograph etc
from torch import nn
from torch.optim import Adam
#used for the optimization
from torch_geometric.datasets import TUDataset
#this is used for the loads for ENZYMES(from the TU graph repo)
from torch_geometric.loader import DataLoader
#minibatches of graphs
from torch_geometric.transforms import NormalizeFeatures

from models import GCNGraphClassifier
#imports the GCN classifier
#that is a deep learning model that extends convolution operations from images to graph-structured data, learning  node features by aggregating information  from their neighbors.
from utils import set_seed
#the above line helps in locating the

# Correct, consistent device name
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#for this operation we use the GPU if that is available and else we choose cpu


# Filenames typically present in the TU ENZYMES raw package
REQUIRED_PREFIX = "ENZYMES_"
RAW_FILENAMES: Iterable[str] = (
    "ENZYMES_A.txt",
    "ENZYMES_graph_indicator.txt",
    "ENZYMES_graph_labels.txt",
    "ENZYMES_node_labels.txt",
    "ENZYMES_node_attributes.txt",
    "ENZYMES_edge_labels.txt",
)
#Raw file handling thhat is so PyG that uses the local dataset that i have provided
def _copy_raw_if_found(src_dir: str, root: str, name: str = "ENZYMES") -> None:
    """
    If the user has ENZYMES_*.txt directly under `src_dir` (your case),
    copy them to the PyG-expected folder: {root}/{name}/raw/
    Example: src_dir="data/ENZYMES" -> copies into "data/ENZYMES/ENZYMES/raw/"
    """
    if not os.path.isdir(src_dir):
        return  #  if the folder does not exist nothing to do

    dataset_dir = os.path.join(root, name)         # e.g., data/ENZYMES/ENZYMES
    raw_dir = os.path.join(dataset_dir, "raw")     # e.g., data/ENZYMES/ENZYMES/raw
    os.makedirs(raw_dir, exist_ok=True)

    found_any = False
    for fname in os.listdir(src_dir):
        if fname.startswith(REQUIRED_PREFIX) and fname.endswith(".txt"):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(raw_dir, fname)
            shutil.copy2(src, dst)
            found_any = True

    # If we copied raw files, drop any old processed cache so PyG regenerates cleanly
    if found_any:
        processed_dir = os.path.join(dataset_dir, "processed")
        if os.path.isdir(processed_dir) and os.listdir(processed_dir):
            print("NOTE: Removing existing processed files to re-process from your local raw files...")
            shutil.rmtree(processed_dir)
    #if we found any files we change the rae files and there were old processed tensors that delete  them so Pyg rebuild clenly


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42) -> Tuple[torch.utils.data.Dataset, ...]:
    #we split the data set using this funtion
    """Random 80/10/10 split with fixed seed."""
    g = torch.Generator().manual_seed(seed)
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return torch.utils.data.random_split(dataset, [n_train, n_val, n_test], generator=g)
#the above line computes sizes and call the pyto+orch's random_split reproducibly

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    """Return average loss per graph and accuracy over a DataLoader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for data in loader:
        data = data.to(DEVICE)
        logits = model(data)
        loss = loss_fn(logits, data.y)
        total_loss += float(loss) * data.num_graphs
        pred = logits.argmax(dim=-1)
        correct += int((pred == data.y).sum())
        total += data.num_graphs
    return total_loss / total, correct / total

def train_one_epoch(model, loader, optimizer, loss_fn):
    """One training epoch. or called as train loop"""
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = loss_fn(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(loader.dataset)

# -------- ensure usable node features even if ENZYMES raw lacks float x --------
def ensure_node_features(ds) -> int:
    """
    Ensures every graph has float node features in data.x.
    Priority:
      1) If x exists and is float -> keep it.
      2) If x exists and is integer labels -> one-hot encode across the dataset.
      3) Else -> use constant ones (1-dim).
    Returns: num_features (int).
    """
    import torch as _torch

    d0 = ds[0]
    if getattr(d0, "x", None) is not None and d0.x.numel() > 0:
        if d0.x.dtype in (_torch.float32, _torch.float64):
            return d0.x.size(-1)

        # If x is integer labels, one-hot encode
        if d0.x.dtype in (_torch.int64, _torch.int32):
            max_label = 0
            for d in ds:
                max_label = max(max_label, int(d.x.max().item()))
            feat_dim = max_label + 1
            for d in ds:
                idx = d.x.view(-1).long()
                one_hot = _torch.zeros(idx.numel(), feat_dim, dtype=_torch.float32)
                one_hot[_torch.arange(idx.numel()), idx] = 1.0
                d.x = one_hot
            return feat_dim

    # No usable x -> constant ones
    for d in ds:
        N = d.num_nodes
        d.x = _torch.ones((N, 1), dtype=_torch.float32)
    return 1

def main():
    parser = argparse.ArgumentParser(description="GCN graph classification on ENZYMES (uses local raw files).")
    parser.add_argument("--root", type=str, default="data/ENZYMES",
                        help="Dataset root. Raw files will be placed under {root}/ENZYMES/raw/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)

    # 1) If ENZYMES_*.txt are directly under root (your screenshot), copy them to {root}/ENZYMES/raw/
    _copy_raw_if_found(src_dir=args.root, root=args.root, name="ENZYMES")

    # 2) Load dataset WITHOUT NormalizeFeatures first (so we can build features if needed)
    print("Processing...")
    dataset = TUDataset(root=args.root, name="ENZYMES", use_node_attr=True)
    print("Done!")

    # 3) Ensure node features exist (handle num_features == 0 cases), then normalize
    num_features = ensure_node_features(dataset)
    dataset.transform = NormalizeFeatures()

    num_classes = dataset.num_classes
    assert num_classes == 6, f"Expected 6 classes; got {num_classes}"

    # 4) Splits + loaders
    train_set, val_set, test_set = split_dataset(dataset, 0.8, 0.1, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    # 5) Model / loss / optimizer
    model = GCNGraphClassifier(num_features, args.hidden, args.layers, num_classes, dropout=args.dropout).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 6) Train with early stopping on val acc
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
              f"| val_acc={val_acc:.4f} | best_val_acc={best_val_acc:.4f}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best val_acc={best_val_acc:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 7) Final test
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print(f"TEST | loss={test_loss:.4f} | acc={test_acc:.4f}")

    # 8) Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/enzymes_gcn.pt")
    with open("artifacts/metrics.txt", "w") as f:
        f.write(f"val_acc={best_val_acc:.4f}\n")
        f.write(f"test_acc={test_acc:.4f}\n")
    print("Saved: artifacts/enzymes_gcn.pt and artifacts/metrics.txt")

if __name__ == "__main__":
    main()
