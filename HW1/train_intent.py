import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    
    # TODO: crecate DataLoader for train / dev datasets
    train_data = torch.utils.data.DataLoader(
        datasets[TRAIN], 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=datasets[TRAIN].collate_fn)
    eval_data = torch.utils.data.DataLoader(
        datasets[DEV], 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        datasets[TRAIN].num_classes,
        args.max_len,
    ).to(args.device)

    # TODO: init optimizer
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        running_loss = 0.0
        for data in train_data:
            text = data['text'].to(args.device)
            intent = data['intent'].to(args.device)
            optimizer.zero_grad()
            pred = model(text)
            loss = criterion(pred, intent)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        running_loss /= len(train_data)
        
        # TODO: Evaluation loop - calculate accuracy and save model weights
        correct_count = 0
        total_count = 0
        model.eval()
        for data in eval_data:
            with torch.no_grad():
                text = data['text'].to(args.device)
                intent = data['intent'].to(args.device)
                pred = model(text)
                pred = torch.argmax(pred, dim=1)
                total_count += len(pred)
                correct_count += (pred == intent).sum().item() 
        
        acc = correct_count / total_count
        epoch_pbar.set_description(f'loss: {running_loss}, acc: {round(acc,3)}, best: {round(best_acc,3)}')
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.ckpt_dir / "model.pt")

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=32)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=384)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
