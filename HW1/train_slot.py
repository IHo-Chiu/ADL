import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
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
    model = SeqTagger(embeddings,
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
            tokens = data['tokens'].to(args.device)
            tags = data['tags'].to(args.device)
            optimizer.zero_grad()
            pred = model(tokens)
            loss = criterion(pred, tags)
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
                tokens = data['tokens'].to(args.device)
                tags = data['tags'].to(args.device)
                lens = data['lens'].to(args.device)
                pred = model(tokens)
                pred = torch.argmax(pred, dim=1)
                total_count += len(pred)
                for i, p in enumerate(pred):
                    idx = p[:lens[i]]
                    tag = tags[i][:lens[i]]
                    correct_count += torch.all(idx.eq(tag)).sum().item() 
        
        acc = correct_count / total_count
        epoch_pbar.set_description(f'loss: {running_loss}, acc: {round(acc,3)}, best: {round(best_acc,3)}')
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.ckpt_dir / "model.pt")
    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=35)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

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