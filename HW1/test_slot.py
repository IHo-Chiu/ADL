import csv
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    
    # TODO: create DataLoader for test dataset
    test_data = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        args.max_len,
    ).to(args.device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    
    # load weights into model
    model.load_state_dict(ckpt)

    # TODO: predict dataset        
    ans = {}
    if args.eval:
        ans2 = {}
    for data in test_data:
        with torch.no_grad():
            tokens = data['tokens'].to(args.device)
            lens = data['lens'].to(args.device)
            pred = model(tokens)
            pred = torch.argmax(pred, dim=1)
            
            for i, p in enumerate(pred):
                idxs = p[:lens[i]].cpu().detach().numpy()
                labels = [dataset.idx2label(idx) for idx in idxs]
                ans[data['id'][i]] = labels
                
            if args.eval:
                tags = data['tags']
                for i, p in enumerate(tags):
                    idxs = p[:lens[i]].cpu().detach().numpy()
                    labels = [dataset.idx2label(idx) for idx in idxs]
                    ans2[data['id'][i]] = labels
            
    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'tags'])
        for idx, label in ans.items():
            writer.writerow([idx, ' '.join(label)])
            
    # for evaluation
    if args.eval:
        y_true = []
        y_pred = []
        for key in ans:
            y_true.append(ans2[key])
            y_pred.append(ans[key])
            
        joint_correct_count = 0
        token_correct_count = 0
        joint_total = 0
        token_total = 0
        for a, b in zip(y_true, y_pred):
            joint_total += 1
            
            is_all_correct = True
            for x, y in zip(a, b):
                token_total += 1
                if x == y:
                    token_correct_count += 1
                else:
                    is_all_correct = False
                    
            if is_all_correct:
                joint_correct_count += 1
                
        print(f'joint acc: {joint_correct_count/joint_total}')
        print(f'token acc: {token_correct_count/token_total}')
        print(classification_report(y_true, y_pred, scheme=IOB2, mode='strict'))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=35)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    
    # evaluation
    parser.add_argument("--eval", type=bool, default=False)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)