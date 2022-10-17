from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len

import torch


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        text_batch = [sample['text'].split(' ') for sample in samples]
        text_batch = self.vocab.encode_batch(text_batch, self.max_len)
        text_batch = torch.LongTensor(text_batch)
        id_batch = [sample['id'] for sample in samples]
        try:
            intent_batch = [self.label2idx(sample['intent']) for sample in samples]
            intent_batch = torch.LongTensor(intent_batch)
            batch = {'text': text_batch, 'intent': intent_batch, 'id': id_batch}
        except:
            batch = {'text': text_batch, 'id': id_batch}
            
        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples) -> Dict:
        # TODO: implement collate_fn
        len_batch = [len(sample['tokens']) for sample in samples]
        len_batch = torch.ByteTensor(len_batch)
        
        tokens_batch = [sample['tokens'] for sample in samples]
        tokens_batch = self.vocab.encode_batch(tokens_batch, self.max_len)
        tokens_batch = torch.LongTensor(tokens_batch)
        
        id_batch = [sample['id'] for sample in samples]
        
        try:
            tags_batch = [[self.label2idx(tag) for tag in sample['tags']] for sample in samples]
            tags_batch = pad_to_len(tags_batch, self.max_len, self.ignore_idx)
            tags_batch = torch.LongTensor(tags_batch)
            batch = {'lens': len_batch, 'tokens': tokens_batch, 'tags': tags_batch, 'id': id_batch}
        except:
            batch = {'lens': len_batch, 'tokens': tokens_batch, 'id': id_batch}

        return batch
        
