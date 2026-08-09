import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import AddedToken
import torch
import json


class SolvDataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        smiles_col: str = 'smiles',
        label_cols: list = None,
        is_pretrain: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_pretrain = is_pretrain
        
        # Read the input data.
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.tsv'):
            self.data = pd.read_csv(data_path, sep='\t')
        else:
            raise ValueError(f" unsupported file format : {data_path}")
        
        if smiles_col not in self.data.columns:
            raise ValueError(f" column '{smiles_col}' does not exist on data file in ")
        
        self.smiles_col = smiles_col
        
        if label_cols is None:
            label_cols = ['Ex1', 'Ex2', 'Ex3', 'Rx1', 'Rx2', 'Rx3']
        
        if not is_pretrain:
            missing_cols = [col for col in label_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f" Below label column does not exist on data file in : {missing_cols}")
        
        self.label_cols = label_cols
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles_combination = str(row[self.smiles_col])
        
        # Tokenize
        encoded = self.tokenizer(
            smiles_combination,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }
        
        if not self.is_pretrain:
            labels = []
            for col in self.label_cols:
                val = row[col]
                if pd.isna(val):
                    labels.append(0.0)
                else:
                    labels.append(float(val))
            result['labels'] = torch.tensor(labels, dtype=torch.float32)
        
        return result


def create_simple_tokenizer(vocab_size: int = 1000):
    common_chars = [
        '[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]',
        'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
        'c', 'n', 'o', 's',
        '(', ')', '[', ']', '{', '}',
        '=', '#', '-', '+', '.',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '@', '/', '\\', '%',
    ]
    
    vocab = {}
    for i, char in enumerate(common_chars):
        vocab[char] = i
    
    all_chars = set(''.join(common_chars))
    for i in range(32, 127):
        char = chr(i)
        if char not in vocab and len(vocab) < vocab_size:
            vocab[char] = len(vocab)
    
    ids_to_tokens = {v: k for k, v in vocab.items()}
    
    class SimpleTokenizer(PreTrainedTokenizer):
        def __init__(self, vocab_dict, ids_to_tokens_dict, **kwargs):
            super().__init__(**kwargs)
            self.vocab = vocab_dict
            self.ids_to_tokens = ids_to_tokens_dict
            self._pad_token = '[PAD]'
            self._unk_token = '[UNK]'
            self._cls_token = '[CLS]'
            self._sep_token = '[SEP]'
            self._mask_token = '[MASK]'
            
            self.pad_token = self._pad_token
            self.unk_token = self._unk_token
            self.cls_token = self._cls_token
            self.sep_token = self._sep_token
            self.mask_token = self._mask_token
            
            self.pad_token_id = self.vocab.get(self._pad_token, 0)
            self.unk_token_id = self.vocab.get(self._unk_token, 1)
            self.cls_token_id = self.vocab.get(self._cls_token, 2)
            self.sep_token_id = self.vocab.get(self._sep_token, 3)
            self.mask_token_id = self.vocab.get(self._mask_token, 4)
        
        @property
        def vocab_size(self):
            return len(self.vocab)
        
        def _tokenize(self, text):
            return list(text)
        
        def _convert_token_to_id(self, token):
            return self.vocab.get(token, self.unk_token_id)
        
        def _convert_id_to_token(self, index):
            return self.ids_to_tokens.get(index, self._unk_token)
        
        def convert_tokens_to_string(self, tokens):
            return ''.join(tokens)
        
        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            if token_ids_1 is None:
                return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]
        
        def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
            if already_has_special_tokens:
                return super().get_special_tokens_mask(
                    token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
                )
            
            if token_ids_1 is not None:
                return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
            return [1] + ([0] * len(token_ids_0)) + [1]
        
        def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
            sep = [self.sep_token_id]
            cls = [self.cls_token_id]
            if token_ids_1 is None:
                return len(cls + token_ids_0 + sep) * [0]
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
        
        def save_pretrained(self, save_directory, **kwargs):
            os.makedirs(save_directory, exist_ok=True)
            
            # Save the generated artifacts.
            vocab_file = os.path.join(save_directory, 'vocab.txt')
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                    f.write(f"{token}\n")
            
            # Save the generated artifacts.
            tokenizer_config = {
                'tokenizer_class': 'SimpleTokenizer',
                'vocab_size': len(self.vocab),
                'model_max_length': 512,
                'padding_side': 'right',
                'truncation_side': 'right',
                'do_lower_case': False,
                'pad_token': self._pad_token,
                'unk_token': self._unk_token,
                'cls_token': self._cls_token,
                'sep_token': self._sep_token,
                'mask_token': self._mask_token,
            }
            
            config_file = os.path.join(save_directory, 'tokenizer_config.json')
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    
    tokenizer = SimpleTokenizer(vocab, ids_to_tokens)
    return tokenizer


def create_smiles_combination(solvent_smiles: str, solute_smiles: str) -> str:
    return f"{solvent_smiles}.{solute_smiles}"


def build_tokenizer(
    vocab_path: Optional[str] = None,
    model_name: str = "bert-base-uncased",
    train_data_path: Optional[str] = None,
    vocab_size: int = 1000,
    local_files_only: bool = False
):
    # Load the input data.
    if vocab_path and os.path.exists(vocab_path):
        try:
            print(f" from Local path load tokenizer: {vocab_path}")
            tokenizer = AutoTokenizer.from_pretrained(vocab_path, local_files_only=True)
            return tokenizer
        except Exception as e:
            print(f"Warning: unable to load tokenizer from {vocab_path}: {e}")
    
    # Load the input data.
    # Save the generated artifacts.
    possible_paths = [
        './checkpoints',
        './experiments',
        os.path.join(os.path.dirname(__file__), '..', 'checkpoints'),
    ]
    
    for base_path in possible_paths:
        if os.path.exists(base_path):
            tokenizer_config_path = os.path.join(base_path, 'tokenizer_config.json')
            vocab_file_path = os.path.join(base_path, 'vocab.txt')
            if os.path.exists(tokenizer_config_path) or os.path.exists(vocab_file_path):
                try:
                    print(f" from {base_path} load saved tokenizer")
                    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
                    return tokenizer
                except Exception as e:
                    print(f"Warning: unable to load tokenizer from {base_path}: {e}")
                    continue
    
    try:
        if local_files_only:
            print(f" attempt from Local Cache load tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        else:
            print(f" from HuggingFace load tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        if local_files_only:
            # Load the input data.
            print(f"Warning: tokenizer '{model_name}' is unavailable in offline mode.")
            print(f"Error: {e}")
            print(" create Easy Character Level tokenizer as a fallback ...")
            tokenizer = create_simple_tokenizer(vocab_size=vocab_size)
            print("Created the character-level fallback tokenizer.")
            return tokenizer
        else:
            print(f"Online download failed; trying the local cache: {e}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                print("Loaded tokenizer from the local cache.")
            except Exception as e2:
                print("Warning: pretrained tokenizer unavailable; using the character-level fallback.")
                print(f"Online download failed: {e}")
                print(f" local cache is also unavailable : {e2}")
                tokenizer = create_simple_tokenizer(vocab_size=vocab_size)
                print("Created the character-level fallback tokenizer.")
    
    if train_data_path:
        # Read the input data.
        if train_data_path.endswith('.csv'):
            df = pd.read_csv(train_data_path)
        elif train_data_path.endswith('.tsv'):
            df = pd.read_csv(train_data_path, sep='\t')
        else:
            raise ValueError(f" unsupported file format : {train_data_path}")
        
        smiles_col = 'smiles' if 'smiles' in df.columns else df.columns[0]
        all_smiles = df[smiles_col].astype(str).tolist()
        
        print(f" use pretraining tokenizer: {model_name}")
    
    return tokenizer


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }
    
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
        result['labels'] = labels
    
    return result


def mask_tokens_for_mlm(
    input_ids: torch.Tensor,
    tokenizer,
    mlm_probability: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids.clone()
    
    # Configure the runtime device.
    device = input_ids.device
    
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    mask_token_id = tokenizer.mask_token_id
    
    # Configure the runtime device.
    probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
    
    special_tokens_mask = (input_ids == pad_token_id) | (input_ids == cls_token_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    # Set the random seed.
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Compute the training loss.
    
    # Set the random seed.
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id
    
    # Set the random seed.
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]
    
    return input_ids, labels

