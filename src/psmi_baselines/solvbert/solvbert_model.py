import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import AutoTokenizer


class SolvBERT(nn.Module):
    
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        cls_token_id: int = 101,
        mask_token_id: int = 103,
        hidden_dropout_rate: float = 0.4,
        num_outputs: int = 6,  # Configure the output artifacts.
    ):
        super(SolvBERT, self).__init__()
        
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            mask_token_id=mask_token_id,
        )
        
        self.bert = BertModel(config)
        
        self.regression_head = nn.Sequential(
            nn.Dropout(hidden_dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_rate),
            nn.Linear(hidden_size // 2, num_outputs)  # Configure the output artifacts.
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        return_embeddings=False
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Generate model predictions.
        prediction = self.regression_head(cls_embedding)  # [batch_size, num_outputs]
        
        if return_embeddings:
            return prediction, cls_embedding
        return prediction
    
    def get_embeddings(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs.last_hidden_state[:, 0, :]


class SolvBERTForMLM(nn.Module):
    
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        cls_token_id: int = 101,
        mask_token_id: int = 103,
    ):
        super(SolvBERTForMLM, self).__init__()
        
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            mask_token_id=mask_token_id,
        )
        
        self.bert = BertModel(config)
        
        # Generate model predictions.
        self.cls = nn.Linear(hidden_size, vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.cls(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), labels.view(-1))
        
        return loss, prediction_scores

