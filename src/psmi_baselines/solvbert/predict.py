"""Implement the solvbert predict baseline module."""
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from .solvbert_model import SolvBERT
from .data_utils import build_tokenizer, create_smiles_combination


def load_model(model_path, tokenizer_path, device):
    """Run the load model baseline operation."""
    # Load the input data.
    tokenizer = build_tokenizer(vocab_path=tokenizer_path)
    
    # Load the input data.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Baseline workflow step.
        config = {
            'vocab_size': len(tokenizer),
            'hidden_size': 256,
            'num_layers': 6,
            'num_heads': 8,
            'intermediate_size': 1024,
            'hidden_dropout_rate': 0.4,
        }
    
    # Configure the baseline model.
    model = SolvBERT(
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size'],
        num_hidden_layers=config.get('num_layers', 6),
        num_attention_heads=config.get('num_heads', 8),
        intermediate_size=config.get('intermediate_size', 1024),
        hidden_dropout_rate=config.get('hidden_dropout_rate', 0.4),
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        mask_token_id=tokenizer.mask_token_id,
    ).to(device)
    
    # Load the input data.
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model, tokenizer


def predict_single(model, tokenizer, solvent_smiles, solute_smiles, device, max_length=512):
    """Run the predict single baseline operation."""
    # Baseline workflow step.
    smiles_combination = create_smiles_combination(solvent_smiles, solute_smiles)
    
    # Tokenize
    encoded = tokenizer(
        smiles_combination,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Generate model predictions.
    with torch.no_grad():
        prediction = model(
            input_ids=encoded['input_ids'].to(device),
            attention_mask=encoded['attention_mask'].to(device)
        )
    
    return prediction.item()


def predict_batch(model, tokenizer, data_path, output_path, device, max_length=512, 
                  solvent_col='solvent', solute_col='solute', batch_size=32):
    """Run the predict batch baseline operation."""
    # Read the input data.
    df = pd.read_csv(data_path)
    
    predictions = []
    
    # Baseline workflow step.
    for i in tqdm(range(0, len(df), batch_size), desc=" prediction in "):
        batch_df = df.iloc[i:i+batch_size]
        
        batch_smiles = []
        for _, row in batch_df.iterrows():
            solvent = str(row[solvent_col])
            solute = str(row[solute_col])
            smiles_combination = create_smiles_combination(solvent, solute)
            batch_smiles.append(smiles_combination)
        
        # Tokenize
        encoded = tokenizer(
            batch_smiles,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Generate model predictions.
        with torch.no_grad():
            batch_predictions = model(
                input_ids=encoded['input_ids'].to(device),
                attention_mask=encoded['attention_mask'].to(device)
            ).squeeze(-1).cpu().numpy()
        
        predictions.extend(batch_predictions)
    
    # Save the generated artifacts.
    df['prediction'] = predictions
    df.to_csv(output_path, index=False)
    print(f" predictions saved to : {output_path}")


def main():
    parser = argparse.ArgumentParser(description='SolvBERT prediction ')
    parser.add_argument('--model_path', type=str, required=True, help=' model path ')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='tokenizer path ')
    parser.add_argument('--solvent', type=str, default=None, help=' solvent SMILES( single-sample prediction )')
    parser.add_argument('--solute', type=str, default=None, help=' solute SMILES( single-sample prediction )')
    parser.add_argument('--input_data', type=str, default=None, help=' input data path ( batch prediction )')
    parser.add_argument('--output_data', type=str, default='predictions.csv', help=' output data path ')
    parser.add_argument('--solvent_col', type=str, default='solvent', help=' solvent column First Name ')
    parser.add_argument('--solute_col', type=str, default='solute', help=' solute column First Name ')
    parser.add_argument('--batch_size', type=int, default=32, help=' batch size ')
    parser.add_argument('--max_length', type=int, default=512, help=' maximum sequence length ')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help=' device ')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f" use device : {device}")
    
    # Load the input data.
    print(" load model ...")
    model, tokenizer = load_model(args.model_path, args.tokenizer_path, device)
    print(" model load complete ")
    
    # Generate model predictions.
    if args.solvent and args.solute:
        prediction = predict_single(
            model, tokenizer, args.solvent, args.solute, device, args.max_length
        )
        print(f"\n solvent : {args.solvent}")
        print(f" solute : {args.solute}")
        print(f" prediction : {prediction:.4f}")
    
    # Generate model predictions.
    elif args.input_data:
        predict_batch(
            model, tokenizer, args.input_data, args.output_data, device,
            args.max_length, args.solvent_col, args.solute_col, args.batch_size
        )
    
    else:
        print(" error : Please provide --solvent and --solute ( single-sample prediction ) or --input_data ( batch prediction )")


if __name__ == '__main__':
    main()

