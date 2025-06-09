#!/usr/bin/env python3
"""
simple neural network training script for training feedforward network on the iris dataset
"""

import json
import os
import sys
import argparse
import time
from datetime import datetime
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleNN(nn.Module):
    """simple feedforward neural network"""
    
    def __init__(self, config):
        super(SimpleNN, self).__init__()
        
        model_config = config['model']
        layers = []
        
        in_features = model_config['input_size']  # input layer
        
        # hidden layers
        for i, layer_size in enumerate(model_config['layer_sizes']):
            layers.append(nn.Linear(in_features, layer_size))
            
            # activation function
            if model_config['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif model_config['activation'] == 'tanh':
                layers.append(nn.Tanh())
            elif model_config['activation'] == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            if model_config['dropout'] > 0:
                layers.append(nn.Dropout(model_config['dropout']))
            
            in_features = layer_size
        
        layers.append(nn.Linear(in_features, model_config['output_size'])) # output layer
        layers.append(nn.Softmax(dim=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_config(config_path='config.json'):
    """load configuration from JSON file"""
    if not os.path.exists(config_path):
        print(f"error: config file '{config_path}' not found!")
        print("hint: copy the baseline config first:")
        print(f"   cp config.json {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        return json.load(f)


def load_data():
    """load & preprocess the iris dataset"""
    print("loading iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def create_data_loaders(X_train, X_test, y_train, y_test, batch_size):
    """torcch data loaders"""
    
    # convert input features and labels to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_model(model, train_loader, config):
    """train the neural network"""
    training_config = config['training']
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    
    print(f"training for {training_config['epochs']} epochs...")
    print(f"learning rate: {training_config['learning_rate']}")
    print(f"batch size: {training_config['batch_size']}")
    
    model.train()
    for epoch in range(training_config['epochs']):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"epoch [{epoch+1}/{training_config['epochs']}], loss: {avg_loss:.4f}")


def evaluate_model(model, test_loader):
    """evaluate the model using accuracy and f1 score"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.numpy())
            all_targets.extend(batch_y.numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    return accuracy, f1, all_targets, all_predictions


def log_results(config, accuracy, f1, training_time):
    """log the experiment results to a json file"""
    
    os.makedirs('evaluation', exist_ok=True)
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": config['experiment']['name'],
        "author": config['experiment']['author'],
        "description": config['experiment']['description'],
        "config": {
            "model": config['model'],
            "training": config['training']
        },
        "results": {
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
            "training_time_seconds": round(training_time, 2)
        }
    }
    
    
    results_file = f"evaluation/results_{config['experiment']['name']}.json"
    # check if results file exists
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {"experiments": []}
    
    # append new results
    results["experiments"].append(result)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"results logged to {results_file}")

    return results_file


def main():
    parser = argparse.ArgumentParser(description='train a neural network on iris dataset')
    parser.add_argument('--config', type=str, default='config.json', 
                       help='path to configuration file (default: config.json)')
    parser.add_argument('--no-log', action='store_true',
                        help='disable outputting results to json file')
    args = parser.parse_args()
    
    print("starting training script...")
    print("=" * 50)
    print(f"using config file: {args.config}")
    print()
    
    config = load_config(args.config)
    print(f"loaded config for experiment: {config['experiment']['name']}")
    print(f"author: {config['experiment']['author']}")
    print(f"description: {config['experiment']['description']}")
    print()
    
    X_train, X_test, y_train, y_test = load_data()
    train_loader, test_loader = create_data_loaders(
        X_train, X_test, y_train, y_test, 
        config['training']['batch_size']
    )
    
    print(f"dataset: {len(X_train)} training samples, {len(X_test)} test samples")
    print()
    
    model = SimpleNN(config)
    print(f"model architecture:")
    print(f"- Layers: {config['model']['layer_sizes']}")
    print(f"- Activation: {config['model']['activation']}")
    print(f"- Dropout: {config['model']['dropout']}")
    print()
    
    start_time = time.time()
    train_model(model, train_loader, config)
    training_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("evaluation results (test set):")
    print("=" * 50)
    
    accuracy, f1, y_true, y_pred = evaluate_model(model, test_loader)
    
    print(f"accuracy: {accuracy:.4f}")
    print(f"f1-score: {f1:.4f}")
    print(f"training Time: {training_time:.2f} seconds")
    
    if not args.no_log:
        results_file = log_results(config, accuracy, f1, training_time)    
        print(f"\n training complete! check {results_file} for logged results.")

if __name__ == "__main__":
    main() 