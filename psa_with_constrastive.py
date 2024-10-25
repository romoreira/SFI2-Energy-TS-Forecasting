import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from pandas import read_csv
from pyswarm import pso
from sklearn.metrics import f1_score
import numpy as np

# Defina uma rede de exemplo para aprender representações das séries temporais
class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TimeSeriesEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # Usamos o último estado oculto como embedding
        return h_n.squeeze(0)

# Defina o modelo siamesa para o treinamento contrastivo
class ContrastiveModel(nn.Module):
    def __init__(self, encoder):
        super(ContrastiveModel, self).__init__()
        self.encoder = encoder

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)  # embedding da amostra i
        h_j = self.encoder(x_j)  # embedding da amostra j
        return h_i, h_j


# Definir a função de perda InfoNCE
def contrastive_loss(h_i, h_j, temperature=0.5):
    h_i = nn.functional.normalize(h_i, dim=-1)
    h_j = nn.functional.normalize(h_j, dim=-1)

    logits = torch.matmul(h_i, h_j.T) / temperature
    labels = torch.arange(h_i.shape[0]).to(h_i.device)
    return nn.CrossEntropyLoss()(logits, labels)


#---------------------------------#

# Leitura e pré-processamento do CSV
df = read_csv("dataset/maio.csv")
df = df.drop(columns=['fecha_esp32', 'weekday', 'MAC'])
df = df.head(1000)  # Usar apenas um subconjunto para treinamento

# Normalizar a feature 'corriente'
scaler = StandardScaler()
df['corriente'] = scaler.fit_transform(df[['corriente']])

# Criar janelas de séries temporais (sequências) para usar como pares
def create_pairs(df, window_size=10):
    X_i, X_j = [], []
    for i in range(len(df) - window_size):
        X_i.append(df['corriente'].iloc[i:i+window_size].values)
        X_j.append(df['corriente'].iloc[i+1:i+1+window_size].values)  # Pares deslocados
    return torch.tensor(X_i, dtype=torch.float32), torch.tensor(X_j, dtype=torch.float32)

# Criar os pares
X_i, X_j = create_pairs(df)

# Definir DataLoader para o treinamento
dataset = TensorDataset(X_i, X_j)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Definir modelo e otimizador
encoder = TimeSeriesEncoder(input_dim=1, hidden_dim=16)  # 1 feature de entrada (corriente)
model = ContrastiveModel(encoder)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Loop de treinamento
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for x_i, x_j in dataloader:
        optimizer.zero_grad()
        h_i, h_j = model(x_i.unsqueeze(-1), x_j.unsqueeze(-1))  # Adicionar dimensão extra para LSTM
        loss = contrastive_loss(h_i, h_j)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

# Identificar anomalias
def detect_anomalies(model, df, threshold=0.5):
    model.eval()
    X_i, X_j = create_pairs(df)
    h_i, h_j = model(X_i.unsqueeze(-1), X_j.unsqueeze(-1))
    distances = torch.norm(h_i - h_j, dim=1)
    anomalies = distances > threshold
    return anomalies

# Detectar anomalias
anomalies = detect_anomalies(model, df)
print(f"Anomalies detected: {anomalies.sum().item()}")


# Obter os índices das anomalias
anomaly_indices = torch.nonzero(anomalies).flatten().cpu().numpy()

# Plotar o gráfico de corrente com anomalias marcadas
df['index'] = range(0, len(df))
plt.figure()
plt.plot(df['index'], df['corriente'], label="Corriente")

# Usar índices em vez de um array booleano
plt.scatter(df['index'][anomaly_indices], df['corriente'][anomaly_indices], color='red', label="Anomalias")
plt.xlabel('Time Index')
plt.ylabel('Corriente')
plt.legend()
plt.savefig('anomaly_plot_without_optimization.png')


# Função para detectar anomalias com sensibilidade controlada
def detect_anomalies_with_threshold(model, df, threshold):
    model.eval()
    X_i, X_j = create_pairs(df)
    h_i, h_j = model(X_i.unsqueeze(-1), X_j.unsqueeze(-1))
    distances = torch.norm(h_i - h_j, dim=1).detach().cpu().numpy()
    anomalies = distances > threshold
    return anomalies, distances

# Função objetivo para otimização
def objective(threshold):
    _, distances = detect_anomalies_with_threshold(model, df, threshold)
    # Usar uma métrica baseada nas distâncias, como a média
    score = np.mean(distances > threshold)
    return -score  # Minimizar o número de falsos positivos

# Definindo os limites de threshold
lb = [0.0]
ub = [2.0]  # Ajuste conforme necessário

# Executando o PSO
best_threshold, best_loss = pso(objective, lb, ub, swarmsize=50, maxiter=100)

print(f"Best threshold: {best_threshold}")
print(f"Best loss (negative score): {best_loss}")

# Detectar anomalias usando o melhor threshold encontrado
anomalies, _ = detect_anomalies_with_threshold(model, df, best_threshold)
print(f"Anomalies detected with optimized threshold: {anomalies.sum().item()}")

# Plotar o gráfico de corrente com anomalias marcadas
anomaly_indices = torch.nonzero(anomalies).flatten().cpu().numpy()
df['index'] = range(0, len(df))
plt.figure()
plt.plot(df['index'], df['corriente'], label="Corriente")
plt.scatter(df['index'][anomaly_indices], df['corriente'][anomaly_indices], color='red', label="Anomalias")
plt.xlabel('Time Index')
plt.ylabel('Corriente')
plt.legend()
plt.savefig('anomaly_plot_with_optimization.png')
