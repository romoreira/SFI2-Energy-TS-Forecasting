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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("===Training on: "+str(device)+str("==="))

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
df_may = pd.read_csv("dataset/maio.csv")
df_august = pd.read_csv("dataset/agosto.csv", skiprows=1, on_bad_lines='warn')
df = pd.concat([df_may, df_august], ignore_index=True)
df = df.drop(columns=['fecha_esp32', 'weekday', 'MAC'])

# Normalizar a feature 'corriente'
scaler = StandardScaler()
df['corriente'] = scaler.fit_transform(df[['corriente']])

# Criar janelas de séries temporais (sequências) para usar como pares com offset
def create_pairs(df, window_size=10, offset=1):
    X_i, X_j = [], []
    for i in range(len(df) - window_size - offset):
        X_i.append(df['corriente'].iloc[i:i+window_size].values)
        X_j.append(df['corriente'].iloc[i+offset:i+offset+window_size].values)  # Pares deslocados
    return torch.tensor(X_i, dtype=torch.float32), torch.tensor(X_j, dtype=torch.float32)

# Criar os pares
X_i, X_j = create_pairs(df)

# Definir DataLoader para o treinamento
dataset = TensorDataset(X_i, X_j)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Definir modelo e otimizador
encoder = TimeSeriesEncoder(input_dim=1, hidden_dim=16).to(device)
model = ContrastiveModel(encoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 30
epoch_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for x_i, x_j in dataloader:
        x_i, x_j = x_i.to(device), x_j.to(device)
        optimizer.zero_grad()
        h_i, h_j = model(x_i.unsqueeze(-1), x_j.unsqueeze(-1))  # Adicionar dimensão extra para LSTM
        loss = contrastive_loss(h_i, h_j)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss / len(dataloader)
    epoch_losses.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}")

# Plotar o gráfico de perda (loss) ao longo das épocas
plt.figure()
plt.plot(range(1, num_epochs + 1), epoch_losses, label="Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.savefig('loss_over_epochs.pdf')

# Identificar anomalias
def detect_anomalies(model, df, threshold=0.5, offset=1):
    model.eval()
    X_i, X_j = create_pairs(df, offset=offset)
    X_i, X_j = X_i.to(device), X_j.to(device)
    h_i, h_j = model(X_i.unsqueeze(-1), X_j.unsqueeze(-1))
    distances = torch.norm(h_i - h_j, dim=1)
    anomalies = distances > threshold
    return anomalies, distances

# Detectar anomalias com diferentes offsets
offsets = [1, 2, 5, 10]  # Exemplo de diferentes valores de offset
results = {}
early_prediction_times = {}

for offset in offsets:
    anomalies, _ = detect_anomalies(model, df, threshold=0.5, offset=offset)
    early_prediction_times[offset] = offset  # O tempo de antecedência é igual ao offset
    print(f"Anomalies detected with offset {offset}: {anomalies.sum().item()}")
    results[offset] = anomalies

# Visualizar os resultados para diferentes offsets
for offset, anomalies in results.items():
    anomaly_indices = torch.nonzero(anomalies).flatten().cpu().numpy()
    df['index'] = range(0, len(df))
    plt.figure()
    plt.plot(df['index'], df['corriente'], label="Corriente")
    plt.scatter(df['index'][anomaly_indices], df['corriente'][anomaly_indices], color='red', label=f"Anomalias (offset {offset})")
    plt.xlabel('Time Index')
    plt.ylabel('Corriente')
    plt.legend()
    plt.title(f'Anomaly Plot with Offset {offset}')
    plt.savefig(f'anomaly_plot_offset_{offset}.pdf')

# Plotar os tempos de antecedência
plt.figure()
plt.plot(list(early_prediction_times.keys()), list(early_prediction_times.values()), marker='o')
plt.xlabel('Offset')
plt.ylabel('Early Prediction Time')
plt.title('Early Prediction Time vs. Offset')
plt.savefig('early_prediction_times.pdf')
