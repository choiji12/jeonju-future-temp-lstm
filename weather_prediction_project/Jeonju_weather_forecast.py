import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import itertools

# 1. 데이터 로드 및 전처리
file_path = './data/daily_average_temperature_20years.csv'
data = pd.read_csv(file_path)

# 날짜 변환 및 정렬
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

# 데이터셋 분할
train_size = int(0.6 * len(data))  # 60% 학습 데이터
val_size = int(0.2 * len(data))    # 20% 검증 데이터
test_size = len(data) - train_size - val_size  # 20% 테스트 데이터

# 데이터 분할
train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:train_size + val_size]
test_data = data.iloc[train_size + val_size:]

# 각 데이터셋의 온도 스케일링
scaler = MinMaxScaler()
train_data['temperature_scaled'] = scaler.fit_transform(train_data[['temperature']])
val_data['temperature_scaled'] = scaler.transform(val_data[['temperature']])
test_data['temperature_scaled'] = scaler.transform(test_data[['temperature']])

# 2. 시계열 데이터 생성 함수
def create_sequences(data, sequence_length=30):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data['temperature_scaled'].iloc[i-sequence_length:i].values)
        y.append(data['temperature_scaled'].iloc[i])
    return torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1), torch.tensor(np.array(y), dtype=torch.float32)

# 3. LSTM 모델 정의
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 타임스텝의 출력 사용
        return out

# 4. 하이퍼파라미터 탐색
hidden_size = [50, 100]
num_layers = [1, 2 ,3]
seq_lengths = [15, 30, 60, 90]
learning_rates = [0.001, 0.01]
epochs = [30,50]
dropouts = [0.1, 0.2, 0.3]
batch_sizes = [16, 32]

# 하이퍼파라미터 조합 생성
param_grid = itertools.product(hidden_size, num_layers, seq_lengths, learning_rates, epochs, dropouts, batch_sizes)
param_grid = list(param_grid)

# 배치 데이터 생성 함수
def create_batches(X, y, batch_size):
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델 학습 및 손실 저장 + RMSE 계산
def train_and_evaluate(params, dataset='val'):
    h_size, n_layers, seq_len, lr, n_epochs, d_rate, batch_size = params

    # 데이터 준비
    X_train_seq, y_train_seq = create_sequences(train_data, seq_len)
    if dataset == 'val':
        X_eval_seq, y_eval_seq = create_sequences(val_data, seq_len)
    else:  # 테스트 데이터
        X_eval_seq, y_eval_seq = create_sequences(test_data, seq_len)

    # 배치 데이터 생성
    train_loader = create_batches(X_train_seq, y_train_seq, batch_size)

    # 모델 초기화
    model = LSTM(input_size=1, hidden_size=h_size, num_layers=n_layers, output_size=1, dropout=d_rate)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 학습 및 손실 저장
    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss_avg = epoch_loss / len(train_loader)
        losses.append(epoch_loss_avg)

    # 검증 또는 테스트 데이터 평가
    model.eval()
    predicted = model(X_eval_seq).detach().numpy()
    actual = y_eval_seq.numpy()

    # RMSE 계산
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return rmse, model, losses, predicted, actual

# 검증 데이터에서 Top 10 파라미터 추출
val_results = []
for i, params in enumerate(param_grid):
    print(f"Validating combination {i + 1}/{len(param_grid)}: {params}")
    rmse, _, _, _, _ = train_and_evaluate(params, dataset='val')
    val_results.append((params, rmse))

val_results = sorted(val_results, key=lambda x: x[1])
top_10_params = val_results[:10]

print("\nTop 10 Configurations (Validation):")
for config, rmse in top_10_params:
    print(f"Config: {config}, Validation RMSE: {rmse:.4f}")

# 테스트 데이터에서 Top 10 파라미터 평가
test_results = []
for config, _ in top_10_params:
    print(f"Testing configuration: {config}")
    rmse, model, losses, predicted, actual = train_and_evaluate(config, dataset='test')
    test_results.append((config, rmse, model, losses, predicted, actual))

test_results = sorted(test_results, key=lambda x: x[1])
best_config, best_rmse, best_model, best_losses, best_predicted, best_actual = test_results[0]

print(f"\nBest Configuration (Test): {best_config}, Test RMSE: {best_rmse:.4f}")

# 최종 손실 값 출력
final_loss = best_losses[-1]
print(f"\nFinal Loss of the Best Model: {final_loss:.4f}")

# 최적 모델 학습 손실 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(best_losses) + 1), best_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve for Best Model')
plt.legend()
plt.show()

# 테스트 데이터 결과 시각화 (실제 vs 예측)
best_predicted_original = scaler.inverse_transform(best_predicted.reshape(-1, 1)).flatten()
best_actual_original = scaler.inverse_transform(best_actual.reshape(-1, 1)).flatten()

plt.figure(figsize=(10, 6))
plt.plot(best_actual_original, label='Actual Temperature', alpha=0.7)
plt.plot(best_predicted_original, label='Predicted Temperature', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Test Data: Actual vs Predicted Temperature for Best Model')
plt.legend()
plt.show()

# 다음날 온도 예측
last_sequence = test_data['temperature_scaled'].iloc[-best_config[2]:].values
last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

predicted_scaled = best_model(last_sequence).item()
predicted_temp = scaler.inverse_transform([[predicted_scaled]])[0][0]

print(f"\nPredicted Temperature for Next Day: {predicted_temp:.2f}°C")
