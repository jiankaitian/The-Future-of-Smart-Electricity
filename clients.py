import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from models import PowerConsumptionLSTM

class Client(object):
    def __init__(self, train_data, test_data, device='cpu'):
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        self.model = None  # Initialize model attribute

    def get_test_data_loader(self, batch_size=1, shuffle=False):
        """创建并返回测试数据的 DataLoader"""
        test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=shuffle)
        return test_loader

    def set_model(self, model):
        self.model = model

    def train(self, epochs, batch_size, loss_fn, optimizer):
        assert self.model is not None, "Model not set."
        self.model.train()
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        epoch_losses = []  # 新增：用于记录每个epoch的平均损失
        for epoch in range(epochs):
            batch_losses = []  # 用于记录当前epoch中每个batch的损失
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if inputs.dim() == 2:
                    inputs = inputs.unsqueeze(1)  # 将inputs变为[batch_size, 1, features]
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_loss = np.mean(batch_losses)  # 计算当前epoch的平均损失
            epoch_losses.append(epoch_loss)
        return epoch_losses  # 返回每个epoch的平均损失

    def evaluate(self, model, loss_fn):
        model.eval()
        test_loader = self.get_test_data_loader(batch_size=1, shuffle=False)
        total_loss = 0
        actuals = []
        predictions = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if inputs.dim() == 2:
                    inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)  # 使用和训练时相同的损失函数
                total_loss += loss.item()
                actuals.extend(targets.view(-1).tolist())
                predictions.extend(outputs.view(-1).tolist())
        average_loss = total_loss / len(test_loader)  # 注意这里是除以test_loader的长度
        return average_loss, actuals, predictions

    def collect_update(self):
        assert self.model is not None, "Model not set."
        return {name: param.cpu().data.numpy() for name, param in self.model.state_dict().items()}

    def update_model(self, global_weights):
        """更新客户端模型的权重为全局模型权重"""
        with torch.no_grad():
            for param, new_param in zip(self.model.parameters(), global_weights):
                param.data = new_param.data.clone()

def load_data():
    # 读取CSV文件
    data_path = 'data/PowerDataZon1.csv'
    df = pd.read_csv(data_path)
    feature_cols = ['hour', 'Temperature', 'general diffuse flows']
    target_col = 'Zone 1 Power Consumption'
    features = df[feature_cols].values
    labels = df[target_col].values.reshape(-1, 1)
    return features, labels

def prepare_dataset():
    features, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    return train_dataset, test_dataset

def create_clients(num_clients=2):
    train_dataset, test_dataset = prepare_dataset()
    shard_size = len(train_dataset) // num_clients
    clients = []
    for i in range(num_clients):
        start_index = i * shard_size
        end_index = start_index + shard_size if i < num_clients - 1 else len(train_dataset)
        client_train_dataset = TensorDataset(train_dataset[start_index:end_index][0], train_dataset[start_index:end_index][1])
        client = Client(client_train_dataset, test_dataset)
        clients.append(client)

        # 打印客户端接收的训练数据集的前几条记录
        print(f"Client {i+1} Training Dataset Sample:")
        print("Features:\n", client_train_dataset.tensors[0][:5].numpy())  # 打印前5个特征
        print("Labels:\n", client_train_dataset.tensors[1][:5].numpy())    # 打印前5个标签
    return clients


def plot_actual_vs_predicted(actual, predicted, file_name='actual_vs_predicted.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Power Consumption')
    plt.plot(predicted, label='Predicted Power Consumption', linestyle='--')
    plt.title('Actual vs Predicted Power Consumption')
    plt.xlabel('Sample Index')
    plt.ylabel('Power Consumption')
    plt.legend()
    plt.savefig(file_name)
    plt.show()

if __name__ == "__main__":
    train_dataset, test_dataset = prepare_dataset()
    clients = create_clients(num_clients=2)
    for index, client in enumerate(clients):
        model = PowerConsumptionLSTM(input_size=3, hidden_layer_size=100, output_size=1, num_layers=2).to(client.device)
        client.set_model(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        client.train(epochs=5, batch_size=32, loss_fn=loss_fn, optimizer=optimizer)
        loss, actuals, predictions = client.evaluate(model)
        print(f"Client {index} evaluation loss:", loss)
        plot_actual_vs_predicted(actuals, predictions, f'client_{index}_actual_vs_predicted.png')
