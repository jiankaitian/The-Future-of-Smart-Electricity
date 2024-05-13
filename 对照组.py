import json
import torch
from torch import optim
import torch.nn as nn
from models import PowerConsumptionLSTM  # 确保引入的是正确的模型
from clients import create_clients  # 假设clients.py已正确设置创建客户端的函数

def evaluate_client(client, model, device):
    """评估客户端模型的性能，并返回RMSE"""
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for data, targets in client.get_test_data_loader(batch_size=1, shuffle=False):
            # 确保数据和目标的类型为torch.float32
            data, targets = data.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)
            if data.dim() == 2:
                data = data.unsqueeze(1)
            outputs = model(data)
            mse = nn.functional.mse_loss(outputs, targets, reduction='sum').item()
            total_mse += mse
    rmse = (total_mse / len(client.test_data))**0.5
    return rmse

if __name__ == "__main__":
    # 加载配置
    conf_path = r'utils/conf.json'  # 请根据实际情况调整路径
    with open(conf_path, 'r') as conf_file:
        conf = json.load(conf_file)

    device = torch.device(conf['device'])

    # 实例化模型并设置优化器...
    model = PowerConsumptionLSTM(conf['input_size'], conf['hidden_layer_size'], conf['output_size'], conf['number_of_layers'])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=conf['learning_rate'])

    # 创建客户端实例...
    clients = create_clients(num_clients=conf['num_of_clients'])

    for round in range(conf['number_of_communications']):
        print(f"Communication round {round+1}")

        round_rmses = []
        for client in clients:
            client.set_model(model)
            client.train(epochs=conf['local_epochs'], batch_size=conf['batch_size'], loss_fn=nn.MSELoss(), optimizer=optimizer)
            rmse = evaluate_client(client, model, device)
            round_rmses.append(rmse)

        print(f"RMSEs after round {round + 1}:", round_rmses)
