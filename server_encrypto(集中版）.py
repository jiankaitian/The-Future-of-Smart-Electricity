import json
import torch
from torch import optim
import torch.nn as nn
from models import PowerConsumptionLSTM  # 确保引入的是正确的模型
from clients import create_clients, load_data  # 使用clients.py中的函数创建客户端和加载数据
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

def plot_actual_vs_predicted(actual, predicted, client_id):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Power Consumption', color='blue')
    plt.plot(predicted, label='Predicted Power Consumption', color='orange')
    plt.title(f'Client {client_id}: Actual vs Predicted Power Consumption')
    plt.xlabel('Sample Index')
    plt.ylabel('Power Consumption')
    plt.legend()
    plt.show()

def rmse(targets, predictions):
    return sqrt(mean_squared_error(targets, predictions))

def main():
    # 加载配置
    conf_path = r'utils/conf.json'  # 请确保路径正确
    with open(conf_path, 'r') as conf_file:
        conf = json.load(conf_file)

    device = torch.device(conf['device'])

    # 创建客户端实例
    clients = create_clients(num_clients=conf['num_of_clients'])

    # 为每个客户端进行独立训练和评估
    for index, client in enumerate(clients):
        model = PowerConsumptionLSTM(conf['input_size'], conf['hidden_layer_size'], conf['output_size'], conf['number_of_layers']).to(device)
        client.set_model(model)
        optimizer = optim.Adam(model.parameters(), lr=conf['learning_rate'])
        loss_fn = nn.MSELoss()

        for epoch in range(conf['local_epochs']):
            # 客户端独立训练
            client.train(epochs=1, batch_size=conf['batch_size'], loss_fn=loss_fn, optimizer=optimizer)

            # 客户端独立评估
            _, actuals, predictions = client.evaluate(model, loss_fn)
            epoch_rmse = rmse(actuals, predictions)
            print(f"Client {index + 1}, Epoch {epoch + 1}, RMSE: {epoch_rmse}")

        # 最终评估
        _, final_actuals, final_predictions = client.evaluate(model, loss_fn)
        final_rmse = rmse(final_actuals, final_predictions)
        print(f"Client {index + 1} Final RMSE: {final_rmse}")

        # 绘制实际值与预测值的对比图
        plot_actual_vs_predicted(final_actuals, final_predictions, index + 1)

if __name__ == "__main__":
    main()
