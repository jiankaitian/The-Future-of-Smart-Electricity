import os
import json
from tqdm import tqdm
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from models import PowerConsumptionLSTM  # 确保引入的是正确的模型
from clients import create_clients  # 假设clients.py已正确设置创建客户端的函数
import paillier
from sklearn.metrics import mean_squared_error
from math import sqrt
import sm2
import matplotlib.pyplot as plt

# 确保目录存在
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_actual_vs_predicted(actual, predicted):
    plt.figure(figsize=(15, 5))  # 图表的大小可根据需要进行调整
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predicted, label='Predicted', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Global Active Power')
    plt.title('Actual vs Predicted Power Consumption')
    plt.legend()
    plt.show()

def callback_function(rmse_value, actual_values, predicted_values):
    # 这个函数将由GUI提供，用于接收RMSE值和图表数据
    # 例如，GUI可以使用这些数据更新用户界面
    pass

# 加密模型权重
def encrypt_weights(public_key, weights):
    encrypted_weights = {}
    for k, v in weights.items():
        flat_v = v.cpu().numpy().flatten()
        encrypted_weights[k] = [public_key.encrypt(x) for x in flat_v]
    return encrypted_weights

# 解密模型权重
def decrypt_weights(private_key, encrypted_weights):
    decrypted_weights = {}
    for k, v in encrypted_weights.items():
        decrypted_weights[k] = np.array([private_key.decrypt(x) for x in v])
    return decrypted_weights
'''
# 添加噪声
def add_noise(weights, noise_level, sigma):
    if noise_level > 0:
        noisy_weights = {}
        for k, v in weights.items():
            shape = v.shape
            noise = np.random.laplace(0, sigma, size=shape)
            noisy_weights[k] = v + noise
        return noisy_weights
    return weights
'''
# 更新 add_noise 函数以接受 numpy.ndarray 类型的 weights
def add_noise(aggregated, noise_level, sigma):
    if noise_level > 0:
        # 直接在 aggregated 数组上添加噪声
        noise = np.random.laplace(0, sigma, aggregated.shape)
        noisy_aggregated = aggregated + noise
        return noisy_aggregated
    return aggregated

def rmse(targets, predictions):
    return sqrt(mean_squared_error(targets, predictions))

private_key = '00B9AB0B828FF68872F21A837FC303668428DEA11DCD1B24429D0C99E24EED83D5'
public_key = 'B9C9A6E04E9C91F7BA880429273747D7EF5DDEB0BB2FF6317EB00BEF331A83081A6994B8993F3F5D6EADDDB81872266C87C018FB4162F5AF347B483E24620207'
sm2_crypt = sm2.CryptSM2(public_key=public_key, private_key=private_key)

def main():
    # 加载配置
    conf_path = r'utils/conf.json'  # 请根据实际情况调整路径
    with open(conf_path, 'r') as conf_file:
        conf = json.load(conf_file)

    device = torch.device(conf['device'])

    # 实例化模型并设置优化器...
    model = PowerConsumptionLSTM(conf['input_size'], conf['hidden_layer_size'], conf['output_size'], conf['number_of_layers'])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=conf['learning_rate'])

    # 生成密钥对和创建客户端实例...
    public_key, private_key = paillier.generate_paillier_keypair()
    clients = create_clients(num_clients=conf['num_of_clients'])

    for round in range(conf['number_of_communications']):
        print(f"Communication round {round + 1}")
        client_losses = []  # 用于收集每个客户端的损失信息

        all_actual_values = []
        all_predicted_values = []

        # 在训练开始前为每个客户端设置模型
        for client in clients:
            client.set_model(model)

        # 初始化客户端更新列表
        client_updates = []

        for client in tqdm(clients, desc="Collecting client updates"):
            # 正确捕获client.train(...)的返回值
            epoch_losses = client.train(epochs=conf['local_epochs'], batch_size=conf['batch_size'],
                                        loss_fn=nn.MSELoss(), optimizer=optimizer)
            update = client.collect_update()  # 收集客户端的更新
            client_updates.append(update)

            # 打印每个客户端的训练损失
            print(f"Client {clients.index(client)} training losses: {epoch_losses}")
            client_losses.append(epoch_losses)  # 收集每个客户端的损失以便后续分析

            # 新增：评估每个客户端的模型并计算RMSE
            loss, actuals, predictions = client.evaluate(model, nn.MSELoss())
            client_rmse = rmse(actuals, predictions)
            print(f"Client {clients.index(client)} RMSE: {client_rmse}")

            # 可选：收集每个客户端的RMSE以便后续分析
            client_losses.append((epoch_losses, client_rmse))

        # 聚合和加密更新
        global_update = {}
        for k in model.state_dict().keys():
            # 使用 np.stack 对同一键的更新进行堆叠，然后计算平均值
            updates_stack = np.stack([u[k] for u in client_updates])
            aggregated = np.mean(updates_stack, axis=0)
            noisy_aggregated = add_noise(aggregated, conf['noise'], conf['sigma'])
            # 将处理后的更新重新放入 global_update 字典
            global_update[k] = noisy_aggregated

        actual_values = []
        predicted_values = []
        with torch.no_grad():
            total_rmse = 0.0
            count = 0
            for client in clients:
                test_loader = client.get_test_data_loader(batch_size=1, shuffle=False)
                for data, targets in test_loader:
                    data, targets = data.to(device), targets.to(device)
                    if data.dim() == 2:
                        data = data.unsqueeze(1)  # 将data变为[batch_size, 1, features]
                    outputs = model(data)
                    mse = mean_squared_error(targets.cpu().numpy(), outputs.cpu().detach().numpy())
                    total_rmse += sqrt(mse)
                    count += 1
                    actual_values.extend(targets.cpu().numpy())
                    predicted_values.extend(outputs.cpu().detach().numpy())
            average_rmse = total_rmse / count if count > 0 else float('inf')
            print(f"Average RMSE after round {round + 1}: {average_rmse}")

            callback_function(average_rmse, actual_values, predicted_values)

           # plot_actual_vs_predicted(actual_values, predicted_values)

    # 保存模型
    ensure_dir(conf['checkpoint_save_path'])
    model_save_path = os.path.join(conf['checkpoint_save_path'], 'final_model.pt')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# 这里检查是否该文件被直接运行，并在是的情况下调用 main 函数
if __name__ == "__main__":
    main()
