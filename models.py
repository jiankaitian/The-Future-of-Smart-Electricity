import torch
import torch.nn as nn
import torch.nn.functional as F

class PowerConsumptionLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=100, output_size=1, num_layers=2):
        super(PowerConsumptionLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)

        # 输出层
        self.linear = nn.Linear(hidden_layer_size, output_size)
    

    def forward(self, x):
        # 自动调整 h0 和 c0 的大小以匹配输入批次大小
        batch_size = x.size(0)  # 获取批次大小
        h0 = torch.zeros(num_layers, batch_size, self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(num_layers, batch_size, self.hidden_layer_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# 设定模型参数
input_size = 3  # 特征数量：hour, Temperature, general diffuse flows
hidden_layer_size = 100  # 隐藏层大小
output_size = 1  # 输出大小：预测的Zone 1 Power Consumption
num_layers = 2  # LSTM层的数量

# 实例化模型
model = PowerConsumptionLSTM(input_size, hidden_layer_size, output_size, num_layers)
print(model)
