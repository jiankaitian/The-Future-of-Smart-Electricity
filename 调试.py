import json
import tkinter as tk
from tkinter import ttk
import threading
import server_encrypto
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为微软雅黑，或者其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 读取和保存 JSON 配置部分保持不变

def load_config(filepath='utils/conf.json'):
    with open(filepath, 'r') as file:
        return json.load(file)

def save_config(config, filepath='utils/conf.json'):
    with open(filepath, 'w') as file:
        json.dump(config, file, indent=4)

# 更新 GUI 界面并填充参数部分保持不变

def create_gui():
    config = load_config()

    root = tk.Tk()
    root.title("配置编辑器")

    entries = {}
    for i, (key, value) in enumerate(config.items()):
        tk.Label(root, text=key).grid(row=i, column=0)
        entry = tk.Entry(root)
        entry.insert(0, str(value))
        entry.grid(row=i, column=1)
        entries[key] = entry

    def save_action():
        for key, entry in entries.items():
            config[key] = type(config[key])(entry.get())
        save_config(config)
        tk.messagebox.showinfo("保存", "配置已成功保存！")

    # 第一段代码结束的地方修改 run_script 函数，使其调用图表界面
    def run_script():
        root.destroy()  # 关闭当前窗口
        show_charts_gui()  # 调用显示图表的 GUI

    tk.Button(root, text="保存", command=save_action).grid(row=len(config), column=0)
    tk.Button(root, text="保存配置文件", command=run_script).grid(row=len(config), column=1)

    root.mainloop()

# 将第二段代码的内容整合为一个函数
def show_charts_gui():
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import threading
    import torch
    from models import PowerConsumptionLSTM  # 确保引入了正确的模型
    import server_encrypto

    # 假设你的模型已经加载并准备好了
    model = PowerConsumptionLSTM(input_size=3, hidden_layer_size=100, output_size=1, num_layers=2)
    model.load_state_dict(torch.load('checkpoints/final_model.pt'))
    model.eval()

    current_round = 0  # 用于跟踪当前的轮数

    def predict():
        try:
            hour = float(hour_entry.get())
            temperature = float(temperature_entry.get())
            diffuse_flows = float(diffuse_flows_entry.get())
            input_tensor = torch.tensor([[hour, temperature, diffuse_flows]]).float()
            if input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(0)  # 调整维度以符合模型要求
            prediction = model(input_tensor)
            prediction_label.config(text=f'预测电力消耗: {prediction.item():.4f}')
        except ValueError:
            prediction_label.config(text="请输入有效的数字。")

    def gui_callback_function(rmse_value, actual_values, predicted_values):
        global current_round
        current_round += 1  # 在调用 update_gui 前更新轮次

        def update_gui():
            rmse_var.set(f"第 {current_round} 轮: RMSE: {rmse_value:.4f}")
            ax.clear()
            ax.plot(actual_values, label='实际值', color='blue')
            ax.plot(predicted_values, label='预测值', color='orange')
            ax.set(xlabel='时间', ylabel='全球活跃功率', title=f'第 {current_round} 轮: 实际与预测电力消耗')
            ax.legend()
            canvas.draw()
            waiting_label.config(text=f"正在更新第 {current_round+1} 轮...")  # 根据当前轮次更新状态

        waiting_label.config(text="正在更新，请稍候...")  # 在开始更新前显示等待提示
        window.after(0, update_gui)

    server_encrypto.callback_function = gui_callback_function

    def run_server_encrypto():
        global current_round
        current_round = 0
        waiting_label.config(text="请等待...")  # 开始时即更新等待提示

        def target():
            try:
                server_encrypto.main()
            except Exception as e:
                print(f"运行 server_encrypto.main 出错: {e}")

        threading.Thread(target=target).start()

    window = tk.Tk()
    window.title("综合电力消耗预测界面")

    prediction_frame = ttk.Frame(window)
    prediction_frame.pack(fill=tk.X)

    hour_label = ttk.Label(prediction_frame, text="小时:")
    hour_label.pack(side=tk.LEFT)
    hour_entry = ttk.Entry(prediction_frame)
    hour_entry.pack(side=tk.LEFT)

    temperature_label = ttk.Label(prediction_frame, text="温度:")
    temperature_label.pack(side=tk.LEFT)
    temperature_entry = ttk.Entry(prediction_frame)
    temperature_entry.pack(side=tk.LEFT)

    diffuse_flows_label = ttk.Label(prediction_frame, text="散射流:")
    diffuse_flows_label.pack(side=tk.LEFT)
    diffuse_flows_entry = ttk.Entry(prediction_frame)
    diffuse_flows_entry.pack(side=tk.LEFT)

    predict_button = ttk.Button(prediction_frame, text="预测", command=predict)
    predict_button.pack(side=tk.LEFT)

    prediction_label = ttk.Label(window, text="预测电力消耗: ")
    prediction_label.pack()

    rmse_var = tk.StringVar(value="RMSE: N/A")
    rmse_label = ttk.Label(window, textvariable=rmse_var)
    rmse_label.pack()

    waiting_label = ttk.Label(window, text="")
    waiting_label.pack()

    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    run_button = ttk.Button(window, text="运行 Server Encrypto", command=run_server_encrypto)
    run_button.pack()

    window.mainloop()


if __name__ == "__main__":
    create_gui()
