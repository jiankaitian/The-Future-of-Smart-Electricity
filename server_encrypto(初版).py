import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import server_encrypto

current_round = 0  # 用于跟踪当前的轮数

# 更新 server_encrypto 中的 callback_function 以供 GUI 使用
def gui_callback_function(rmse_value, actual_values, predicted_values):
    global current_round
    # 由于更新GUI的操作需要在主线程中进行，我们使用window.after来安排更新
    def update_gui():
        global current_round
        # 更新 RMSE 显示和轮数
        rmse_var.set(f"Round {current_round}: RMSE: {rmse_value:.4f}")
        # 清除旧图表并绘制新图表
        ax.clear()
        ax.plot(actual_values, label='Actual', color='blue')
        ax.plot(predicted_values, label='Predicted', color='orange')
        ax.set(xlabel='Time', ylabel='Global Active Power', title=f'Round {current_round}: Actual vs Predicted Power Consumption')
        ax.legend()
        canvas.draw()

    current_round += 1  # 更新轮数
    window.after(0, update_gui)

# 确保 server_encrypto.callback_function 能够被正确设置
server_encrypto.callback_function = gui_callback_function

def run_server_encrypto():
    # 在新线程中运行 server_encrypto 的主逻辑，以避免阻塞 GUI
    global current_round
    current_round = 0  # 在开始新的运行前重置轮数
    def target():
        try:
            server_encrypto.main()
        except Exception as e:
            print(f"Error running server_encrypto.main: {e}")
    threading.Thread(target=target).start()

# 创建 GUI 窗口
window = tk.Tk()
window.title("Server Encrypto GUI")

# RMSE 标签
rmse_var = tk.StringVar(value="RMSE: N/A")
rmse_label = ttk.Label(window, textvariable=rmse_var)
rmse_label.pack()

# 图表区域
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=window)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# 运行按钮
run_button = ttk.Button(window, text="Run Server Encrypto", command=run_server_encrypto)
run_button.pack()

window.mainloop()