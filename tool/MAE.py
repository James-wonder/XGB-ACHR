import matplotlib.pyplot as plt
import numpy as np

# --- 在这里修改图表数据 (数据来自您的第一张原始图片) ---

# 设置全局字体 (建议使用 Arial, Helvetica 或其他清晰的无衬线字体)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none' # 在保存为SVG时将字体保存为文本

# 图表 (a) 的数据
title_a = 'Internal Validation (Without Ossicle Implant)'
values_a = [14.26, 11.69, 11.21, 11.18, 13.04, 12.35, 10.03]
avg_r2_a = 12.18

# 图表 (b) 的数据
title_b = 'External Validation (Without Ossicle Implant)'
values_b = [16.80, 20.50, 19.75, 22.50, 19.25, 27.00, 18.87]
avg_r2_b = 20.50

# 图表 (c) 的数据
title_c = 'Internal Validation (With Ossicle Implant)'
values_c = [11.26, 12.60, 12.43, 12.33, 11.75, 9.86, 10.56]
avg_r2_c = 11.60

# 图表 (d) 的数据
title_d = 'External Validation (With Ossicle Implant)'
values_d = [16.60, 19.24, 18.43, 20.08, 17.94, 17.60, 17.57]
avg_r2_d = 18.03

# --- 代码主体部分 ---

# X轴的类别标签
categories = ['0.25kHz', '0.5kHz', '1kHz', '2kHz', '4kHz', '8kHz', 'PTA']

# Nature风格的专业颜色
nature_colors = ['#336699', '#99CCEE', '#CC6677', '#336699', '#99CCEE', '#CC6677', '#99CCEE']

# 创建一个2x2的图表布局
fig, axs = plt.subplots(2, 2, figsize=(12, 9))

# --- 用于绘制单个图表的函数 (Nature期刊风格) ---
def create_nature_style_chart(ax, title, data_values, avg_r2, subplot_label):
    """在指定的坐标轴上绘制一个Nature风格的柱状图。"""
    
    # 绘制柱状图
    bars = ax.bar(categories, data_values, color=nature_colors, zorder=3)
    
    # 移除顶部和右侧的轴线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置坐标轴刻度线
    ax.tick_params(axis='both', which='major', direction='out', labelsize=10)
    
    # 添加水平网格线
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # 设置Y轴范围和标签
    ax.set_ylim(0, 30)
    ax.set_ylabel('Average MAE', fontsize=12)
    
    # 设置图表标题和X轴下方的平均R²值
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(f'Avg MAE = {avg_r2:.3f}', fontsize=12)
    
    # 绘制 "Good" 和 "Fair" 的水平虚线
    ax.axhline(y=10, color='dimgray', linestyle='--', linewidth=1, label='Good')
    ax.axhline(y=15, color='darkgray', linestyle='--', linewidth=1, label='Fair')
    
    # 设置图例
    ax.legend(frameon=False, loc='upper right')
    
    # 在每个柱子的顶部显示其具体数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=8.5, color='black')
                
    # 在图表的左上角添加带括号的子图标签
    ax.text(-0.15, 1.1, subplot_label, transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top')

# --- 调用函数来绘制四个图表 (已更正为带括号的标签) ---
create_nature_style_chart(axs[0, 0], title_a, values_a, avg_r2_a, 'a.')
create_nature_style_chart(axs[0, 1], title_b, values_b, avg_r2_b, 'b.')
create_nature_style_chart(axs[1, 0], title_c, values_c, avg_r2_c, 'c.')
create_nature_style_chart(axs[1, 1], title_d, values_d, avg_r2_d, 'd.')

# 调整布局以防止标签重叠
plt.tight_layout()

# 显示最终生成的图像
plt.show()