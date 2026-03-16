import matplotlib.pyplot as plt
import numpy as np

# --- 1. 颜色定义 (补全了缺失的模型颜色) ---
colors = {
    'XGB-ACHR': "#898988",          # 深蓝色
    'Random Forest': "#79cb9b",           # 柔和蓝
    'MLP': "#ffc48a",     # 浅蓝色             # 橙色
    'Liner Regression': "#547ac0" # 绿色 (保留了您数据中的原始拼写)
}

plt.rcParams['font.sans-serif'] = 'Arial'
BACKGROUND_COLOR ="#FFFFFF"
FONT_COLOR = '#333333'

# --- 2. 数据准备 ---
plot_data = {
    'ax1': {
        'title': 'Without OCR',
        'models': ['XGB-ACHR', 'Random Forest', 'MLP','Liner Regression'],
        'values': [12.18, 12.21, 16.12, 13.53],
        'ylabel': 'Internal Validation Cohort'
    },
    'ax2': {
        'title': 'With OCR',
        'models': ['XGB-ACHR', 'Random Forest', 'MLP','Liner Regression'],
        'values': [11.60, 10.87,14.81,11.76],
        'ylabel': None
    },
    'ax3': {
        'title': None,
        'models': ['XGB-ACHR', 'Random Forest', 'MLP','Liner Regression'],
        'values': [20.50,21.37,23.46,21.69],
        'ylabel': 'External Validation Cohort'
    },
    'ax4': {
        'title': None,
        'models': ['XGB-ACHR', 'Random Forest', 'MLP','Liner Regression'],
        'values': [18.03,17.91,23.55,17.28],
        'ylabel': None
    }
}

# --- 3. 创建画布 ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # 高度稍微减小一点点使其紧凑
fig.set_facecolor(BACKGROUND_COLOR)

# --- 4. 循环绘图 ---
axes_flat = axes.flatten()
plot_keys = ['ax1', 'ax2', 'ax3', 'ax4']

for i, ax in enumerate(axes_flat):
    key = plot_keys[i]
    data = plot_data[key]
    
    models = data['models']
    values = data['values']
    y_pos = np.arange(len(models))
    
    ax.set_facecolor(BACKGROUND_COLOR)

    # 绘制水平条形图
    bars = ax.barh(y_pos, values, height=0.75, color=[colors[model] for model in models], zorder=3)

    # --- 核心修改：添加数值标签 ---
    for bar in bars:
        width = bar.get_width()
        ax.text(
            # x坐标：柱子总长度减去一个偏移量 (0.5)，确保文字在柱子"里面"
            width - 0.5,             
            # y坐标：柱子的中心高度
            bar.get_y() + bar.get_height() / 2, 
            # 显示的文字：保留2位小数
            f'{width:.2f}',          
            # 对齐方式：Right (右对齐)，意味着文字的右边缘对齐到上面的x坐标
            ha='right',              
            va='center',             
            color='white',           # 白色文字以显示在深色柱子上
            fontweight='bold',       
            fontsize=10              
        )

    # --- 样式美化 ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    
    if data['title']:
        ax.set_title(data['title'], fontsize=14, fontweight='bold', pad=15, color=FONT_COLOR)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11, color=FONT_COLOR)
    ax.invert_yaxis() # 反转Y轴，让第一个模型在最上面

    ax.set_xticks([]) # 隐藏X轴刻度（因为已经有数值显示了）
    
    if data['ylabel']:
        ax.set_ylabel(data['ylabel'], fontsize=12, labelpad=15, color=FONT_COLOR, fontweight='bold')
        
    ax.set_xlabel('Average MAE Score', fontsize=11, labelpad=8, color=FONT_COLOR)

# --- 5. 调整布局 ---
plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.1, hspace=0.3, wspace=0.2)

plt.show()