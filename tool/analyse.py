import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re # 引入正则库处理复杂字符串

# ==================== 配置 ====================
# 忽略警告
warnings.filterwarnings('ignore')

# 1. 先设置 Seaborn 风格 (必须在设置字体之前！)
sns.set_style("whitegrid")
sns.set_palette(["#3498db", "#e74c3c"]) 

# 2. 再设置中文字体
# 优先使用 'Microsoft YaHei' (微软雅黑)，通常在 Windows 上比 SimHei 更稳定
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial'] 

# 3. 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据加载与处理 (重点修改部分) ====================

def clean_column_names(df):
    """
    清洗列名专用函数：
    1. 去除所有空格 (解决 '0. 25' 问题)
    2. 统一 khz 为 kHz (解决大小写问题)
    3. 统一 pta 为 PTA
    4. 确保 AC/BC 大写
    """
    new_cols = []
    for col in df.columns:
        if col == 'ID':
            new_cols.append(col)
            continue
            
        # 1. 全部转小写处理，方便替换
        c = col.lower()
        
        # 2. 去除所有空格 (关键！解决 "0. 25" 变成 "0.25")
        c = c.replace(' ', '')
        
        # 3. 替换单位
        c = c.replace('khz', 'kHz')
        
        # 4. 恢复 AC/BC/PTA 的大写
        c = c.replace('ac-', 'AC-')
        c = c.replace('bc-', 'BC-')
        c = c.replace('pta', 'PTA')
        
        new_cols.append(c)
    
    df.columns = new_cols
    return df

def load_and_merge(base_path, label):
    """加载并合并 AC/BC 数据"""
    if 'val' in base_path:
        p1 = os.path.join(base_path, 'No implantation of ossicles/No ossicles implanted.xlsx')
        p2 = os.path.join(base_path, 'No implantation of ossicles/no ossicles implanted-postoperative hearing.xlsx')
    else:
        p1 = os.path.join(base_path, 'No ossicles implanted.xlsx')
        p2 = os.path.join(base_path, 'No ossicles implanted-postoperative hearing.xlsx')

    if not os.path.exists(p1) or not os.path.exists(p2):
        print(f"Warning: Files not found in {base_path}")
        return None

    # 读取 Excel
    df_pre = pd.read_excel(p1)
    df_post = pd.read_excel(p2)
    
    # ========== 核心修改：在合并前清洗列名 ==========
    # 这步操作会将 "AC-0. 25khz" 变成标准的 "AC-0.25kHz"
    df_pre = clean_column_names(df_pre)
    df_post = clean_column_names(df_post)
    # ============================================

    # 合并 (此时列名已经干净了，合并后会自动加上后缀)
    # 例如：AC-0.25kHz -> AC-0.25kHz_pre 和 AC-0.25kHz_post
    df = pd.merge(df_pre, df_post, on='ID', suffixes=('_pre','_post'))
    
    # 筛选听力相关的列 (只保留包含 AC/BC 且包含 pre/post 的列)
    cols = [c for c in df.columns if ('AC-' in c or 'BC-' in c) and ('_pre' in c or '_post' in c)]
    df_hearing = df[cols].copy()
    df_hearing['Dataset'] = label
    
    return df_hearing

def prep_long_format(df):
    """将宽表转换为长表"""
    df_melt = df.melt(id_vars=['Dataset'], var_name='Metric', value_name='dB')
    
    # 解析列名逻辑: AC-0.25kHz_pre (因为我们在前面清洗过了，这里格式是固定的)
    def parse(s):
        try:
            type_ = s.split('-')[0]     # AC
            remain = s.split('-')[1]    # 0.25kHz_pre
            freq = remain.split('_')[0] # 0.25kHz
            time = remain.split('_')[1] # pre
            return type_, freq, time
        except:
            return None, None, None
    
    parsed = df_melt['Metric'].apply(parse)
    df_melt['Type'] = [x[0] for x in parsed]
    df_melt['Freq'] = [x[1] for x in parsed]
    df_melt['Time'] = [x[2] for x in parsed]
    
    # 过滤掉解析错误的行
    df_melt = df_melt.dropna(subset=['Freq'])
    
    # 定义频率顺序 (必须和大写的 kHz 匹配)
    freq_order = ['0.25kHz', '0.5kHz', '1kHz', '2kHz', '4kHz', '8kHz', 'PTA']
    df_melt['Freq'] = pd.Categorical(df_melt['Freq'], categories=freq_order, ordered=True)
    
    return df_melt

# ==================== 2. 三种绘图函数 ====================

def plot_1_audiogram(df_long, save_folder):
    """生成听力图风格均值对比图"""
    print(">>> 正在生成图 1: 听力曲线对比 (Line Plot)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    scenarios = [('AC', 'pre'), ('AC', 'post'), ('BC', 'pre'), ('BC', 'post')]
    
    for i, (ctype, ctime) in enumerate(scenarios):
        ax = axes[i//2, i%2]
        data = df_long[(df_long['Type'] == ctype) & (df_long['Time'] == ctime)]
        
        if len(data) == 0: continue
        
        sns.pointplot(data=data, x='Freq', y='dB', hue='Dataset', 
                      markers=['o', '^'], linestyles=['-', '--'],
                      errorbar='sd', capsize=0.1, ax=ax)
        
        ax.set_title(f"{ctype} {ctime} (均值 ± 标准差)", fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.set_ylabel("Hearing Level (dB HL)")
        ax.set_xlabel("Frequency")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title=None)
        
    plt.suptitle("图1：听力曲线均值对比 (实线=内部, 虚线=外部)", fontsize=20, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "1_Audiogram_Comparison.png"), dpi=300)
    plt.close()

def plot_2_boxplot(df_long, save_folder):
    """生成箱线图"""
    print(">>> 正在生成图 2: 箱线图对比 (Box Plot)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    scenarios = [('AC', 'pre'), ('AC', 'post'), ('BC', 'pre'), ('BC', 'post')]
    
    for i, (ctype, ctime) in enumerate(scenarios):
        ax = axes[i//2, i%2]
        data = df_long[(df_long['Type'] == ctype) & (df_long['Time'] == ctime)]
        
        if len(data) == 0: continue

        sns.boxplot(data=data, x='Freq', y='dB', hue='Dataset', ax=ax, width=0.6)
        
        ax.set_title(f"{ctype} {ctime} 分布箱线图", fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.set_ylabel("Hearing Level (dB HL)")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(title=None)

    plt.suptitle("图2：听力值离散程度对比 (Box Plot)", fontsize=20, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "2_Boxplot_Comparison.png"), dpi=300)
    plt.close()

def plot_3_kde_all(df_wide, save_folder):
    """生成所有频率的 KDE 密度图"""
    print(">>> 正在生成图 3: 核密度估计图 (KDE Plot)...")
    
    cols = [c for c in df_wide.columns if c != 'Dataset' and ('AC-' in c or 'BC-' in c)]
    
    # 这里的 sort_key 现在可以很简单，因为列名已经标准化为 0.25kHz 了
    def sort_key(col_name):
        s_type = 0 if 'AC-' in col_name else 100
        s_time = 0 if '_pre' in col_name else 10
        
        # 提取频率数字
        freq_part = col_name.split('-')[1].split('_')[0] # 如 0.25kHz
        
        if 'PTA' in freq_part:
            s_freq = 99
        else:
            # 去掉 kHz 转数字
            try:
                s_freq = float(freq_part.replace('kHz', ''))
            except:
                s_freq = 999
        return s_type + s_time + s_freq

    sorted_cols = sorted(cols, key=sort_key)
    
    for main_type in ['AC', 'BC']:
        sub_cols = [c for c in sorted_cols if f'{main_type}-' in c]
        if not sub_cols: continue
        
        n = len(sub_cols)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
        if nrows * ncols == 1: axes = np.array([axes]) # 防止单个子图报错
        else: axes = axes.flatten()
        
        internal = df_wide[df_wide['Dataset'] == 'Internal']
        external = df_wide[df_wide['Dataset'] == 'External']
        
        for i, col in enumerate(sub_cols):
            ax = axes[i]
            sns.kdeplot(internal[col], fill=True, color="#3498db", alpha=0.3, label='Internal', ax=ax)
            sns.kdeplot(external[col], fill=True, color="#e74c3c", alpha=0.3, label='External', ax=ax)
            
            ax.set_title(col, fontsize=12, fontweight='bold')
            ax.set_xlabel("dB HL")
            ax.set_ylabel("Density")
            if i == 0: ax.legend()
            
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            
        plt.suptitle(f"图3-{main_type}：{main_type} 全频率分布密度对比", fontsize=20, y=0.99)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"3_KDE_Comparison_{main_type}.png"), dpi=300)
        plt.close()

# ==================== 3. 主流程 ====================
def run_full_comparison():
    # 路径 (根据你的实际情况修改)
    BASE_PATH = 'no implanted/data/'
    VAL_PATH = 'no implanted/val/data/'
    SAVE_FOLDER = "All_Hearing_Comparisons"
    
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    print(f">>> 结果将保存在文件夹: {SAVE_FOLDER}")
    
    # 1. 加载
    df_in = load_and_merge(BASE_PATH, 'Internal')
    df_ex = load_and_merge(VAL_PATH, 'External')
    
    if df_in is None or df_ex is None:
        print("Error: 数据加载失败")
        return

    # 2. 合并宽表
    df_wide = pd.concat([df_in, df_ex], axis=0, ignore_index=True)
    
    # 3. 转换为长表
    df_long = prep_long_format(df_wide)
    
    # 4. 生成图
    plot_1_audiogram(df_long, SAVE_FOLDER)
    plot_2_boxplot(df_long, SAVE_FOLDER)
    plot_3_kde_all(df_wide, SAVE_FOLDER)
    
    print("\n>>> 全部完成！请查看文件夹 'All_Hearing_Comparisons'")

if __name__ == "__main__":
    run_full_comparison()