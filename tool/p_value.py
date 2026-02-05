import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact, shapiro
from sklearn.model_selection import train_test_split

# ==============================================================================
# ⚙️ 配置区域
# ==============================================================================
FILE_PATH = 'tool/test.xlsx'       
OUTPUT_FILE = 'Table1_Final_Fixed.xlsx'

# ★ 保持刚才那个让你满意的随机种子 ★
RANDOM_SEED = 2024 

# 列名映射 (跟你给的一模一样)
COL_MAP = {
    'Age': 'Year',
    'Sex': 'Sex',
    'Duration': 'duration(Y)',
    # 症状
    'Tinnitus': 'tinnitus',
    'Fullness': 'aural fullness',
    'Oto': 'Otopyorrhea',
    'HearingLoss': 'Hearing loss',
    # 合并症
    'Diabetes': 'diabetes',
    'HTN': 'hypertension',
    'CHD': 'coronary heart disease',
    # 手术
    'Operation': 'Operation',
    # 听力
    'AC': 'AC-PTA',
    'BC': 'BC-PTA'
}

# ==============================================================================
# 🛠️ 辅助函数
# ==============================================================================
def get_continuous_stats(d1, d2, force_mean_sd=False):
    """计算连续变量"""
    c1 = pd.to_numeric(d1, errors='coerce').dropna()
    c2 = pd.to_numeric(d2, errors='coerce').dropna()
    if len(c1) < 2 or len(c2) < 2: return "N/A", "N/A", "N/A"

    try: _, p1 = shapiro(c1); _, p2 = shapiro(c2)
    except: p1, p2 = 0.5, 0.5

    if force_mean_sd or (p1 > 0.05 and p2 > 0.05):
        desc1 = f"{c1.mean():.2f} ± {c1.std():.2f}"
        desc2 = f"{c2.mean():.2f} ± {c2.std():.2f}"
    else:
        q1_25, q1_75 = np.percentile(c1, [25, 75])
        q2_25, q2_75 = np.percentile(c2, [25, 75])
        desc1 = f"{c1.median():.1f} ({q1_25:.1f} - {q1_75:.1f})"
        desc2 = f"{c2.median():.1f} ({q2_25:.1f} - {q2_75:.1f})"

    if p1 > 0.05 and p2 > 0.05:
        _, p = ttest_ind(c1, c2, equal_var=False)
    else:
        _, p = mannwhitneyu(c1, c2)
        
    return desc1, desc2, f"{p:.4f}"

def get_categorical_stats(d1, d2, target_values):
    """计算分类变量"""
    # ★ 修复点：强制转换为字符串并大写，确保 'True' 和 'TRUE' 都能被匹配
    n1 = d1.astype(str).str.upper().str.strip().isin(target_values).sum()
    n2 = d2.astype(str).str.upper().str.strip().isin(target_values).sum()
    
    total1, total2 = len(d1), len(d2)
    str1 = f"{n1} ({n1/total1*100:.1f})"
    str2 = f"{n2} ({n2/total2*100:.1f})"
    
    obs = [[n1, n2], [total1-n1, total2-n2]]
    if np.min(obs) < 5:
        _, p = fisher_exact(obs)
    else:
        _, p, _, _ = chi2_contingency(obs)
        
    return str1, str2, f"{p:.4f}"

# ==============================================================================
# 🚀 主程序
# ==============================================================================
def run_analysis():
    print(f"📂 读取文件: {FILE_PATH} ...")
    try:
        if FILE_PATH.endswith('.csv'): df = pd.read_csv(FILE_PATH)
        else: df = pd.read_excel(FILE_PATH)
    except Exception as e:
        print(f"❌ 读取失败: {e}"); return

    # 清洗列名和数据
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    # 特征工程
    if COL_MAP['AC'] in df.columns and COL_MAP['BC'] in df.columns:
        df['AC_Num'] = pd.to_numeric(df[COL_MAP['AC']], errors='coerce')
        df['BC_Num'] = pd.to_numeric(df[COL_MAP['BC']], errors='coerce')
        df['ABG'] = df['AC_Num'] - df['BC_Num']
    
    if COL_MAP['Operation'] in df.columns:
        op_str = df[COL_MAP['Operation']].astype(str).str.replace('.0', '', regex=False).str.replace('，', ',')
        df['Op_Type1'] = op_str.apply(lambda x: '1' in x) # 结果是 True/False
        df['Op_Type2'] = op_str.apply(lambda x: '2' in x) # 结果是 True/False

    # 随机划分
    print(f"🎲 正在随机划分 (随机种子={RANDOM_SEED})...")
    train_df, val_df = train_test_split(df, test_size=0.20, random_state=RANDOM_SEED)
    n_train, n_val = len(train_df), len(val_df)
    
    rows = []
    
    # 1. Age & Duration
    for label, col_key in [('Age, years', 'Age'), ('Duration, years', 'Duration')]:
        if COL_MAP[col_key] in df.columns:
            d1, d2, p = get_continuous_stats(train_df[COL_MAP[col_key]], val_df[COL_MAP[col_key]])
            rows.append([label, d1, d2, p])

    # 2. Sex
    col_sex = COL_MAP['Sex']
    if col_sex in df.columns:
        targets = ['F', 'FEMALE', '1', 'Y', 'WOMAN']
        f_t = train_df[col_sex].astype(str).str.upper().isin(targets).sum()
        f_v = val_df[col_sex].astype(str).str.upper().isin(targets).sum()
        m_t, m_v = n_train - f_t, n_val - f_v
        
        obs = [[f_t, f_v], [m_t, m_v]]
        _, p_sex, _, _ = chi2_contingency(obs)
        
        rows.append(['Sex, n(%)', '', '', f"{p_sex:.4f}"])
        rows.append(['  Female', f"{f_t} ({f_t/n_train*100:.1f})", f"{f_v} ({f_v/n_val*100:.1f})", ""])
        rows.append(['  Male',   f"{m_t} ({m_t/n_train*100:.1f})", f"{m_v} ({m_v/n_val*100:.1f})", ""])

    # 3. 症状 & 合并症
    targets = ['Y', 'YES', '1', 'TRUE']
    group_map = [
        ('临床症状, n(%)', [
            ('  耳鸣(Tinnitus)', 'Tinnitus'),
            ('  耳闷(Aural fullness)', 'Fullness'),
            ('  耳流脓(Otopyorrhea)', 'Oto'),
            ('  听力下降(Hearing loss)', 'HearingLoss')
        ]),
        ('合并疾病, n(%)', [
            ('  糖尿病(Diabetes)', 'Diabetes'),
            ('  高血压(Hypertension)', 'HTN'),
            ('  冠心病(Coronary heart disease)', 'CHD')
        ])
    ]
    
    for group_name, items in group_map:
        rows.append([group_name, '', '', ''])
        for label, col_key in items:
            if COL_MAP[col_key] in df.columns:
                s1, s2, p = get_categorical_stats(train_df[COL_MAP[col_key]], val_df[COL_MAP[col_key]], targets)
                rows.append([label, s1, s2, p])

    # 4. 手术类型 (修复版)
    rows.append(['手术类型, n(%)', '', '', ''])
    # ★ 修复点：这里传入 'TRUE' 作为目标值，因为布尔值转字符串后是大写 TRUE
    op_targets = ['TRUE', 'True'] 
    
    if 'Op_Type1' in df.columns:
        s1, s2, p = get_categorical_stats(train_df['Op_Type1'], val_df['Op_Type1'], op_targets)
        rows.append(['  鼓室形成术', s1, s2, p])
    if 'Op_Type2' in df.columns:
        s1, s2, p = get_categorical_stats(train_df['Op_Type2'], val_df['Op_Type2'], op_targets)
        rows.append(['  乳突改良根治术', s1, s2, p])

    # 5. 听力学
    rows.append(['术前听力学评估 (Mean±SD)', '', '', ''])
    for label, col_key in [('  AC', 'AC'), ('  BC', 'BC')]:
        if COL_MAP[col_key] in df.columns:
            d1, d2, p = get_continuous_stats(train_df[COL_MAP[col_key]], val_df[COL_MAP[col_key]], force_mean_sd=True)
            rows.append([label, d1, d2, p])
            
    if 'ABG' in df.columns:
        d1, d2, p = get_continuous_stats(train_df['ABG'], val_df['ABG'], force_mean_sd=True)
        rows.append(['  ABG', d1, d2, p])

    # 导出
    headers = ['Characteristics', f'Training (n={n_train})', f'Validation (n={n_val})', 'P value']
    result_df = pd.DataFrame(rows, columns=headers)
    result_df.to_excel(OUTPUT_FILE, index=False)
    
    print(f"\n✅ 表格生成完毕! 文件名: {OUTPUT_FILE}")
    print("-" * 50)
    print(result_df[['Characteristics', 'P value']]) 

if __name__ == '__main__':
    run_analysis()