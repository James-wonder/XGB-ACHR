import pandas as pd
import numpy as np

def clean_duration_column(series):
    """
    辅助函数：清理并转换病程（Duration）列。
    """
    return pd.to_numeric(series.astype(str).str.split(' ').str[0], errors='coerce')

def generate_baseline_table_random_split(file_path, train_ratio=0.8, seed=42):
    """
    读取患者数据文件，按指定比例随机划分为训练和验证队列，并计算基线特征。
    """
    # --- 1. 读取和准备数据 ---
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("不支持的文件格式，请使用 .xlsx 或 .csv 文件。")
        print(f"成功读取文件: {file_path}, 共 {len(df)} 条记录。")
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请确保文件名正确且文件与脚本在同一文件夹中。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    df.columns = [col.strip().lower() for col in df.columns]

    # --- 2. 随机划分数据集 ---
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_total = len(df_shuffled)
    n_train = int(n_total * train_ratio)
    
    training_df = df_shuffled.iloc[:n_train].copy()
    validation_df = df_shuffled.iloc[n_train:].copy()
    
    print(f"数据已按 {int(train_ratio*100)}:{int((1-train_ratio)*100)} 比例随机划分为：训练队列 (n={len(training_df)}) 和 内部验证队列 (n={len(validation_df)})。\n")

    # --- 3. 定义计算函数 ---
    def calculate_stats(cohort_df):
        stats = {}
        n = len(cohort_df)

        # 中位数 (四分位距)
        for name, col in [('Age, years', 'year'), ('Duration, years', 'duration(y)')]:
            numeric_col = clean_duration_column(cohort_df[col]) if 'duration' in col else cohort_df[col]
            median, q1, q3 = numeric_col.median(), numeric_col.quantile(0.25), numeric_col.quantile(0.75)
            stats[name] = f"{median:.1f} ({q1:.1f} - {q3:.1f})"

        # #############################################################
        # ## 核心修正：在计数前清理性别数据 ##
        # #############################################################
        # 1. 移除前后空格，并统一转为大写，以应对 ' f ', 'F', ' f' 等情况
        cleaned_sex = cohort_df['sex'].astype(str).str.strip().str.upper()
        # 2. 在清理后的数据上进行计数
        sex_counts = cleaned_sex.value_counts()
        
        # 确保分母是总人数 n
        stats['Female'] = f"{sex_counts.get('F', 0)} ({sex_counts.get('F', 0)/n*100:.1f}%)"
        stats['male'] = f"{sex_counts.get('M', 0)} ({sex_counts.get('M', 0)/n*100:.1f}%)"
        
        # 检查总和是否为100%（允许微小浮点误差）
        total_sex_count = sex_counts.get('F', 0) + sex_counts.get('M', 0)
        if total_sex_count != n:
             print(f"警告：性别列计数 ({total_sex_count}) 与队列总人数 ({n}) 不符。请检查数据中是否存在 'F'/'M' 之外的值。")
        # #############################################################

        # 其他分类变量
        categorical_cols = {
            '耳鸣 (Tinnitus)': 'tinnitus', '耳闷 (Aural fullness)': 'aural fullness',
            '耳流脓 (Otopyorrhea)': 'otopyorrhea', '听力下降 (Hearing loss)': 'hearing loss',
            '糖尿病 (Diabetes)': 'diabetes', '高血压 (Hypertension)': 'hypertension',
            '冠心病 (Coronary heart disease)': 'coronary heart disease'
        }
        for display_name, col_name in categorical_cols.items():
            positive_count = cohort_df[col_name].astype(str).str.upper().isin(['Y', 'T']).sum()
            stats[display_name] = f"{positive_count} ({positive_count/n*100:.1f}%)"

        op_counts = cohort_df['operation'].astype(str).value_counts()
        op1_count = op_counts.get('1', 0) + op_counts.get('1.0', 0)
        op2_count = op_counts.get('2', 0) + op_counts.get('2.0', 0)
        stats['鼓室形成术 (Tympanoplasty)'] = f"{op1_count} ({op1_count/n*100:.1f}%)"
        stats['乳突改良根治术 (Modified radical mastoidectomy)'] = f"{op2_count} ({op2_count/n*100:.1f}%)"

        # 均值 ± 标准差
        stats['平均气导听阈 (AC-PTA)'] = f"{cohort_df['ac-pta'].mean():.2f} ± {cohort_df['ac-pta'].std():.2f}"
        stats['平均骨导听阈 (BC-PTA)'] = f"{cohort_df['bc-pta'].mean():.2f} ± {cohort_df['bc-pta'].std():.2f}"
        abg = cohort_df['ac-pta'] - cohort_df['bc-pta']
        stats['平均气骨导差 (Average ABG)'] = f"{abg.mean():.2f} ± {abg.std():.2f}"
        
        return stats

    # --- 4. 计算并打印 ---
    train_stats = calculate_stats(training_df)
    valid_stats = calculate_stats(validation_df)
    
    # (打印部分与之前相同，此处省略以保持简洁)
    # --- 5. 打印格式化的表格 (与之前版本相同) ---
    header = f"| {'Characteristics':<55} | {'Training cohort (n=' + str(len(training_df)) + ')':<35} | {'Internal Validation cohort (n=' + str(len(validation_df)) + ')':<40} |"
    separator = f"|:{'-'*55}:|:{'-'*35}:|:{'-'*40}:|"
    print(header)
    print(separator)

    print_order = [
        'Age, years', 'Female', 'male', 'Duration, years',
        '耳鸣 (Tinnitus)', '耳闷 (Aural fullness)', '耳流脓 (Otopyorrhea)', '听力下降 (Hearing loss)',
        '糖尿病 (Diabetes)', '高血压 (Hypertension)', '冠心病 (Coronary heart disease)',
        '鼓室形成术 (Tympanoplasty)', '乳突改良根治术 (Modified radical mastoidectomy)',
        '平均气导听阈 (AC-PTA)', '平均骨导听阈 (BC-PTA)', '平均气骨导差 (Average ABG)'
    ]
    
    grouped_items = {
        'Age, years': '**Age**, years', 'Female': '**Sex**, n(%)', 'Duration, years': '**Duration**, years',
        '耳鸣 (Tinnitus)': '**临床症状 (Clinical symptoms)**, n(%)', '糖尿病 (Diabetes)': '**合并疾病 (Comorbidities)**, n(%)',
        '手术类型 (Operation)':'**, n(%)',
        '平均气导听阈 (AC-PTA)': '**术前听力学评估 (Preoperative audiological evaluation)**'
    }

    for char_name in print_order:
        if char_name in grouped_items:
            print(f"| {grouped_items[char_name]:<55} | {'':<35} | {'':<40} |")
        
        display_name = f"  {char_name}" if char_name not in ['Age, years', 'Female', 'male', 'Duration, years'] and char_name not in grouped_items else char_name
        if char_name in ['Female', 'male']: display_name = f"  {char_name}"
        if char_name in grouped_items: display_name = char_name

        print(f"| {display_name:<55} | {train_stats[char_name]:<35} | {valid_stats[char_name]:<40} |")


# --- 主程序入口 ---
if __name__ == "__main__":
    file_to_analyze = "tool/val.xlsx"  # <--- 修改这里
    generate_baseline_table_random_split(file_to_analyze)