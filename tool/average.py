import pandas as pd
import numpy as np

def clean_duration_column(series):
    """
    辅助函数：清理并转换病程（Duration）列。
    该函数提取字符串（如 '10 Y', '0.5 N'）中的数字部分。
    """
    return pd.to_numeric(series.astype(str).str.split(' ').str[0], errors='coerce')

def generate_total_cohort_table(file_path):
    """
    读取患者数据文件，将所有数据视为一个总队列，并为其计算基线特征。

    参数:
    file_path (str): 指向您的 Excel (.xlsx) 或 CSV (.csv) 数据文件的路径。
    """
    # --- 1. 读取和准备数据 ---
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("不支持的文件格式，请使用 .xlsx 或 .csv 文件。")
        print(f"成功读取文件: {file_path}, 开始分析总共 {len(df)} 条记录。\n")
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请确保文件名正确且文件与脚本在同一文件夹中。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 预处理：统一列名，处理潜在的大小写问题
    df.columns = [col.strip().lower() for col in df.columns]

    # --- 2. 定义计算函数 (对整个队列进行计算) ---
    def calculate_stats(cohort_df):
        stats = {}
        n = len(cohort_df)

        # 中位数 (四分位距)
        for name, col in [('Age, years', 'year'), ('Duration, years', 'duration(y)')]:
            numeric_col = clean_duration_column(cohort_df[col]) if 'duration' in col else cohort_df[col]
            median = numeric_col.median()
            q1 = numeric_col.quantile(0.25)
            q3 = numeric_col.quantile(0.75)
            stats[name] = f"{median:.1f} ({q1:.1f} - {q3:.1f})"

        # 计数 (百分比) - 性别 (包含数据清理)
        cleaned_sex = cohort_df['sex'].astype(str).str.strip().str.upper()
        sex_counts = cleaned_sex.value_counts()
        stats['Female'] = f"{sex_counts.get('F', 0)} ({sex_counts.get('F', 0)/n*100:.1f}%)"
        stats['male'] = f"{sex_counts.get('M', 0)} ({sex_counts.get('M', 0)/n*100:.1f}%)"

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

    # --- 3. 计算并整理结果 ---
    total_stats = calculate_stats(df)

    # --- 4. 打印格式化的表格 ---
    n_total = len(df)
    header = f"| {'Characteristics':<55} | {'Total Cohort (n=' + str(n_total) + ')':<35} |"
    separator = f"|:{'-'*55}:|:{'-'*35}:|"
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
        '鼓室形成术 (Tympanoplasty)': '**手术类型 (Operation)**, n(%)',
        '平均气导听阈 (AC-PTA)': '**术前听力学评估 (Preoperative audiological evaluation)**'
    }

    for char_name in print_order:
        if char_name in grouped_items:
            print(f"| {grouped_items[char_name]:<55} | {'':<35} |")
        
        display_name = f"  {char_name}"
        if char_name in grouped_items:
            display_name = char_name

        print(f"| {display_name:<55} | {total_stats[char_name]:<35} |")

# --- 主程序入口 ---
if __name__ == "__main__":
    # #############################################################
    # ## 请在这里修改您的文件名 ##
    # #############################################################
    file_to_analyze = "tool/val.xlsx"  # <--- 修改这里

    generate_total_cohort_table(file_to_analyze)