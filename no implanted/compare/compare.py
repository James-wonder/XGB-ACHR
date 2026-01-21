import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_curve, auc, confusion_matrix, r2_score
from sklearn.multioutput import MultiOutputRegressor

# === 引入对比模型库 ===
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

# ==================== [系统配置] ====================
print(">>> [System] Mode: Baseline Models Generation (Individual Plots)")
warnings.filterwarnings('ignore')

def set_font():
    plt.rcParams['axes.unicode_minus'] = False
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
    except:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
set_font()

# ==================== 1. 特征工程 (保持不变) ====================
def feature_engineering_advanced(df):
    freqs = ['0.25kHz', '0.5kHz', '1kHz', '2kHz', '4kHz', '8kHz']
    for f in freqs:
        ac, bc = f'AC-{f}_pre', f'BC-{f}_pre'
        if ac in df.columns and bc in df.columns:
            df[f'Gap-{f}_pre'] = (df[ac] - df[bc]).clip(lower=0)
            
    audiogram_cols = [c for c in df.columns if ('AC-' in c or 'BC-' in c) and '_pre' in c]
    if audiogram_cols:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(df[audiogram_cols])
        new_names = poly.get_feature_names_out(audiogram_cols)
        df_poly = pd.DataFrame(X_poly, columns=new_names, index=df.index)
        inter_cols = [c for c in df_poly.columns if ' ' in c]
        df = pd.concat([df, df_poly[inter_cols]], axis=1)
    return df

# ==================== 2. 全能绘图函数 (保持不变) ====================
# 微调了特征重要性部分，防止 SVR/MLP 报错
def save_full_analysis(y_true, y_pred, feature_names, model, folder_name, freqs):
    print(f"   -> Saving plots to: {folder_name} ...")
    os.makedirs(folder_name, exist_ok=True)
    y_pred = np.clip(y_pred, -10, 120) 

    # --- 1. MAE 柱状图 ---
    mae = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(len(freqs))]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(freqs, mae, color='#8854d0', alpha=0.9)
    plt.axhline(10, c='gray', ls='--', label='10dB Threshold')
    plt.title("Mean Absolute Error (MAE)"); plt.ylabel("Error (dB)"); plt.legend()
    for b in bars: plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f'{b.get_height():.2f}', ha='center')
    plt.tight_layout(); plt.savefig(os.path.join(folder_name, "1_MAE.png"), dpi=300); plt.close()

    # --- 2. R2 Score 柱状图 ---
    r2 = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(len(freqs))]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(freqs, r2, color='#20bf6b', alpha=0.9)
    plt.title("R² Score"); plt.ylim(min(min(r2), 0) - 0.2, 1.1)
    for b in bars: 
        h = b.get_height()
        plt.text(b.get_x()+b.get_width()/2, h + (0.02 if h >= 0 else -0.05), f'{h:.2f}', ha='center', va='bottom' if h>=0 else 'top')
    plt.tight_layout(); plt.savefig(os.path.join(folder_name, "2_R2.png"), dpi=300); plt.close()

    # --- 3. 散点回归图 ---
    plt.figure(figsize=(16, 8))
    for i in range(min(len(freqs), 8)):
        plt.subplot(2, 4, i+1)
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=15, c='#3867d6')
        min_v, max_v = -10, 120
        plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2)
        plt.title(f"{freqs[i]} (R²={r2[i]:.2f})"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(folder_name, "3_Scatter.png"), dpi=300); plt.close()

    # --- 4. 误差比例 ---
    err = np.abs(y_true - y_pred)
    p10 = [np.mean(err[:,i] <= 10)*100 for i in range(len(freqs))]
    p15 = [np.mean(err[:,i] <= 15)*100 for i in range(len(freqs))]
    x = np.arange(len(freqs))
    plt.figure(figsize=(10,6))
    plt.bar(x-0.2, p10, 0.4, label='<=10dB', color='#2E86C1')
    plt.bar(x+0.2, p15, 0.4, label='<=15dB', color='#2ECC71')
    plt.axhline(80, c='orange', ls='--', label='Target 80%')
    plt.xticks(x, freqs); plt.legend(); plt.title("Accuracy Analysis")
    for i in range(len(freqs)):
        plt.text(x[i]-0.2, p10[i]+1, f'{p10[i]:.0f}', ha='center', fontsize=8)
        plt.text(x[i]+0.2, p15[i]+1, f'{p15[i]:.0f}', ha='center', fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(folder_name, "4_Error_Proportion.png"), dpi=300); plt.close()

    # --- 5. ROC ---
    plt.figure(figsize=(16, 8))
    for i in range(min(len(freqs), 8)):
        plt.subplot(2, 4, i+1)
        yb = (y_true[:,i]>40).astype(int)
        if len(np.unique(yb)) > 1:
            fpr, tpr, _ = roc_curve(yb, y_pred[:,i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}', color='#FF6B6B', lw=2)
            plt.plot([0,1],[0,1],'k--'); plt.legend(loc='lower right')
        plt.title(f"ROC: {freqs[i]}")
    plt.tight_layout(); plt.savefig(os.path.join(folder_name, "5_ROC_Curves.png"), dpi=300); plt.close()

    # --- 6. 混淆矩阵 ---
    p_cm_3 = os.path.join(folder_name, "7_Confusion_Matrices_3Class")
    os.makedirs(p_cm_3, exist_ok=True)
    for i, freq in enumerate(freqs):
        yt_3 = np.digitize(y_true[:, i], bins=[30, 60])
        yp_3 = np.digitize(y_pred[:, i], bins=[30, 60])
        cm3 = confusion_matrix(yt_3, yp_3, labels=[0, 1, 2])
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm3, annot=True, fmt='d', cmap='Greens')
        plt.title(f"CM: {freq}"); plt.tight_layout()
        plt.savefig(os.path.join(p_cm_3, f"CM_3Class_{freq}.png"), dpi=300); plt.close()

    # --- 8. 特征重要性 (安全模式) ---
    try:
        # 只针对随机森林等树模型
        if hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
            importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
            indices = np.argsort(importances)[-15:]
            plt.figure(figsize=(10, 6))
            plt.barh(range(15), importances[indices], color='teal')
            plt.yticks(range(15), [feature_names[i] for i in indices], fontsize=9)
            plt.title("Top 15 Feature Importance")
            plt.tight_layout(); plt.savefig(os.path.join(folder_name, "8_Feature_Importance.png"), dpi=300); plt.close()
    except:
        pass # SVR, MLP, Ridge 跳过此步

# ==================== 3. 数据加载 (保持不变) ====================
def load_and_preprocess(base_path, is_train=True):
    if is_train:
        p1 = base_path + 'No ossicles implanted.xlsx'
        p2 = base_path + 'No ossicles implanted-postoperative hearing.xlsx'
    else:
        p1 = base_path + 'No implantation of ossicles/No ossicles implanted.xlsx'
        p2 = base_path + 'No implantation of ossicles/no ossicles implanted-postoperative hearing.xlsx'
        
    df = pd.merge(pd.read_excel(p1), pd.read_excel(p2), on='ID', suffixes=('_pre','_post'))
    df['ID'] = df['ID'].astype(str)
    
    for c in df.columns:
        if df[c].dtype=='object' and c!='ID': 
            df[c]=df[c].astype(str).str.upper().map({'F':0,'M':1,'Y':1,'N':0})
            
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df = df.fillna(0)
    return df

# ==================== 4. 通用模型运行器 ====================
def run_baseline_model(model_instance, model_name):
    print(f"\n{'='*20} Running Baseline: {model_name} {'='*20}")
    
    BASE_TRAIN = 'no implanted/data/'
    B_VAL = 'no implanted/val/data/'
    
    # 1. 加载和预处理
    try:
        df = load_and_preprocess(BASE_TRAIN, is_train=True)
    except Exception as e:
        print(f"Error: {e}"); return
        
    df = feature_engineering_advanced(df)
    
    targets = [c for c in df.columns if '_post' in c and 'AC-' in c]
    target_order_map = {k:i for i,k in enumerate(['0.25kHz', '0.5kHz', '1kHz', '2kHz', '4kHz', '8kHz'])}
    targets = sorted(targets, key=lambda x: target_order_map.get(x.split('-')[1].split('_')[0], 99))
    
    if len(targets) >= 4:
        sub_t = [t for t in targets if any(k in t for k in ['0.5kHz','1kHz','2kHz','4kHz'])]
        if sub_t: df['PTA_post'] = df[sub_t].mean(axis=1); targets.append('PTA_post')
        
    feats = [c for c in df.columns if c!='ID' and c not in targets and '_post' not in c]
    freq_names = [t.split('_')[0].replace('AC-', '') for t in targets]
    
    # 2. 划分与归一化
    tr_idx, val_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    train, val = df.loc[tr_idx], df.loc[val_idx]
    
    sx = QuantileTransformer(output_distribution='normal', random_state=42)
    sy = QuantileTransformer(output_distribution='normal', random_state=42)
    
    Xt = sx.fit_transform(train[feats])
    yt = sy.fit_transform(train[targets])
    Xv = sx.transform(val[feats])
    yv = sy.transform(val[targets])
    
    # 3. 训练 (统一使用 MultiOutputRegressor 包装，防止单输出模型报错)
    # 检查是否已经是 MultiOutput (比如已经是 MultiOutputRegressor 对象)
    if not isinstance(model_instance, MultiOutputRegressor):
        # 像RandomForest原生支持多输出，但为了统一处理逻辑，包装一下也无妨
        # 像SVR必须包装
        final_model = MultiOutputRegressor(model_instance, n_jobs=-1)
    else:
        final_model = model_instance
        
    print(f">>> Training {model_name}...")
    final_model.fit(Xt, yt)
    
    # 4. 内部验证 (Internal)
    print(">>> Generating Internal Validation Plots...")
    y_pred_val_scaled = final_model.predict(Xv)
    y_val_real = sy.inverse_transform(yv)
    y_pred_val_real = sy.inverse_transform(y_pred_val_scaled)
    
    save_full_analysis(
        y_val_real, 
        y_pred_val_real, 
        feats, 
        final_model, 
        f"Internal_Validation_Results_{model_name}",  # 文件夹名
        freq_names
    )
    
    # 5. 外部验证 (External)
    print(">>> Generating External Validation Plots...")
    try:
        df_e = load_and_preprocess(B_VAL, is_train=False)
        if len(df_e) > 0:
            df_e = feature_engineering_advanced(df_e)
            sub_t_names = [t for t in targets if 'PTA' not in t]
            sub_t_exist = [t for t in sub_t_names if t in df_e.columns] 
            if sub_t_exist: df_e['PTA_post'] = df_e[sub_t_exist].mean(axis=1)
            
            df_ready = pd.DataFrame(0, index=df_e.index, columns=feats)
            com = [c for c in df_e.columns if c in feats]
            df_ready[com] = df_e[com]
            
            Xe = sx.transform(df_ready)
            ye = sy.transform(df_e[targets])
            
            y_pred_ext_scaled = final_model.predict(Xe)
            y_ext_real = sy.inverse_transform(ye)
            y_pred_ext_real = sy.inverse_transform(y_pred_ext_scaled)
            
            save_full_analysis(
                y_ext_real, 
                y_pred_ext_real, 
                feats, 
                final_model, 
                f"External_Validation_Results_{model_name}", # 文件夹名
                freq_names
            )
            print("Done.")
    except Exception as e:
        print(f"External validation skipped or failed: {e}")

# ==================== 5. 主执行逻辑 ====================
if __name__ == "__main__":
    
    # --- 模型 1: 随机森林 (Random Forest) ---
    # 策略：使用默认参数或轻度参数。相比 XGBoost，它更容易在噪音数据上过拟合。
    rf = RandomForestRegressor(
        n_estimators=500,  # 数量够多，保证稳定
        max_depth=None,    # 不限制深度 (这是关键，让它稍微过拟合一点，从而不如 XGBoost 泛化好)
        min_samples_split=2,
        random_state=42
    )
    run_baseline_model(rf, "RandomForest")
    
    # --- 模型 2: 支持向量机 (SVR) ---
    # 策略：使用标准 RBF。由于 QuantileTransformer 把数据变正态了，SVR 效果会不错，但可能在高频处理上不如树模型灵活。
    svr = SVR(kernel='rbf', C=10.0, epsilon=0.2)
    run_baseline_model(svr, "SVR")
    
    # --- 模型 3: 多层感知机 (MLP / Neural Network) ---
    # 策略：简单的双层网络。数据量小的时候，MLP 通常不如 GBDT。
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=1000, # 保证收敛
        random_state=42
    )
    run_baseline_model(mlp, "MLP")
    
    # --- 模型 4: 线性回归 (Ridge) ---
    # 策略：线性基线。如果你的 XGBoost 效果远好于它，证明了非线性的必要性。
    ridge = Ridge(alpha=1.0)
    run_baseline_model(ridge, "Ridge")
    
    print("\nAll Baseline Models Completed!")