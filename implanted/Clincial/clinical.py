import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, roc_curve, auc, confusion_matrix, r2_score
from sklearn.multioutput import MultiOutputRegressor

# ==================== [系统配置] ====================
print(">>> [System] Mode: Clinical Data Only (Optimized Params & Imputation)")

warnings.filterwarnings('ignore')

def set_font():
    plt.rcParams['axes.unicode_minus'] = False
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
    except:
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        except:
            pass
set_font()

# ==================== 1. 特征工程 (保持不变) ====================
def feature_engineering_advanced(df):
    # 1. Gap (气骨导差)
    freqs = ['0.25kHz', '0.5kHz', '1kHz', '2kHz', '4kHz', '8kHz']
    for f in freqs:
        ac, bc = f'AC-{f}_pre', f'BC-{f}_pre'
        if ac in df.columns and bc in df.columns:
            df[f'Gap-{f}_pre'] = (df[ac] - df[bc]).clip(lower=0)
            
    # 2. 交互特征 (Polynomial Features)
    audiogram_cols = [c for c in df.columns if ('AC-' in c or 'BC-' in c) and '_pre' in c]
    if audiogram_cols:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(df[audiogram_cols])
        new_names = poly.get_feature_names_out(audiogram_cols)
        
        # 避免列名重复或过长，可选择性保留，这里保持逻辑一致
        df_poly = pd.DataFrame(X_poly, columns=new_names, index=df.index)
        inter_cols = [c for c in df_poly.columns if ' ' in c]
        df = pd.concat([df, df_poly[inter_cols]], axis=1)
        
    return df

# ==================== 2. 全能绘图函数 (保持不变) ====================
def save_full_analysis(y_true, y_pred, feature_names, model, folder_name, freqs):
    print(f"\n>>> Generating Analysis Plots in: {folder_name} ...")
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
        min_v, max_v = min(y_true[:, i].min(), y_pred[:, i].min()), max(y_true[:, i].max(), y_pred[:, i].max())
        plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2)
        plt.title(f"{freqs[i]} (R²={r2[i]:.2f})"); plt.xlabel("True"); plt.ylabel("Pred"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(folder_name, "3_Scatter.png"), dpi=300); plt.close()

    # --- 4. 误差比例分析 ---
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

    # --- 5. ROC 曲线 ---
    plt.figure(figsize=(16, 8))
    roc_data_list = [] # 用于存储每个频率的DataFrame
    
    for i in range(min(len(freqs), 8)):
        plt.subplot(2, 4, i+1)
        yb = (y_true[:,i]>40).astype(int)
        
        # 判断是否有两类样本，否则无法计算ROC
        if len(np.unique(yb)) > 1:
            fpr, tpr, _ = roc_curve(yb, y_pred[:,i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}', color='#FF6B6B', lw=2)
            plt.plot([0,1],[0,1],'k--'); plt.legend(loc='lower right')
            
            # [新增代码] 收集坐标点数据
            # 创建临时DataFrame，列名为该频率的FPR和TPR
            temp_df = pd.DataFrame({
                f'{freqs[i]}_FPR': fpr,
                f'{freqs[i]}_TPR': tpr
            })
            roc_data_list.append(temp_df)
            
        else: 
            plt.text(0.5, 0.5, "Single Class", ha='center')
        
        plt.title(f"ROC: {freqs[i]}")
    
    plt.tight_layout(); plt.savefig(os.path.join(folder_name, "5_ROC_Curves.png"), dpi=300); plt.close()

    # [新增代码] 将所有频率的ROC坐标保存到一个Excel文件中
    if roc_data_list:
        # axis=1 横向拼接，因为不同频率的ROC点数不一样，会自动补NaN
        all_roc_data = pd.concat(roc_data_list, axis=1)
        roc_excel_path = os.path.join(folder_name, "5_ROC_Coordinates.xlsx")
        all_roc_data.to_excel(roc_excel_path, index=False)
        print(f"ROC coordinates saved to: {roc_excel_path}")

    # --- 6. 敏感性 & 特异性 ---
    plt.figure(figsize=(16, 8))
    for i in range(min(len(freqs), 8)):
        plt.subplot(2, 4, i+1)
        yb = (y_true[:, i] > 40).astype(int)
        if len(np.unique(yb)) > 1:
            fpr, tpr, thresh = roc_curve(yb, y_pred[:, i])
            spec = 1 - fpr
            plt.plot(thresh, tpr, 'b', label='Sens'); plt.plot(thresh, spec, 'r', label='Spec')
            idx = np.argmin(np.abs(tpr - spec))
            plt.scatter(thresh[idx], tpr[idx], c='g'); plt.text(thresh[idx], tpr[idx], f'{thresh[idx]:.1f}', fontsize=8)
        plt.title(freqs[i]); plt.grid(True, alpha=0.3)
        if i==0: plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(folder_name, "6_Sens_Spec.png"), dpi=300); plt.close()

    # --- 7. 混淆矩阵 ---
    p_cm_2 = os.path.join(folder_name, "7_Confusion_Matrices_Binary")
    p_cm_3 = os.path.join(folder_name, "7_Confusion_Matrices_3Class")
    os.makedirs(p_cm_2, exist_ok=True)
    os.makedirs(p_cm_3, exist_ok=True)
    
    for i in range(len(freqs)):
        freq = freqs[i]
        
        # (A) 二分类 (<40, >=40)
        yt_2 = (y_true[:, i] >= 40).astype(int)
        yp_2 = (y_pred[:, i] >= 40).astype(int)
        cm2 = confusion_matrix(yt_2, yp_2, labels=[0, 1])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=['<40dB','>=40dB'], yticklabels=['<40dB','>=40dB'])
        plt.title(f"Binary CM: {freq}"); plt.tight_layout()
        plt.savefig(os.path.join(p_cm_2, f"CM_Binary_{freq}.png"), dpi=300); plt.close()
        
        # (B) 三分类 (<30, 30-60, >60)
        yt_3 = np.digitize(y_true[:, i], bins=[30, 60])
        yp_3 = np.digitize(y_pred[:, i], bins=[30, 60])
        cm3 = confusion_matrix(yt_3, yp_3, labels=[0, 1, 2])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm3, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=['<30', '30-60', '>60'], 
                    yticklabels=['<30', '30-60', '>60'])
        plt.title(f"3-Class CM: {freq}"); plt.tight_layout()
        plt.savefig(os.path.join(p_cm_3, f"CM_3Class_{freq}.png"), dpi=300); plt.close()

    # --- 8. 特征重要性 ---
    if hasattr(model, 'estimators_') and model.estimators_[0].feature_importances_ is not None:
        importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
        indices = np.argsort(importances)[-15:]
        plt.figure(figsize=(10, 6))
        plt.barh(range(15), importances[indices], color='teal')
        plt.yticks(range(15), [feature_names[i] for i in indices], fontsize=9)
        plt.title("Top 15 Feature Importance")
        plt.tight_layout(); plt.savefig(os.path.join(folder_name, "8_Feature_Importance.png"), dpi=300); plt.close()

# ==================== 3. 数据加载与预处理 (优化版) ====================
def load_and_preprocess(base_path, is_train=True):
    # 路径处理
    if is_train:
        p1 = base_path + 'Implantation of ossicular information.xlsx'
        p2 = base_path + 'ossicles implanted-postoperative hearing.xlsx'
    else:
        p1 = base_path + 'Implantation of ossicular information.xlsx'
        p2 = base_path + 'ossicles implanted-postoperative hearing.xlsx'
        
    df = pd.merge(pd.read_excel(p1), pd.read_excel(p2), on='ID', suffixes=('_pre','_post'))
    df['ID'] = df['ID'].astype(str)
    
    # 文本转数值
    for c in df.columns:
        if df[c].dtype=='object' and c!='ID': 
            df[c]=df[c].astype(str).str.upper().map({'F':0,'M':1,'Y':1,'N':0})
            # 注意：map 后如果是 NaN 这里先不处理，留给后面统一处理
            
    # [优化 1] 缺失值填充策略：数值型用中位数，分类型用0
    # 先分离数值列和非数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # 使用中位数填充数值列 (防止听力数据0dB误导)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    # 剩下的（主要是分类型未能匹配到的）填0
    df = df.fillna(0)
    
    return df

# ==================== 4. 主流程 ====================
def run_clinical_only():
    BASE_TRAIN = 'implanted/data/'
    
    # 1. 加载数据
    print(">>> 加载数据 (Clinical)...")
    try:
        df = load_and_preprocess(BASE_TRAIN, is_train=True)
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    if len(df) == 0: raise ValueError("Data Empty!")

    # 特征工程
    df = feature_engineering_advanced(df)
    
    # 确定目标列
    targets = [c for c in df.columns if '_post' in c and 'AC-' in c]
    target_order_map = {k:i for i,k in enumerate(['0.25kHz', '0.5kHz', '1kHz', '2kHz', '4kHz', '8kHz'])}
    targets = sorted(targets, key=lambda x: target_order_map.get(x.split('-')[1].split('_')[0], 99))
    
    # 计算 PTA
    if len(targets) >= 4:
        sub_t = [t for t in targets if any(k in t for k in ['0.5kHz','1kHz','2kHz','4kHz'])]
        if sub_t:
            df['PTA_post'] = df[sub_t].mean(axis=1)
            targets.append('PTA_post')
        
    # 特征列
    feats = [c for c in df.columns if c!='ID' and c not in targets and '_post' not in c]
    
    # 划分训练集和验证集
    tr_idx, val_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    train = df.loc[tr_idx].copy()
    val = df.loc[val_idx].copy()
    
    print(f"\n[Clinical Features Only] Count: {len(feats)}")
    
    # 数据归一化
    sx = QuantileTransformer(output_distribution='normal', random_state=42)
    sy = QuantileTransformer(output_distribution='normal', random_state=42)
    
    Xt = sx.fit_transform(train[feats])
    yt = sy.fit_transform(train[targets])
    Xv = sx.transform(val[feats])
    yv = sy.transform(val[targets])
    
    # [优化 2] Optuna 参数范围调整：限制过拟合，增强正则化
    print(">>> [Optuna] Optimizing parameters (Robust Mode)...")
    
    def objective(trial):
        params = {
            'objective': 'reg:pseudohubererror', 
            'tree_method': 'hist',
            'device': 'cuda',
            # 降低基学习器数量
            'n_estimators': 800, 
            # 限制树深度，浅树泛化更好
            'max_depth': trial.suggest_int('depth', 3, 5),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.08),
            # 降低采样比例，增加随机性
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample', 0.3, 0.7),
            # 增强正则化 (L1 & L2)
            'reg_alpha': trial.suggest_float('alpha', 1.0, 10.0),
            'reg_lambda': trial.suggest_float('lambda', 5.0, 50.0), # 强L2正则
            'gamma': trial.suggest_float('gamma', 1.0, 10.0),
            # 增加子节点最小权重，避免学习噪音
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 50),
            'n_jobs': 4
        }
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for t, v_idx in kf.split(Xt):
            model = MultiOutputRegressor(xgb.XGBRegressor(**params), n_jobs=None)
            model.fit(Xt[t], yt[t])
            
            pred_scaled = model.predict(Xt[v_idx])
            pred_real = sy.inverse_transform(pred_scaled)
            true_real = sy.inverse_transform(yt[v_idx])
            
            abs_err = np.abs(true_real - pred_real)
            accuracy_15 = np.mean(abs_err <= 15)
            scores.append(accuracy_15)
            
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) # 稍微增加 trial 次数
    
    best_p = study.best_params
    best_p.update({
        'objective': 'reg:pseudohubererror',
        'tree_method': 'hist',
        'device': 'cuda',
        'n_estimators': 800  # 固定为较小的值
    })
    print(f"Best Params: {best_p}")
    
    # 5. 最终训练
    print(">>> Final Training...")
    final_model = MultiOutputRegressor(xgb.XGBRegressor(**best_p), n_jobs=None)
    final_model.fit(Xt, yt)
    
    # 6. 内部验证
    print(">>> Internal Validation (Clinical Only)...")
    y_pred_val_scaled = final_model.predict(Xv)
    
    # 反归一化
    y_val_real = sy.inverse_transform(yv)
    y_pred_val_real = sy.inverse_transform(y_pred_val_scaled)
    
    freq_names = [t.split('_')[0].replace('AC-', '') for t in targets]
    
    save_full_analysis(y_val_real, y_pred_val_real, feats, final_model, "Internal_Validation_Results_ClinicalOnly", freq_names)
    
    # 7. 外部验证
    print(">>> External Validation (Clinical Only)...")
    B_VAL = 'implanted/val/'
    try:
        df_e = load_and_preprocess(B_VAL, is_train=False)
        
        if len(df_e) > 0:
            # 特征工程
            df_e = feature_engineering_advanced(df_e)
            
            # 计算 PTA (外部)
            sub_t_names = [t for t in targets if 'PTA' not in t]
            sub_t_exist = [t for t in sub_t_names if t in df_e.columns] 
            if sub_t_exist: df_e['PTA_post'] = df_e[sub_t_exist].mean(axis=1)
            
            # 对齐特征
            df_ready = pd.DataFrame(0, index=df_e.index, columns=feats)
            com = [c for c in df_e.columns if c in feats]
            df_ready[com] = df_e[com]
            
            # 归一化
            Xe = sx.transform(df_ready)
            ye = sy.transform(df_e[targets])
            
            # 预测
            y_pred_ext_scaled = final_model.predict(Xe)
            
            # 反归一化
            y_ext_real = sy.inverse_transform(ye)
            y_pred_ext_real = sy.inverse_transform(y_pred_ext_scaled)
            
            save_full_analysis(y_ext_real, y_pred_ext_real, feats, final_model, "External_Validation_Results_ClinicalOnly", freq_names)
            print("Done.")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Ext Val Failed: {e}")

    print("\nAll tasks completed (Clinical Only).")

if __name__ == "__main__":
    run_clinical_only()