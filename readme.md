# XGB-ACHR: XGBoost-based Air Conduction Hearing Recovery

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Topic](https://img.shields.io/badge/Topic-Audiology-green)
![Status](https://img.shields.io/badge/Data-Private-red)

## 📖 Project Background

**XGB-ACHR** is a machine learning model built on the **XGBoost** algorithm, specifically designed for **Postoperative Air Conduction Hearing Prediction** in patients with hearing impairment.

The project aims to predict postoperative hearing recovery by analyzing preoperative clinical features. To explore the impact of different surgical strategies on prognosis, the model divides the data into two core cohorts:
1.  **Implanted**: Patients who received ossicular bone implants.
2.  **No Implanted**: Patients who did not receive ossicular implants.

Through comparative analysis and modeling, this project provides decision support for otologists to evaluate surgical prognosis.

## ⚠️ Data Privacy Notice

Due to patient privacy laws and medical ethics restrictions, **this repository does not contain raw training or validation data**.

*   The `data/` directories or specific category folders (`implanted` / `no implanted`) in the root directory are currently empty (or contain placeholders only).
*   If you wish to reproduce this project or use this code for training, please refer to the [Data Dictionary](#-data-dictionary) below to prepare your own dataset.

## 📂 Directory Structure

```text
XGB-ACHR/
├── implanted/          # [Placeholder] Stores data for the "Ossicular Implant" cohort
├── no implanted/       # [Placeholder] Stores data for the "No Implant" cohort
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

🛠️ Requirements
To run this project, the following Python libraries are required:

Python >= 3.8
XGBoost: Core modeling framework
Pandas & NumPy: Data processing
Scikit-learn: Model evaluation and data splitting
Matplotlib / Seaborn: Result visualization

Installation Command:

Bash
pip install xgboost pandas numpy scikit-learn matplotlib shap

📊 Data Dictionary
The model expects input data (Excel format) with the following clinical features.
Note: The script automatically handles mapping F/M to 0/1 and Y/N to 1/0.
    Demographic & Clinical History
    Column Name (Raw)	Description	Type
    ID	Patient Identifier	String
    Sex	Gender (F/M)	Categorical
    Year	Patient Age	Numerical
    duration(Y)	Duration of disease (Years)	Numerical
    Tinnitus	Presence of Tinnitus (Y/N)	Binary
    aural fullness	Sensation of Ear Fullness (Y/N)	Binary
    Otopyorrhea	Ear Discharge/Pus (Y/N)	Binary
    Hearing loss	Subjective Hearing Loss (Y/N)	Binary
Comorbidities
    Column Name (Raw)	Description
    Diabetes	Diabetes Mellitus status (Y/N)
    Hypertension	High Blood Pressure status (Y/N)
    coronary heart disease	CHD status (Y/N)
Surgical & Audiometric Data
    Column Name (Raw)	Description	Note
    Operation	Surgery Type	1: Tympanoplasty, 2: Modified Radical Mastoidectomy
    AC-0.25hz ... AC-8khz	Air Conduction Thresholds	Pre-op and Post-op columns required
    BC-0.25hz ... BC-8khz	Bone Conduction Thresholds	Used to calculate Air-Bone Gap
    AC-PTA / BC-PTA	Pure Tone Average	Avg of 0.5, 1, 2, 4 kHz
Please place the organized data files into the specific locations under the implanted and no implanted folders.

🚀 Usage
1.Clone the Repository

Bash
git clone https://github.com/James-wonder/XGB-ACHR.git
cd XGB-ACHR

2.Prepare Data
Format your data according to the table above and place it in the corresponding folders.

3.Run Training
(Please modify the command below according to your actual script name)

Bash
# Example: Run the main training script
python train_model.py

4.View Results
Evaluation metrics output by the model (e.g., RMSE, MAE, R2 Score) and feature importance charts will be saved in the results/ folder (if applicable).

📝 Methodology
This project employs the XGBoost (eXtreme Gradient Boosting) regression/classification model. Compared to traditional statistical methods, XGBoost demonstrates superior robustness in handling non-linear relationships, missing values, and small-sample medical data.

Key Processes:

1.Data Preprocessing: Missing value imputation and normalization.
2.Feature Engineering: Extraction of key audiological indicators (e.g., Air-Bone Gap).
3.Hyperparameter Tuning: Based on GridSearch or Bayesian Optimization.
4.Validation: 5-Fold Cross Validation to assess model stability.

🤝 Contribution
Contributions are welcome!

If you have suggestions for optimizing the code architecture, please submit an Issue or Pull Request.
For medical collaboration or data-related inquiries, please contact the author via their GitHub Profile.

📄 License
This project is licensed under the MIT License.
Please note: While this code is open source, when using it to process medical data, please ensure compliance with local data protection regulations (such as HIPAA or GDPR).