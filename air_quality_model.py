import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, classification_report, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Utils
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import optuna

# -------------------------------
# 1. LOAD DATA
# -------------------------------
print("Loading Sensor Data...")
data = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Documents\IILM doc\Abhishek.jha\abhi.minor.proj\IoT_Indoor_Air_Quality_Dataset.csv")

# -------------------------------
# 2. CLEAN DATA
# -------------------------------
data.columns = (
    data.columns.str.strip()
    .str.replace(r"\(.*?\)", "", regex=True)
    .str.replace(" ", "_")
)

def get_col(name):
    return [col for col in data.columns if name.lower() in col.lower()][0]

temp = get_col("temp")
hum = get_col("hum")
co2 = get_col("co2")
pm25 = get_col("pm2.5")
pm10 = get_col("pm10")

# -------------------------------
# 3. TARGET CREATION
# -------------------------------
data = data.dropna(subset=[pm25])

data['Air_Quality'] = pd.qcut(
    data[pm25],
    q=3,
    labels=["Good", "Moderate", "Poor"]
)

le = LabelEncoder()
data['Air_Quality'] = le.fit_transform(data['Air_Quality'])

# -------------------------------
# 4. FEATURE ENGINEERING
# -------------------------------
print("Feature Engineering...")

data['Temp_Humidity'] = data[temp] * data[hum]
data['PM_Ratio'] = data[pm25] / (data[pm10] + 1e-6)
data['PM_Total'] = data[pm25] + data[pm10]
data['Pollution_Index'] = data[pm25] + data[pm10] + data[co2]

data['Temp_sq'] = data[temp] ** 2
data['Humidity_sq'] = data[hum] ** 2
data['CO2_log'] = np.log1p(data[co2])

# -------------------------------
# 5. SPLIT DATA
# -------------------------------
X = data.select_dtypes(include=[np.number]).drop(columns=['Air_Quality'])
y = data['Air_Quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 6. HANDLE MISSING
# -------------------------------
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# -------------------------------
# 7. SCALE DATA
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 8. SMOTE
# -------------------------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# -------------------------------
# 9. TRAIN MODELS
# -------------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, eval_metric='mlogloss'),
    "SVM": SVC(C=10, probability=True),
    "Logistic Regression": LogisticRegression(max_iter=3000),
   # "MLP": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300)
}

results = {}

print("\nTraining Models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, pred)
    print(f"{name}: {results[name]:.4f}")

# -------------------------------
# 10. SELECT BEST MODEL
# -------------------------------
best_model_name = max(results, key=results.get)
print("\nBest Model:", best_model_name)

best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# -------------------------------
# 11. FINAL EVALUATION
# -------------------------------
pred = best_model.predict(X_test)

# 1. ACCURACY
acc = accuracy_score(y_test, pred)
print(f"\nAccuracy: {acc:.4f}")

# -------------------------------
# 12. CONFUSION MATRIX
# -------------------------------
labels = ["Good", "Moderate", "Poor"]
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(5,4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1,2], labels)
plt.yticks([0,1,2], labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# 13. ROC CURVE
# -------------------------------
y_score = best_model.predict_proba(X_test)
y_test_bin = label_binarize(y_test, classes=[0,1,2])

plt.figure(figsize=(6,5))
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=labels[i])

plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# -------------------------------
# 14. FEATURE IMPORTANCE
# -------------------------------
if best_model_name in ["Random Forest", "XGBoost"]:
    importances = best_model.feature_importances_
    feature_names = data.select_dtypes(include=[np.number]).drop(columns=['Air_Quality']).columns

    plt.figure(figsize=(8,5))
    plt.bar(feature_names, importances)
    plt.xticks(rotation=60)
    plt.title("Feature Importance")
    plt.ylabel("Importance Score")
    plt.show()
else:
    print("Feature importance not available")

# -------------------------------
# FINAL OUTPUT
# -------------------------------
print("\nAir Quality Prediction Complete")