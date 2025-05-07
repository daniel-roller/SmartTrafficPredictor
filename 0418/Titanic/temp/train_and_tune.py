# train_and_tune.py
# Automatically test different model and parameter combinations to evaluate prediction performance

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import preprocess_titanic

# Load dataset
print("\n📊 Loading train.csv...")
df = preprocess_titanic("../data/train.csv", is_train=True)
X = df[["Pclass", "Fare", "SibSp", "Parch", "Sex", "Embarked"]]
y = df["Survived"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
cat_features = ["Sex", "Embarked"]
num_features = ["Pclass", "Fare", "SibSp", "Parch"]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])

# Set up experiments: models & parameter combos
experiments = [
    ("DT_depth3", DecisionTreeClassifier(max_depth=3)),
    ("DT_depth5", DecisionTreeClassifier(max_depth=5)),
    ("RF_50", RandomForestClassifier(n_estimators=50)),
    ("RF_100", RandomForestClassifier(n_estimators=100)),
    ("SVM_C1_gamma1", SVC(C=1, gamma=1, kernel="rbf")),
    ("SVM_C10_gauto", SVC(C=10, gamma="auto", kernel="rbf"))
]

results = []

for name, model in experiments:
    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name}: Accuracy={acc:.3f}, F1={f1:.3f}")
    results.append((name, acc, f1))

# Plot results
labels, accs, f1s = zip(*results)
plt.figure(figsize=(10, 5))
plt.bar(labels, accs, alpha=0.6, label="Accuracy")
plt.bar(labels, f1s, alpha=0.6, label="F1-score")
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.xticks(rotation=15)
plt.legend()
plt.tight_layout()
plt.savefig("../figures/tune_result_compare.png")
print("\n✅ Saved as tune_result_compare.png")
