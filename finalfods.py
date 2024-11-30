import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

file_path = "turkiye-student-evaluation_generic.csv"  
data = pd.read_csv(file_path)
print(data.head())
missing_values = data.isnull().sum()
print("Checking missing values")
print(missing_values)
 
correlation_matrix = data.corr()

questions = [f"Q{i}" for i in range(1, 29)]
targets = ["nb.repeat", "attendance", "difficulty"]

for target in targets:
    correlations = correlation_matrix.loc[questions, target].sort_values(ascending=False)
    print(f"\nTop Correlations with {target.capitalize()}:")
    print(correlations.head(5))

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlations.to_frame(), annot=True, cmap="coolwarm", cbar=False)
    plt.title(f"Correlations of Q1-Q28 with {target.capitalize()}")
    plt.show()


X = data[questions]  
targets = ["nb.repeat", "attendance", "difficulty"]

for target in targets:
    y = data[target]  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    feature_importances = pd.Series(rf_model.feature_importances_, index=questions)
    feature_importances = feature_importances.sort_values(ascending=False)

    print(f"\nTop 10 Features for {target.capitalize()}:")
    print(feature_importances.head(10))

    plt.figure(figsize=(10, 6))
    feature_importances.head(10).plot(kind='bar', color='skyblue')
    plt.title(f"Top 10 Features for {target.capitalize()} (Random Forest)")
    plt.ylabel('Feature Importance')
    plt.show()



for target in targets:
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42)
    perm_importance_df = pd.DataFrame(perm_importance.importances_mean, index=questions, columns=['Importance'])
    perm_importance_df = perm_importance_df.sort_values(by='Importance', ascending=False)
    print(f"\nTop 10 Permutation Importance Features for {target.capitalize()}:")
    print(perm_importance_df.head(10))
    plt.figure(figsize=(10, 6))
    perm_importance_df.head(10).plot(kind='bar', color='orange')
    plt.title(f"Top 10 Features for {target.capitalize()} (Permutation Importance)")
    plt.ylabel('Importance')
    plt.show()


