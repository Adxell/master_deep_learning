from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna


data = load_breast_cancer()

X, y = data.data, data.target


#Split data into traing and test
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Traning data shape before scaler", X_train.shape)
print("Test data shape before scaler", X_test.shape)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("Traing data shape: ", X_train.shape)
print("Test data shape: ", X_test.shape)


# Train a base line XGBoost model
baseline_model = XGBClassifier(eval_metrics='logloss', random_state=42)
baseline_model.fit(X_train, y_train)


baseline_preds = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_preds)
print(f"Baseline XGBoost ACcuracy: {baseline_accuracy:.4f}")

#Define the objeciutive function for optuna 

def objective(trial): 
    params = {
        'n_estimators': trial
    }