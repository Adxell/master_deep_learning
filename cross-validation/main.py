import pandas as pd 

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# load dataset 

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


df = pd.read_csv(url)


# Display dataset info 
print("Dataset Info: \n")
print(df.info)
print("\n Class Distribution: \n")
print(df['Class'].value_counts())

# Define X and y
X = df.drop(columns=['Class'])
y = df['Class']


#  Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize K-fold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# train 
rf_model = RandomForestClassifier(random_state=42)
score_kfold = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy')

print(f'K-fold cross validation scores: {score_kfold}')
print(f'Mean Accuracy (K-Fold): {score_kfold.mean():.2f}')

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
score_stratified = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring='accuracy')

print(f'StratifiedKFold K-fold cross validation scores: {score_kfold}')
print(f'Mean Accuracy (StratifiedKFold K-fold): {score_stratified.mean():.2f}')

