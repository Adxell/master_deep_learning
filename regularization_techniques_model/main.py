from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

california = fetch_california_housing()

X, y = california.data, california.target
feature_names = california.feature_names


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



print("Feature Names: \n", feature_names)
print("\n sample Data: \n", pd.DataFrame(X, columns=feature_names).head())


# Train linear regression model 
lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

# predict and evaluate 
y_pred = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred)


print(f"Linear Regression MSE (No regularization): {mse_lr:.2f}")
print(f"Coeficients\n: {lr_model.coef_}")


# Trian Rigde regression model
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)


print(f"Ridge Regrassion MSE: {mse_ridge:.2f}")
print(f"Coeficients\n: {ridge_model.coef_}")



# Trian Rigde regression model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)


# Predict and evaluate
y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)


print(f"Lasso Regrassion MSE: {mse_lasso:.2f}")
print(f"Coeficients\n: {lasso_model.coef_}")

