import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("california_housing_train.csv")

# Check data
print(df.head())
print(df.isnull().sum())

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Separate features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Train-test split (IMPORTANT: before scaling)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
pred_gb = gb.predict(X_test)

# SVR
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train, y_train)
pred_svr = svr.predict(X_test)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

results = {}

def evaluate(actual, predicted, model):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    results[model] = r2
    
    print(model)
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2:", r2)
    print()

# Evaluate all models
evaluate(y_test, pred_lr, "Linear Regression")
evaluate(y_test, pred_dt, "Decision Tree")
evaluate(y_test, pred_rf, "Random Forest")
evaluate(y_test, pred_gb, "Gradient Boosting")
evaluate(y_test, pred_svr, "SVR")


best_model = max(results, key=results.get)
worst_model = min(results, key=results.get)

print("Best Model:", best_model)
print("Worst Model:", worst_model)