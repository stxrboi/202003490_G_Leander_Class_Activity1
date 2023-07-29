import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data and perform necessary transformations
df_raw = pd.read_csv("data.csv", sep=";")

df = df_raw.melt(id_vars=["H03", "H05", "H16"],
                 var_name="Date", value_name="sales")

df[["Prefix", "Month", "Year"]] = df["Date"].astype(  #splits MO041980 into Prefix MO month 04 and year 1980
    "str").str.extract(r'(MO)(\d{2})(\d{4})')

df["date"] = pd.to_datetime(df["Year"].astype(
    str)+"-"+df["Month"].astype(str)+"-"+"01", format="%Y-%m-%d")

df.drop(columns=["Date", "Prefix", "Month", "Year"], inplace=True)
print(df.head())

# Convert categorical variable into dummy/indicator variables
df_encoded = pd.get_dummies(df, columns=['date', 'H03', 'H05', 'H16'])

# Splitting the data into features and target variable
X = df_encoded.drop('sales', axis=1)
y = df_encoded['sales']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a pipeline with StandardScaler, PolynomialFeatures, and Ridge regression
model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), Ridge(alpha=1.0))

# Fit the model to the training data
model.fit(X_train, y_train)

# Predicting the target variable for the test set
y_pred = model.predict(X_test)

# Calculate R2 score
r2_score_val = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2_score_val)

# Writing the result to a file
with open('results.txt', 'w') as f:
    f.write(f"R2 Score: {r2_score_val}\n")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.savefig('actual_vs_predicted.png')  # Save the figure
plt.show()
