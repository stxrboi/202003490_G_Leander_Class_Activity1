import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

#read in data
df_raw = pd.read_csv("data.csv", sep=";")

#transform/tidy up data
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Creating the Random Forest regression model and fitting the data
model = RandomForestRegressor(random_state=0)  # Initialize the Random Forest model
model.fit(X_train, y_train)  # Fit the model to the training data

# Predicting the target variable for the test set
y_pred = model.predict(X_test)

r2_score = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2_score)

# Writing the result to a file
with open('results.txt', 'w') as f:
    f.write(f"R2 Score: {r2_score}\n")

# Plotting a scatterplot of actual vs predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
# Save the figure
plt.savefig('actual_vs_predicted.png')
# Display plot
plt.show()
