import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle


# Load the csv file
df = pd.read_csv("out/df_after_classification.csv")


# Select independent and dependent variable
X = df[["Возраст, лет", "Стаж вождения, лет"]]
y = df["cluster"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Instantiate the model
classifier = RandomForestClassifier(n_estimators=100)

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("out/model.pkl", "wb"))

# y_pred=classifier.predict(X_test)
#
#
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))