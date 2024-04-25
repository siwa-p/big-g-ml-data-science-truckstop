import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

target = pd.read_csv('data/target_logreg.csv', index_col=False)
data = pd.read_csv('data/data_logreg.csv', index_col=False)
class Model:
    def __init__(self, data, target, model):
        self.data = data
        self.target = target
        self.model = model
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('imputer', SimpleImputer(strategy='median'))
        ])
        
    def split(self, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, 
                                                                                self.target, 
                                                                                test_size=test_size, 
                                                                                random_state=321,shuffle=False)
        self.X_train = self.pipeline.fit_transform(self.X_train)
        self.X_test = self.pipeline.transform(self.X_test)
    def fit(self):
        self.fit_model = self.model.fit(self.X_train, self.y_train)
        
    def predict(self):
        result = self.fit_model.predict(self.X_test)
        return result


def plot_data(model_instance, xlabel='xlabel', ylabel='ylabel', title='Plot', color='b', linestyle='-', marker=None, markersize=5):
    y_pred = model_instance.predict()
    # Calculate R^2 score
    r2_score = model_instance.fit_model.score(model_instance.X_test, model_instance.y_test)
    plt.figure(figsize=(8, 6)) 
    
    # Plot y=x line
    plt.plot(model_instance.y_test, model_instance.y_test, color='gray', linestyle='--', label='y=x')
    
    # Plot predicted vs true values
    plt.scatter(model_instance.y_test, y_pred, color=color, marker=marker, s=markersize, label='Data Points')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title + f' (R^2 Score: {r2_score:.2f})')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def model_metrics(model_instance):
    y_pred = model_instance.predict()

    accuracy = accuracy_score(model_instance.y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(model_instance.y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(model_instance.y_test, y_pred))
    
if __name__ == '__main__':
    model_instance = Model(data,target.values.ravel(), LogisticRegression(class_weight='balanced', max_iter=1000))
    model_instance.split(0.2)
    model_instance.fit()
    model_metrics(model_instance)
    