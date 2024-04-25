import pandas as pd
import numpy as np
import graphviz
from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
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
    
    def grid_search(self, param_grid, cv=5):
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_

def plot_tree(model_instance, output_prefix="tree"):
    for i in range(3):
        tree = model_instance.model.estimators_[i]
        dot_data = export_graphviz(tree,
                                feature_names=model_instance.data.columns,  
                                filled=True,  
                                max_depth=2, 
                                impurity=False, 
                                proportion=True)
        graph = graphviz.Source(dot_data)
        graph.render(output_prefix + f"_{i}", format="png")
        
def get_feature_importance(model_instance):
    avg_feature_importance = np.mean([tree.feature_importances_ for tree in model_instance.model.estimators_], axis=0)
    feature_importance = list(zip(model_instance.data.columns, avg_feature_importance))
    sorted_feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    return sorted_feature_importance


if __name__ == '__main__':    
    model_instance = Model(data, target.values.ravel(), RandomForestClassifier(class_weight='balanced'))
    model_instance.split(0.2)
    param_grid = {
        'n_estimators': [5, 10, 15],
        'max_depth': [5, 8, 10],
    }
    # Perform grid search
    model_instance.grid_search(param_grid)
    
    model_instance.fit()
    
    predictions = model_instance.predict()
    # For classification tasks (adjust as necessary for regression tasks)
    accuracy = accuracy_score(model_instance.y_test, predictions)
    report = classification_report(model_instance.y_test, predictions)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    print("Confusion Matrix:")
    print(confusion_matrix(model_instance.y_test, predictions))
    
    
    y_probs = model_instance.fit_model.predict_proba(model_instance.X_test)[:, 1]
    # Calculate AUC score
    auc_score = roc_auc_score(model_instance.y_test, y_probs)
    print("AUC Score:", auc_score)
    # plot_tree(model_instance)
    feature_importance = get_feature_importance(model_instance)
    for feature, importance in feature_importance[:5]:  # Print only the top 5 features
        print(f"Feature: {feature}, Importance: {importance}")
    