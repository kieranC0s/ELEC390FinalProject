import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score, roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from joblib import dump, load

# Read the datasets
for name in ['Kieran', 'Amir', 'Jack']:
    walking_features = pd.read_csv(f'walking_features_{name}.csv', header=None)
    jumping_features = pd.read_csv(f'jumping_features_{name}.csv', header=None)

    # Combine walking and jumping features
    features = pd.concat([walking_features, jumping_features], axis=0)

    # Assign labels (walking = 0, jumping = 1)
    labels = np.concatenate((np.zeros(len(walking_features)), np.ones(len(jumping_features))))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

    # Define logistic regression classifier with a pipeline that includes data normalization
    classifier = make_pipeline(StandardScaler(), LogisticRegression())

    # Train the model
    classifier.fit(X_train, y_train)

    # Save the trained model
    dump(classifier, f'classifier_{name}.joblib')

    # Calculate predictions and probabilities
    y_pred = classifier.predict(X_test)
    y_probs = classifier.predict_proba(X_test)[:, 1]

    # Calculate and print classification accuracy and recall
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f'{name} - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}')

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(2), ['Walking', 'Jumping'])
    plt.yticks(np.arange(2), ['Walking', 'Jumping'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.show()

    # Calculate and print F1 score for z acceleration
    f1 = f1_score(y_test, y_pred)
    print(f'{name} - F1 Score: {f1:.4f}')

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    # Calculate and print AUC
    auc = roc_auc_score(y_test, y_probs)
    print(f'{name} - AUC: {auc:.4f}')