import pandas as pd
import warnings
from imblearn.combine import SMOTEENN

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """Loads dataset, handles missing values, and splits X and y."""
    print("Loading and preprocessing data...")
    # The UCI dataset uses '?' for missing values
    df = pd.read_csv("C:/Users/jdeep/OneDrive/Desktop/Ccp/risk_factors_cervical_cancer.csv", na_values="?")
    
    # Impute missing values with the median of each column
    df = df.fillna(df.median())
    
    # Define features (X) and target (y). Using 'Biopsy' as the target.
    # Dropping other target variables to prevent data leakage.
    X = df.drop(['Hinselmann', 'Schiller', 'Citology', 'Biopsy'], axis=1)
    y = df['Biopsy']
    
    return X, y

def apply_cpso_ga(X_train, y_train, X_test):
    """
    Placeholder for the Hybrid Chameleon Swarm and Genetic Algorithm (CPSO-GA).
    Replace the logic inside this function with your actual algorithm.
    """
    print("Applying CPSO-GA feature selection...")
    
    # --- INSERT YOUR CPSO-GA CODE HERE ---
    # For this runnable template, we will simulate selecting the top 15 features
    # using a simple correlation or random selection just so the code executes.
    # Replace 'selected_features' with the list of columns your algorithm outputs.
    
    # SIMULATED SELECTION (Grabbing the first 15 columns as a dummy test):
    selected_features = X_train.columns[:15].tolist() 
    
    print(f"Features selected by CPSO-GA: {len(selected_features)}")
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    return X_train_selected, X_test_selected

def main():
    # 1. Load Data
    filepath = 'risk_factors_cervical_cancer.csv'
    try:
        X, y = load_and_preprocess_data(filepath)
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Please ensure it is in the same directory.")
        return

    # 2. Train-Test Split (Crucial to do this before SMOTE)
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Feature Selection (CPSO-GA)
    X_train_selected, X_test_selected = apply_cpso_ga(X_train, y_train, X_test)

# 4. Revert to standard SMOTE
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

    # 5. Define Balanced Base Models
    print("Building Balanced Stacked Ensemble model...")
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=300, 
            max_depth=15, # Increased depth to allow better learning
            min_samples_split=2,
            random_state=42
        )),
        ('xgb', XGBClassifier(
            n_estimators=300, 
            max_depth=8, # Loosened the restriction
            learning_rate=0.1, # Sped up learning slightly
            use_label_encoder=False, 
            eval_metric='logloss', 
            random_state=42
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=300, 
            max_depth=15, 
            random_state=42
        ))
    ]

    # 6. Build the Stacking Classifier with standard Meta-Learner
    stacked_ensemble = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(max_iter=1000) # Removed heavy C=0.1 regularization
    )

    # 7. Train the Ensemble
    print("Training the ensemble on augmented data...")
    stacked_ensemble.fit(X_train_resampled, y_train_resampled)

    # 8. Evaluate on Untouched Test Data
    print("Evaluating model on test data...")
    y_pred = stacked_ensemble.predict(X_test_selected)

    # 9. Results
    print("\n" + "="*40)
    print("MODEL PERFORMANCE REPORT")
    print("="*40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()