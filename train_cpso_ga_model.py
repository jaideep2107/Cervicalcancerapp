
# ============================================================================
# CERVICAL CANCER PREDICTION WITH CPSO-GA HYBRID FEATURE SELECTION
# Chicken Particle Swarm Optimization + Genetic Algorithm
# Stacked Ensemble Learning (Random Forest + XGBoost + Extra Trees)
# ============================================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

import joblib
import random
from deap import base, creator, tools, algorithms

print("\n" + "="*80)
print("CERVICAL CANCER PREDICTION - CPSO-GA HYBRID FEATURE SELECTION")
print("="*80)

# ==================== STEP 1: LOAD AND PREPROCESS DATA ====================
print("\n[1/7] Loading and preprocessing data...")

try:
    df = pd.read_csv('risk_factors_cervical_cancer.csv')
    print(f"      ✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
except FileNotFoundError:
    print("      ✗ ERROR: risk_factors_cervical_cancer.csv not found!")
    print("      Download from: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors")
    exit(1)

# Replace '?' with NaN
df = df.replace('?', np.nan)

# Convert to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Use Biopsy as target (most clinically relevant)
target_column = 'Biopsy'
X = df.drop(columns=[target_column])
y = df[target_column]

# Remove rows with missing target
mask = ~y.isna()
X = X[mask]
y = y[mask]

print(f"      ✓ Target distribution: Negative={int((y==0).sum())}, Positive={int((y==1).sum())}")
print(f"      ✓ Class imbalance ratio: {((y==0).sum() / (y==1).sum()):.2f}:1")

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.3, random_state=42, stratify=y
)

# Apply SMOTE for class imbalance
print("\n[2/7] Handling class imbalance with SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"      ✓ After SMOTE: {pd.Series(y_train_balanced).value_counts().to_dict()}")

# Normalize features
print("\n[3/7] Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)
print(f"      ✓ Features normalized: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")

feature_names = X.columns.tolist()
n_features = X_train_scaled.shape[1]

# ==================== STEP 4: PSO IMPLEMENTATION ====================
class PSO:
    """Particle Swarm Optimization for Feature Selection"""

    def __init__(self, n_particles=25, n_iterations=40, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def initialize_particles(self, n_features):
        positions = np.random.randint(0, 2, (self.n_particles, n_features))
        velocities = np.random.uniform(-1, 1, (self.n_particles, n_features))
        return positions, velocities

    def fitness_function(self, position, X_train, y_train):
        selected_indices = np.where(position == 1)[0]

        if len(selected_indices) == 0:
            return 0.0

        X_selected = X_train[:, selected_indices]

        # Use quick classifier for fitness evaluation
        clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        scores = cross_val_score(clf, X_selected, y_train, cv=3, scoring='f1')

        # Fitness: f1-score - penalty for too many features
        fitness = np.mean(scores) - 0.005 * (len(selected_indices) / len(position))
        return fitness

    def optimize(self, X_train, y_train):
        n_features = X_train.shape[1]
        positions, velocities = self.initialize_particles(n_features)

        pbest_positions = positions.copy()
        pbest_fitness = np.array([self.fitness_function(p, X_train, y_train) for p in positions])
        gbest_position = pbest_positions[np.argmax(pbest_fitness)].copy()
        gbest_fitness = np.max(pbest_fitness)

        print(f"      PSO Initial fitness: {gbest_fitness:.4f}")

        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)

                velocities[i] = (self.w * velocities[i] +
                                self.c1 * r1 * (pbest_positions[i] - positions[i]) +
                                self.c2 * r2 * (gbest_position - positions[i]))

                sigmoid = 1 / (1 + np.exp(-velocities[i]))
                positions[i] = (np.random.rand(n_features) < sigmoid).astype(int)

                fitness = self.fitness_function(positions[i], X_train, y_train)

                if fitness > pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest_positions[i] = positions[i].copy()

                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = positions[i].copy()

            if (iteration + 1) % 10 == 0:
                print(f"      PSO Iteration {iteration+1}/{self.n_iterations}: fitness={gbest_fitness:.4f}")

        selected_features = np.where(gbest_position == 1)[0]
        print(f"      ✓ PSO selected {len(selected_features)} features")
        return selected_features, gbest_fitness

# ==================== STEP 5: GA IMPLEMENTATION ====================
class GeneticAlgorithm:
    """Genetic Algorithm for Feature Selection"""

    def __init__(self, n_population=30, n_generations=40, crossover_prob=0.8, mutation_prob=0.1):
        self.n_population = n_population
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def fitness_function(self, individual, X_train, y_train):
        selected_indices = [i for i, bit in enumerate(individual) if bit == 1]

        if len(selected_indices) == 0:
            return 0.0,

        X_selected = X_train[:, selected_indices]
        clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        scores = cross_val_score(clf, X_selected, y_train, cv=3, scoring='f1')

        fitness = np.mean(scores) - 0.005 * (len(selected_indices) / len(individual))
        return fitness,

    def optimize(self, X_train, y_train):
        n_features = X_train.shape[1]

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                        toolbox.attr_bool, n=n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.fitness_function, X_train=X_train, y_train=y_train)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=self.n_population)
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        print(f"      GA Initial fitness: {max([ind.fitness.values[0] for ind in population]):.4f}")

        for generation in range(self.n_generations):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring

            if (generation + 1) % 10 == 0:
                fits = [ind.fitness.values[0] for ind in population]
                print(f"      GA Generation {generation+1}/{self.n_generations}: fitness={max(fits):.4f}")

        best_individual = tools.selBest(population, 1)[0]
        selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]

        print(f"      ✓ GA selected {len(selected_features)} features")

        del creator.FitnessMax
        del creator.Individual

        return selected_features

# ==================== STEP 6: HYBRID CPSO-GA FEATURE SELECTION ====================
print("\n[4/7] CPSO-GA Hybrid Feature Selection...")

# Run PSO
pso = PSO(n_particles=25, n_iterations=40)
pso_features, pso_fitness = pso.optimize(X_train_scaled, y_train_balanced)

# Run GA
ga = GeneticAlgorithm(n_population=30, n_generations=40)
ga_features = ga.optimize(X_train_scaled, y_train_balanced)

# Combine features (union)
combined_features = list(set(pso_features) | set(ga_features))
combined_features.sort()

print(f"\n      PSO selected: {len(pso_features)} features")
print(f"      GA selected: {len(ga_features)} features")
print(f"      Combined: {len(combined_features)} features")

selected_feature_names = [feature_names[i] for i in combined_features]
print(f"\n      Selected features: {selected_feature_names}")

# Select features
X_train_selected = X_train_scaled[:, combined_features]
X_test_selected = X_test_scaled[:, combined_features]

# ==================== STEP 7: STACKED ENSEMBLE LEARNING ====================
print("\n[5/7] Building Stacked Ensemble Model...")

# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)),
    ('xgb', XGBClassifier(n_estimators=150, max_depth=7, learning_rate=0.1, random_state=42, eval_metric='logloss')),
    ('et', ExtraTreesClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1))
]

# Meta model
meta_model = LogisticRegression(random_state=42, max_iter=1000)

# Stacking classifier
stacked_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

print("      ✓ Training base models and meta-learner...")
stacked_model.fit(X_train_selected, y_train_balanced)
print("      ✓ Model training complete!")

# ==================== STEP 8: MODEL EVALUATION ====================
print("\n[6/7] Model Evaluation...")

y_pred_train = stacked_model.predict(X_train_selected)
y_pred_test = stacked_model.predict(X_test_selected)

y_pred_proba_train = stacked_model.predict_proba(X_train_selected)[:, 1]
y_pred_proba_test = stacked_model.predict_proba(X_test_selected)[:, 1]

print("\n      TRAINING SET METRICS:")
train_acc = accuracy_score(y_train_balanced, y_pred_train)
train_prec = precision_score(y_train_balanced, y_pred_train, zero_division=0)
train_rec = recall_score(y_train_balanced, y_pred_train, zero_division=0)
train_f1 = f1_score(y_train_balanced, y_pred_train, zero_division=0)
train_auc = roc_auc_score(y_train_balanced, y_pred_proba_train)

print(f"         Accuracy:  {train_acc:.4f}")
print(f"         Precision: {train_prec:.4f}")
print(f"         Recall:    {train_rec:.4f}")
print(f"         F1-Score:  {train_f1:.4f}")
print(f"         AUC-ROC:   {train_auc:.4f}")

print("\n      TEST SET METRICS:")
test_acc = accuracy_score(y_test, y_pred_test)
test_prec = precision_score(y_test, y_pred_test, zero_division=0)
test_rec = recall_score(y_test, y_pred_test, zero_division=0)
test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
test_auc = roc_auc_score(y_test, y_pred_proba_test)

print(f"         Accuracy:  {test_acc:.4f}")
print(f"         Precision: {test_prec:.4f}")
print(f"         Recall:    {test_rec:.4f}")
print(f"         F1-Score:  {test_f1:.4f}")
print(f"         AUC-ROC:   {test_auc:.4f}")

print("\n      CONFUSION MATRIX (TEST SET):")
cm = confusion_matrix(y_test, y_pred_test)
print(f"         {cm}")

print("\n      CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_test, zero_division=0))

# ==================== STEP 9: SAVE MODEL ====================
print("\n[7/7] Saving model and preprocessing objects...")

joblib.dump(stacked_model, 'cervical_cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(np.array(combined_features), 'selected_features.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

print("      ✓ cervical_cancer_model.pkl")
print("      ✓ scaler.pkl")
print("      ✓ selected_features.pkl")
print("      ✓ feature_names.pkl")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nModel Performance Summary:")
print(f"  Test Accuracy:  {test_acc:.2%}")
print(f"  Test F1-Score:  {test_f1:.4f}")
print(f"  Test AUC-ROC:   {test_auc:.4f}")
print(f"  Features Selected: {len(combined_features)}/36")
print("="*80 + "\n")
