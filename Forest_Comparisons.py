import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('clean_dataset.csv')

feature_cols = ['AGE', 'LOS', 'GENDER', 'Diabetes', 'CHF', 'CAD', 'Hypertension', 'Renal_Failure', 'COPD', 'Sepsis'] + \
               [col for col in df.columns if col.startswith(('max_', 'mean_', 'min_'))]

X = df[feature_cols]
y = df['READMIT_30D']

X = X.copy()
X['GENDER'] = X['GENDER'].map({'M': 1, 'F': 0})

combined = pd.concat([X, y], axis=1)
combined = combined.dropna()
X = combined[feature_cols]
y = combined['READMIT_30D']

print("=" * 80)
print(f"MULTI-MODEL PROGRESSIVE FEATURE SELECTION ANALYSIS")
print("=" * 80)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class distribution: {y.value_counts().to_dict()}")
print(f"Models: RandomForest, CatBoost, XGBoost, LightGBM")
print("=" * 80)

# ============================================================================
# STEP 1: Get baseline feature importances from ALL models
# ============================================================================
print("\nStep 1: Computing baseline feature importances from all models...")

n_splits = 5
n_epochs = 3
model_importances = {
    'RandomForest': [],
    'CatBoost': [],
    'XGBoost': [],
    'LightGBM': []
}

# Quick feature importance calculation
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"  Computing importances - Fold {fold}/{n_splits}", end='\r')
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # RandomForest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, 
                                    class_weight='balanced', n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    model_importances['RandomForest'].append(rf_clf.feature_importances_)
    
    # CatBoost
    cb_clf = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6,
                                random_state=42, verbose=0, auto_class_weights='Balanced')
    cb_clf.fit(X_train, y_train)
    model_importances['CatBoost'].append(cb_clf.feature_importances_)
    
    # XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                           random_state=42, scale_pos_weight=scale_pos_weight,
                           eval_metric='logloss', use_label_encoder=False)
    xgb_clf.fit(X_train, y_train)
    model_importances['XGBoost'].append(xgb_clf.feature_importances_)
    
    # LightGBM
    lgb_clf = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                            random_state=42, class_weight='balanced', verbose=-1)
    lgb_clf.fit(X_train, y_train)
    model_importances['LightGBM'].append(lgb_clf.feature_importances_)

print("\n")

# Compute mean importances for each model
importance_dfs = {}
for model_name in ['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']:
    mean_importances = np.mean(model_importances[model_name], axis=0)
    importance_dfs[model_name] = pd.DataFrame({
        'feature': feature_cols,
        'importance': mean_importances
    }).sort_values('importance', ascending=False)
    
    print(f"\n{model_name} - Top 10 features:")
    print(importance_dfs[model_name].head(10)[['feature', 'importance']].to_string(index=False))

# Create ensemble importance (average across all models)
ensemble_importances = np.mean([
    np.mean(model_importances['RandomForest'], axis=0),
    np.mean(model_importances['CatBoost'], axis=0),
    np.mean(model_importances['XGBoost'], axis=0),
    np.mean(model_importances['LightGBM'], axis=0)
], axis=0)

ensemble_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': ensemble_importances
}).sort_values('importance', ascending=False)

print(f"\n{'='*80}")
print("ENSEMBLE (Average) - Top 15 features:")
print(ensemble_importance_df.head(15).to_string(index=False))

# ============================================================================
# STEP 2: Define feature sets to test
# ============================================================================
feature_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, len(feature_cols)]

print(f"\n{'='*80}")
print(f"Step 2: Testing {len(feature_counts)} different feature set sizes")
print(f"Feature counts to test: {feature_counts}")
print("=" * 80)

# ============================================================================
# STEP 3: Progressive testing with ALL models
# ============================================================================
all_results = {
    'RandomForest': [],
    'CatBoost': [],
    'XGBoost': [],
    'LightGBM': []
}

for n_features in tqdm(feature_counts, desc="Testing feature sets"):
    # Select top N features based on ENSEMBLE importance
    selected_features = ensemble_importance_df.head(n_features)['feature'].tolist()
    X_subset = X[selected_features]
    
    # Test each model
    for model_name in ['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']:
        fold_aucs = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        
        for epoch in range(n_epochs):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + epoch)
            
            for train_idx, test_idx in skf.split(X_subset, y):
                X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train appropriate model
                if model_name == 'RandomForest':
                    clf = RandomForestClassifier(n_estimators=100, random_state=42 + epoch,
                                                class_weight='balanced', n_jobs=-1)
                elif model_name == 'CatBoost':
                    clf = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6,
                                           random_state=42 + epoch, verbose=0,
                                           auto_class_weights='Balanced')
                elif model_name == 'XGBoost':
                    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                    clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                                      random_state=42 + epoch, scale_pos_weight=scale_pos_weight,
                                      eval_metric='logloss', use_label_encoder=False)
                else:  # LightGBM
                    clf = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                                       random_state=42 + epoch, class_weight='balanced',
                                       verbose=-1)
                
                clf.fit(X_train, y_train)
                
                y_pred = clf.predict(X_test)
                y_pred_proba = clf.predict_proba(X_test)[:, 1]
                
                # Metrics
                auc = roc_auc_score(y_test, y_pred_proba)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='binary', zero_division=0
                )
                
                fold_aucs.append(auc)
                fold_precisions.append(precision)
                fold_recalls.append(recall)
                fold_f1s.append(f1)
        
        # Store results
        all_results[model_name].append({
            'n_features': n_features,
            'mean_auc': np.mean(fold_aucs),
            'std_auc': np.std(fold_aucs),
            'mean_precision': np.mean(fold_precisions),
            'mean_recall': np.mean(fold_recalls),
            'mean_f1': np.mean(fold_f1s),
            'features': selected_features
        })

# Convert to DataFrames
results_dfs = {model: pd.DataFrame(all_results[model]) 
               for model in ['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']}

# ============================================================================
# STEP 4: Find optimal feature set for EACH model
# ============================================================================
print("\n" + "=" * 80)
print("OPTIMAL FEATURE SETS BY MODEL")
print("=" * 80)

best_results = {}
for model_name in ['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']:
    df = results_dfs[model_name]
    best_idx = df['mean_auc'].idxmax()
    best_result = df.iloc[best_idx]
    best_results[model_name] = best_result
    
    baseline_auc = df[df['n_features'] == len(feature_cols)]['mean_auc'].values[0]
    improvement = ((best_result['mean_auc'] - baseline_auc) / baseline_auc) * 100
    
    print(f"\n{model_name}:")
    print(f"  Best # features: {best_result['n_features']}")
    print(f"  Best ROC-AUC: {best_result['mean_auc']:.4f} ± {best_result['std_auc']:.4f}")
    print(f"  Improvement: {improvement:+.2f}%")
    print(f"  Baseline (all features): {baseline_auc:.4f}")

# Find overall best
overall_best_model = max(best_results.items(), key=lambda x: x[1]['mean_auc'])
print(f"\n{'='*80}")
print(f"🏆 OVERALL WINNER: {overall_best_model[0]}")
print(f"   ROC-AUC: {overall_best_model[1]['mean_auc']:.4f}")
print(f"   Features: {overall_best_model[1]['n_features']}")
print("=" * 80)

# ============================================================================
# VISUALIZATION 1: Multi-Model Performance vs Features (2x2)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
models = ['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']
colors = ['steelblue', 'green', 'red', 'purple']

for ax, model_name, color in zip(axes.flatten(), models, colors):
    df = results_dfs[model_name]
    best = best_results[model_name]
    
    ax.plot(df['n_features'], df['mean_auc'], 
            marker='o', linewidth=2, markersize=8, color=color, alpha=0.7)
    ax.fill_between(df['n_features'], 
                     df['mean_auc'] - df['std_auc'],
                     df['mean_auc'] + df['std_auc'],
                     alpha=0.2, color=color)
    ax.axvline(x=best['n_features'], color='red', linestyle='--', 
               linewidth=2, label=f'Optimal: {best["n_features"]} features')
    ax.axhline(y=best['mean_auc'], color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
    ax.set_ylabel('ROC-AUC Score', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name}\nBest AUC: {best["mean_auc"]:.4f}', 
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 2: Comparative Analysis
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: All models on same plot
ax1 = axes[0, 0]
for model_name, color in zip(models, colors):
    df = results_dfs[model_name]
    ax1.plot(df['n_features'], df['mean_auc'], 
             marker='o', label=model_name, linewidth=2, color=color, alpha=0.7)

ax1.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax1.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
ax1.set_title('All Models: ROC-AUC vs Feature Count', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Optimal feature counts comparison
ax2 = axes[0, 1]
optimal_counts = [best_results[m]['n_features'] for m in models]
optimal_aucs = [best_results[m]['mean_auc'] for m in models]
bars = ax2.bar(models, optimal_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
for bar, auc in zip(bars, optimal_aucs):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n(AUC:{auc:.3f})', 
             ha='center', va='bottom', fontweight='bold', fontsize=9)
ax2.set_ylabel('Optimal Number of Features', fontsize=12, fontweight='bold')
ax2.set_title('Optimal Feature Count by Model', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=15)
ax2.grid(alpha=0.3, axis='y')

# Plot 3: Performance improvement heatmap
ax3 = axes[1, 0]
improvement_matrix = []
for model_name in models:
    df = results_dfs[model_name]
    baseline = df[df['n_features'] == len(feature_cols)]['mean_auc'].values[0]
    improvements = ((df['mean_auc'] - baseline) / baseline * 100).values
    improvement_matrix.append(improvements)

improvement_matrix = np.array(improvement_matrix)
im = ax3.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=2)
ax3.set_xticks(range(len(feature_counts)))
ax3.set_xticklabels(feature_counts, rotation=45)
ax3.set_yticks(range(len(models)))
ax3.set_yticklabels(models)
ax3.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax3.set_title('Performance Improvement (%) vs Baseline', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax3, label='% Improvement')

# Add text annotations
for i in range(len(models)):
    for j in range(len(feature_counts)):
        text = ax3.text(j, i, f'{improvement_matrix[i, j]:.1f}',
                       ha="center", va="center", color="black", fontsize=8)

# Plot 4: All metrics for best model
ax4 = axes[1, 1]
best_model_name = overall_best_model[0]
df = results_dfs[best_model_name]
ax4.plot(df['n_features'], df['mean_auc'], marker='o', label='ROC-AUC', linewidth=2)
ax4.plot(df['n_features'], df['mean_precision'], marker='s', label='Precision', linewidth=2)
ax4.plot(df['n_features'], df['mean_recall'], marker='^', label='Recall', linewidth=2)
ax4.plot(df['n_features'], df['mean_f1'], marker='d', label='F1-Score', linewidth=2)
ax4.axvline(x=best_results[best_model_name]['n_features'], 
            color='red', linestyle='--', linewidth=2, alpha=0.5)
ax4.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
ax4.set_title(f'{best_model_name} - All Metrics vs Feature Count', 
              fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 3: Feature Importance Comparison
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for ax, model_name, color in zip(axes, models, ['steelblue', 'darkgreen', 'darkred', 'purple']):
    imp_df = importance_dfs[model_name].head(15)
    ax.barh(range(len(imp_df)), imp_df['importance'], color=color, alpha=0.7)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df['feature'], fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name}\nTop 15 Feature Importances', 
                 fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# ============================================================================
# DETAILED RESULTS TABLE
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED RESULTS - BEST MODEL PERFORMANCE AT EACH FEATURE COUNT")
print("=" * 80)

comparison_data = []
for n_feat in feature_counts:
    row = {'n_features': n_feat}
    for model_name in models:
        df = results_dfs[model_name]
        auc = df[df['n_features'] == n_feat]['mean_auc'].values[0]
        row[model_name] = f"{auc:.4f}"
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================
# Save optimal features for each model
for model_name in models:
    best = best_results[model_name]
    optimal_features_df = pd.DataFrame({
        'feature': best['features'],
        'importance': [ensemble_importance_df[ensemble_importance_df['feature'] == f]['importance'].values[0] 
                      for f in best['features']]
    })
    filename = f'optimal_features_{model_name.lower()}.csv'
    optimal_features_df.to_csv(filename, index=False)
    print(f"✓ {model_name} optimal features saved to '{filename}'")

# Save comparison results
comparison_df.to_csv('feature_selection_comparison.csv', index=False)
print(f"✓ Comparison results saved to 'feature_selection_comparison.csv'")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

best_model = overall_best_model[0]
best_auc = overall_best_model[1]['mean_auc']
best_n_features = overall_best_model[1]['n_features']

print(f"\n✓ Use {best_model} with {best_n_features} features")
print(f"✓ Expected ROC-AUC: {best_auc:.4f}")

if best_n_features < len(feature_cols):
    removed = len(feature_cols) - best_n_features
    print(f"✓ Remove {removed} features ({removed/len(feature_cols)*100:.1f}% reduction)")
    print(f"✓ This provides optimal performance")

print(f"\nModel Rankings (by best performance):")
sorted_models = sorted(best_results.items(), key=lambda x: x[1]['mean_auc'], reverse=True)
for i, (model, result) in enumerate(sorted_models, 1):
    print(f"  {i}. {model:15s} - AUC: {result['mean_auc']:.4f} ({result['n_features']} features)")

if best_auc < 0.65:
    print("\n⚠ ROC-AUC is still relatively low (<0.65). Consider:")
    print("  - Feature engineering (interactions, polynomials)")
    print("  - Hyperparameter tuning")
    print("  - Addressing class imbalance further")
    print("  - Collecting more data or better features")
elif best_auc < 0.75:
    print("\n✓ ROC-AUC is moderate (0.65-0.75). Additional improvements possible through:")
    print("  - Hyperparameter tuning (consider Grid/Random Search)")
    print("  - Ensemble methods (stacking, voting)")
    print("  - Advanced feature engineering")
else:
    print("\n✓ ROC-AUC is good (>0.75). Model is performing well!")
    print("  Consider deploying this model for production use.")

print("\n" + "=" * 80)