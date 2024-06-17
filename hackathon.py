import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

train_features = pd.read_csv('training_set_features.csv')
train_labels = pd.read_csv('training_set_labels.csv')
train_data = train_features.join(train_labels)
categorical_features = ['race', 'sex', 'marital_status', 'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa', 'employment_industry', 'employment_occupation']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
categorical_pipeline = ColumnTransformer([('cat', categorical_transformer, categorical_features)], remainder='passthrough')
X_train = categorical_pipeline.fit_transform(train_data.drop(['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'], axis=1))
y_train = train_data[['xyz_vaccine', 'seasonal_vaccine']].values
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model = OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=1000))
model.fit(X_train, y_train)

y_val_probs = model.predict_proba(X_val)
xyz_vaccine_auc = roc_auc_score(y_val[:, 0], y_val_probs[:, 0])
seasonal_vaccine_auc = roc_auc_score(y_val[:, 1], y_val_probs[:, 1])
mean_auc = (xyz_vaccine_auc + seasonal_vaccine_auc) / 2
print(f"ROC AUC for xyz_vaccine: {xyz_vaccine_auc:.4f}")
print(f"ROC AUC for seasonal_vaccine: {seasonal_vaccine_auc:.4f}")
print(f"Mean ROC AUC: {mean_auc:.4f}")
test_features = pd.read_csv('test_set_features.csv')
X_test = categorical_pipeline.transform(test_features.drop(['respondent_id'], axis=1))
y_test_probs = model.predict_proba(X_test)


submission = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'xyz_vaccine': y_test_probs[:, 0],
    'seasonal_vaccine': y_test_probs[:, 1]})
submission.to_csv('submission.csv', index=False)