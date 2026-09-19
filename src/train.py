from pathlib import Path
import json
import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

params = yaml.safe_load(Path('params.yaml').read_text(encoding='utf-8'))
model_params = params['model']

df = pd.read_csv('data/iris.csv')
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=float(params['test_size']),
    random_state=int(params['random_state']),
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=int(model_params['n_estimators']),
    max_depth=model_params['max_depth'],
    min_samples_split=int(model_params['min_samples_split']),
    min_samples_leaf=int(model_params['min_samples_leaf']),
    random_state=int(params['random_state'])
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

metrics = {
    'accuracy': round(float(accuracy_score(y_test, pred)), 5),
    'precision': round(float(precision_score(y_test, pred, average='weighted', zero_division=0)), 5),
    'recall': round(float(recall_score(y_test, pred, average='weighted', zero_division=0)), 5),
    'f1_score': round(float(f1_score(y_test, pred, average='weighted', zero_division=0)), 5)
}

Path('models').mkdir(parents=True, exist_ok=True)
joblib.dump(model, 'models/model.joblib')
Path('metrics.json').write_text(json.dumps(metrics, indent=2) + chr(10), encoding='utf-8')
print(json.dumps(metrics, indent=2))
