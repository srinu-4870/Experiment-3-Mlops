from pathlib import Path
import pandas as pd

RAW = Path('data/raw/iris.data')
OUT = Path('data/iris.csv')

if not RAW.exists():
    raise FileNotFoundError(f'Missing dataset: {RAW}')

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(RAW, header=None, names=columns).dropna().reset_index(drop=True)

label_map = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

df['target'] = df['species'].map(label_map)

if df['target'].isna().any():
    raise ValueError('Unexpected Iris class label found')

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f'Prepared {len(df)} rows -> {OUT}')
