import warnings
warnings.filterwarnings('ignore')

import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, learning_curve, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, recall_score, precision_score,
                              accuracy_score, confusion_matrix,
                              RocCurveDisplay, PrecisionRecallDisplay,
                              ConfusionMatrixDisplay, precision_recall_curve,
                              roc_curve, brier_score_loss)
from sklearn.calibration import CalibrationDisplay
import xgboost as xgb

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE = Path('/Users/angelabetalleluz/Documents/TfMlAngela')
FIG_DIR = BASE / 'figs'
OUT_DIR = BASE / 'outputs'
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# configuracion general de graficos
plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.facecolor': 'white',
})

PALETTE = {'COEF': '#2196F3', 'CNE': '#FF7043'}
PALETTE_BIN = {1: '#2196F3', 0: '#FF7043'}

print("setup listo")

# -------------------------------------------------------
# carga de datos
# -------------------------------------------------------

df = pd.read_csv(BASE / 'ContactoCobranza.csv', sep=';', decimal='.')
print(f"shape: {df.shape}")
print(f"duplicados CLIENTE: {df.duplicated(subset='CLIENTE').sum()}")
print(f"\nnulos:\n{df.isna().sum()}")
print(f"\nMES:\n{df['MES'].value_counts().sort_index()}")

# diccionario de variables
descripciones = {
    'MES': 'Mes de observacion (YYYYMM)',
    'CLIENTE': 'ID unico cliente',
    'NRO_VEC_COB': 'Veces en cobranzas (binned)',
    'PDPs_ROTAS': 'Promesas de pago incumplidas',
    'ESTADO_PDP': 'Estado promesa de pago (0=sin promesa, 1=temprana)',
    'NRO_CUOTAS': 'Cuotas adeudadas (binned)',
    'MES_0': 'Deuda vencida mes actual (S/.)',
    'MES_1': 'Deuda vencida mes anterior',
    'MES_2': 'Deuda vencida hace 2 meses',
    'FECHALLAMADA': 'Fecha gestion (DD/MM/YYYY)',
    'HORA': 'Hora gestion',
    'DEUDA_TOTAL': 'Deuda total al cierre (S/.)',
    'ESTATUS': 'Estado titular -- siempre BT, no sirve',
    'ACTIVACION': 'Año activacion cliente',
    'MORA': 'Estado mora',
    'TIPOCONTACTO': 'Target: COEF=efectivo CNE=no efectivo',
}

dd_rows = []
for col in df.columns:
    dd_rows.append({
        'Columna': col,
        'Tipo': str(df[col].dtype),
        'Descripcion': descripciones.get(col, ''),
        '% Nulos': f"{df[col].isna().mean()*100:.1f}%",
        'Cardinalidad': df[col].nunique(),
        'Ejemplo': str(df[col].dropna().iloc[0]) if df[col].notna().any() else 'N/A',
    })

dd = pd.DataFrame(dd_rows)
dd.to_csv(OUT_DIR / 'diccionario_datos.csv', index=False)
print("diccionario guardado")


# -------------------------------------------------------
# EDA univariado
# -------------------------------------------------------

# fig 1 -- distribucion del target
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
counts = df['TIPOCONTACTO'].value_counts()
pcts   = df['TIPOCONTACTO'].value_counts(normalize=True) * 100
colors = [PALETTE['COEF'], PALETTE['CNE']]

axes[0].bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.8)
axes[0].set_title('Distribucion del Target (TIPOCONTACTO)', pad=10)
axes[0].set_ylabel('Frecuencia')
axes[0].set_ylim(0, max(counts.values) * 1.25)
for i, (v, p) in enumerate(zip(counts.values, pcts.values)):
    axes[0].text(i, v + 60, f'{v:,}\n({p:.1f}%)', ha='center', fontsize=9, fontweight='bold')

axes[1].pie(counts.values, labels=counts.index, colors=colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
axes[1].set_title('Proporcion COEF vs CNE', pad=10)

plt.suptitle('Figura 1 -- Distribucion de la variable objetivo', fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(FIG_DIR / '01_target_dist.png', bbox_inches='tight', dpi=120)
plt.close()
print("fig 1 ok")

# fig 2 -- histogramas numericas
num_cols_raw = ['MES_0', 'MES_1', 'MES_2', 'DEUDA_TOTAL', 'HORA', 'ACTIVACION']

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()
for i, col in enumerate(num_cols_raw):
    data = df[col].dropna()
    axes[i].hist(data, bins=40, color='#5C6BC0', edgecolor='white', linewidth=0.3, alpha=0.85)
    axes[i].set_title(col)
    axes[i].set_xlabel('Valor')
    axes[i].set_ylabel('Frecuencia')
    med = data.median()
    axes[i].axvline(med, color='red', linestyle='--', linewidth=1.2, label=f'Mediana: {med:.0f}')
    axes[i].legend(fontsize=8)

plt.suptitle('Figura 2 -- Histogramas variables numericas', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '02_hist_numericas.png', bbox_inches='tight')
plt.close()
print("fig 2 ok")

# fig 3 -- boxplots vs target
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()
mw_results = []
for i, col in enumerate(num_cols_raw):
    sns.boxplot(data=df, x='TIPOCONTACTO', y=col, ax=axes[i],
                palette=PALETTE, order=['COEF', 'CNE'])
    coef_vals = df.loc[df['TIPOCONTACTO']=='COEF', col].dropna()
    cne_vals  = df.loc[df['TIPOCONTACTO']=='CNE',  col].dropna()
    stat, p = stats.mannwhitneyu(coef_vals, cne_vals, alternative='two-sided')
    if p < 0.001:   sig = '***'
    elif p < 0.01:  sig = '**'
    elif p < 0.05:  sig = '*'
    else:           sig = 'ns'
    axes[i].set_title(f'{col} ({sig})', fontsize=10)
    mw_results.append({'Feature': col, 'U-stat': round(stat, 1), 'p-valor': round(p, 4), 'Sig': sig})

plt.suptitle('Figura 3 -- Boxplots por clase (Mann-Whitney)', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '03_box_target.png', bbox_inches='tight')
plt.close()
pd.DataFrame(mw_results).to_csv(OUT_DIR / 'test_mannwhitney.csv', index=False)
print("fig 3 ok")

# tasas COEF por categoricas -- necesario para el word
cat_cols_raw = ['NRO_VEC_COB', 'PDPs_ROTAS', 'ESTADO_PDP', 'NRO_CUOTAS', 'MORA']
tasa_rows = []
for col in cat_cols_raw:
    tmp = df.groupby(col)['TIPOCONTACTO'].apply(lambda x: (x=='COEF').mean()*100).reset_index()
    tmp.columns = [col, 'tasa_COEF_pct']
    tmp['feature'] = col
    tasa_rows.append(tmp.rename(columns={col: 'valor'}))
tasa_df = pd.concat(tasa_rows, ignore_index=True)
tasa_df.to_csv(OUT_DIR / 'tasa_coef_por_categoria.csv', index=False)

# fig 4 -- categoricas vs target
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(cat_cols_raw):
    tmp = df.copy()
    tmp[col] = tmp[col].fillna('sin_dato').astype(str)
    ct = tmp.groupby([col, 'TIPOCONTACTO']).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind='bar', ax=axes[i],
                color=[PALETTE['COEF'], PALETTE['CNE']],
                edgecolor='white', linewidth=0.5)
    axes[i].set_title(col)
    axes[i].set_ylabel('Porcentaje (%)')
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].legend(title='Tipo', fontsize=8)
axes[5].set_visible(False)
plt.suptitle('Figura 4 -- Distribucion por variable categorica y clase', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '04_cat_target.png', bbox_inches='tight')
plt.close()
print("fig 4 ok")

# fig 5 -- eda temporal
df_tmp = df.copy()
df_tmp['FECHALLAMADA_dt'] = pd.to_datetime(df_tmp['FECHALLAMADA'], format='%d/%m/%Y', errors='coerce')
df_tmp['dia_semana_str'] = df_tmp['FECHALLAMADA_dt'].dt.day_name()
df_tmp['coef_bin'] = (df_tmp['TIPOCONTACTO'] == 'COEF').astype(int)

dias_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

def franja(h):
    if pd.isna(h): return 'desconocida'
    if h < 12: return 'manana'
    if h < 18: return 'tarde'
    return 'noche'

df_tmp['franja_str'] = df_tmp['HORA'].apply(franja)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

tasa_dia = (df_tmp.groupby('dia_semana_str')['coef_bin'].mean() * 100).reindex(
    [d for d in dias_order if d in df_tmp['dia_semana_str'].unique()])
axes[0].bar(range(len(tasa_dia)), tasa_dia.values, color='#42A5F5', edgecolor='white')
axes[0].set_xticks(range(len(tasa_dia)))
axes[0].set_xticklabels([d[:3] for d in tasa_dia.index], rotation=30)
axes[0].set_title('Tasa COEF (%) por dia de semana')
axes[0].set_ylabel('% COEF')
axes[0].axhline(29.4, color='red', linestyle='--', label='Promedio 29.4%')
axes[0].legend(fontsize=8)

hora_tasa = df_tmp.groupby('HORA')['coef_bin'].mean() * 100
axes[1].plot(hora_tasa.index, hora_tasa.values, marker='o', color='#7E57C2', linewidth=2)
axes[1].set_title('Tasa COEF (%) por hora')
axes[1].set_xlabel('Hora')
axes[1].set_ylabel('% COEF')
axes[1].axhline(29.4, color='red', linestyle='--', label='Promedio')
axes[1].legend(fontsize=8)

franja_order = ['manana', 'tarde', 'noche']
tasa_franja = (df_tmp.groupby('franja_str')['coef_bin'].mean() * 100).reindex(
    [f for f in franja_order if f in df_tmp['franja_str'].unique()])
axes[2].bar(range(len(tasa_franja)), tasa_franja.values,
            color=['#FFA726','#EF5350','#8D6E63'], edgecolor='white')
axes[2].set_xticks(range(len(tasa_franja)))
axes[2].set_xticklabels(tasa_franja.index)
axes[2].set_title('Tasa COEF (%) por franja horaria')
axes[2].set_ylabel('% COEF')
axes[2].axhline(29.4, color='red', linestyle='--', label='Promedio')
axes[2].legend(fontsize=8)

plt.suptitle('Figura 5 -- Contactabilidad por variables temporales', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '05_temporal.png', bbox_inches='tight')
plt.close()
print("fig 5 ok")


# -------------------------------------------------------
# EDA multivariado
# -------------------------------------------------------

df_corr = df[num_cols_raw + ['MORA']].copy()
df_corr['TIPOCONTACTO_bin'] = (df['TIPOCONTACTO'] == 'COEF').astype(int)
corr = df_corr.dropna().corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, linewidths=0.5, annot_kws={'size': 9})
ax.set_title('Figura 6 -- Matriz de correlacion (Pearson)', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '06_corr.png', bbox_inches='tight')
plt.close()
print("fig 6 ok")


# -------------------------------------------------------
# limpieza
# -------------------------------------------------------
print("\nlimpieza...")

df_clean = df.copy()

# ESTATUS es constante = BT, no aporta nada
df_clean.drop(columns=['ESTATUS'], inplace=True)

# NRO_VEC_COB -- nulos significan que nunca cayeron en cobranzas, pongo 'sin_dato'
df_clean['NRO_VEC_COB'] = df_clean['NRO_VEC_COB'].fillna('sin_dato')
print(f"  NRO_VEC_COB: {(df_clean['NRO_VEC_COB']=='sin_dato').sum()} imputados como sin_dato")

# MES_2 -- puede ser nulos si recien entraron, creo flag para no perder esa info
df_clean['MES_2_missing'] = df_clean['MES_2'].isna().astype(int)
df_clean['MES_2'] = df_clean['MES_2'].fillna(0.0)

# NRO_CUOTAS -- imputo con moda
moda_cuotas = df_clean['NRO_CUOTAS'].mode()[0]
df_clean['NRO_CUOTAS'] = df_clean['NRO_CUOTAS'].fillna(moda_cuotas)
print(f"  NRO_CUOTAS: moda = '{moda_cuotas}'")

assert df_clean.isna().sum().sum() == 0, "todavia hay nulos!"
print(f"  limpieza ok. shape: {df_clean.shape}")
df_clean.to_parquet(OUT_DIR / 'df_clean.parquet')


# -------------------------------------------------------
# feature engineering
# -------------------------------------------------------
print("\nfeature engineering...")

df_fe = df_clean.copy()

# variables temporales
df_fe['FECHALLAMADA_dt'] = pd.to_datetime(df_fe['FECHALLAMADA'], format='%d/%m/%Y', errors='coerce')
df_fe['dia_semana']  = df_fe['FECHALLAMADA_dt'].dt.dayofweek   # 0=lunes
df_fe['dia_mes']     = df_fe['FECHALLAMADA_dt'].dt.day
df_fe['mes_num']     = df_fe['FECHALLAMADA_dt'].dt.month
df_fe['es_inicio_mes'] = (df_fe['dia_mes'] <= 10).astype(int)

def franja_cat(h):
    if h < 12: return 'manana'
    if h < 18: return 'tarde'
    return 'noche'

df_fe['franja_horaria'] = df_fe['HORA'].apply(franja_cat)
df_fe['hora_pico'] = df_fe['HORA'].apply(lambda h: 1 if (9 <= h <= 11 or 17 <= h <= 19) else 0)

# antiguedad del cliente
df_fe['anio_mes'] = df_fe['MES'] // 100
df_fe['antiguedad_anios'] = df_fe['anio_mes'] - df_fe['ACTIVACION']
df_fe['cliente_nuevo'] = (df_fe['antiguedad_anios'] <= 1).astype(int)

# features de deuda
df_fe['delta_mes_0_1'] = df_fe['MES_0'] - df_fe['MES_1']
df_fe['delta_mes_1_2'] = df_fe['MES_1'] - df_fe['MES_2']
df_fe['ratio_vencido_total'] = df_fe['MES_0'] / df_fe['DEUDA_TOTAL'].replace(0, np.nan)
df_fe['ratio_vencido_total'] = df_fe['ratio_vencido_total'].fillna(0)

def tendencia(d):
    if d > 100:  return 'creciente'
    if d < -100: return 'decreciente'
    return 'estable'

df_fe['tendencia_deuda'] = df_fe['delta_mes_0_1'].apply(tendencia)

# transformaciones log para reducir sesgo en deudas
df_fe['log_deuda_total'] = np.log1p(df_fe['DEUDA_TOTAL'])
df_fe['log_mes_0'] = np.log1p(df_fe['MES_0'])
df_fe['log_mes_1'] = np.log1p(df_fe['MES_1'])

# target
df_fe['target'] = (df_fe['TIPOCONTACTO'] == 'COEF').astype(int)

# saco columnas que no son features
drop_cols = ['CLIENTE', 'FECHALLAMADA', 'FECHALLAMADA_dt', 'HORA',
             'ACTIVACION', 'anio_mes', 'TIPOCONTACTO']
df_fe.drop(columns=drop_cols, inplace=True)

print(f"  shape final: {df_fe.shape}")
print(f"  columnas: {list(df_fe.columns)}")
df_fe.to_parquet(OUT_DIR / 'df_fe.parquet')


# -------------------------------------------------------
# particion temporal
# -------------------------------------------------------

train_mask = df_fe['MES'].isin([201402, 201403])
test_mask  = df_fe['MES'] == 201404

feature_cols = [c for c in df_fe.columns if c not in ['MES', 'target']]

X_train = df_fe.loc[train_mask, feature_cols].reset_index(drop=True)
y_train = df_fe.loc[train_mask, 'target'].reset_index(drop=True)
X_test  = df_fe.loc[test_mask,  feature_cols].reset_index(drop=True)
y_test  = df_fe.loc[test_mask,  'target'].reset_index(drop=True)

print(f"train: {X_train.shape}  COEF={y_train.mean()*100:.1f}%")
print(f"test:  {X_test.shape}   COEF={y_test.mean()*100:.1f}%")


# -------------------------------------------------------
# pipelines de preprocesamiento
# -------------------------------------------------------

num_cols = [
    'MES_0', 'MES_1', 'MES_2', 'DEUDA_TOTAL',
    'delta_mes_0_1', 'delta_mes_1_2', 'ratio_vencido_total',
    'log_deuda_total', 'log_mes_0', 'log_mes_1',
    'antiguedad_anios', 'dia_semana', 'dia_mes', 'mes_num',
    'MES_2_missing', 'hora_pico', 'es_inicio_mes', 'cliente_nuevo'
]

ord_cols_info = [
    ('NRO_VEC_COB', ['sin_dato', '<=10', '>10']),
    ('NRO_CUOTAS',  ['<=24', '<24, 48]', '>48']),
]
ord_col_names  = [c for c, _ in ord_cols_info]
ord_categories = [cats for _, cats in ord_cols_info]

cat_cols = ['PDPs_ROTAS', 'ESTADO_PDP', 'MORA', 'franja_horaria', 'tendencia_deuda']

# verifico que no falte nada
all_expected = set(num_cols + ord_col_names + cat_cols)
all_actual   = set(feature_cols)
missing = all_expected - all_actual
extra   = all_actual - all_expected
if missing: print(f"ADVERTENCIA -- columnas faltantes: {missing}")
if extra:   print(f"columnas extra (no asignadas): {extra}")

# para modelos lineales necesito escalar
pre_lineal = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('ord', OrdinalEncoder(categories=ord_categories,
                           handle_unknown='use_encoded_value',
                           unknown_value=-1), ord_col_names),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore',
                          sparse_output=False), cat_cols),
], remainder='drop')

# para arboles no hace falta escalar
pre_arbol = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('ord', OrdinalEncoder(categories=ord_categories,
                           handle_unknown='use_encoded_value',
                           unknown_value=-1), ord_col_names),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore',
                          sparse_output=False), cat_cols),
], remainder='drop')


# -------------------------------------------------------
# seleccion de caracteristicas
# -------------------------------------------------------
print("\nseleccion de features...")

X_train_arbol = pre_arbol.fit_transform(X_train)
mi_scores = mutual_info_classif(X_train_arbol, y_train, random_state=RANDOM_STATE)

num_names = num_cols
ord_names = ord_col_names
cat_names = list(pre_arbol.named_transformers_['cat'].get_feature_names_out(cat_cols))
all_feature_names = num_names + ord_names + cat_names

mi_df = pd.DataFrame({'Feature': all_feature_names, 'MI_Score': mi_scores})
mi_df = mi_df.sort_values('MI_Score', ascending=False).reset_index(drop=True)
mi_df.to_csv(OUT_DIR / 'mutual_info.csv', index=False)

# fig 7 -- top 15 mutual information
fig, ax = plt.subplots(figsize=(9, 6))
top15 = mi_df.head(15)
bars = ax.barh(range(15), top15['MI_Score'].values, color='#5C6BC0', edgecolor='white')
ax.set_yticks(range(15))
ax.set_yticklabels(top15['Feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Mutual Information Score')
ax.set_title('Figura 7 -- Top 15 variables por Mutual Information', fontweight='bold')
for bar, val in zip(bars, top15['MI_Score'].values):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / '08_mutual_info.png', bbox_inches='tight')
plt.close()
print("fig 7 ok")


# -------------------------------------------------------
# modelado
# -------------------------------------------------------
print("\nmodelado y tuning...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# regresion logistica -- baseline
print("  logistic regression...")
lr_pipe = Pipeline([
    ('pre', pre_lineal),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=2000,
                                solver='liblinear', random_state=RANDOM_STATE))
])
lr_grid = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l1', 'l2']
}
lr_gs = GridSearchCV(lr_pipe, lr_grid, cv=cv, scoring='roc_auc',
                     n_jobs=-1, refit=True, return_train_score=True)
lr_gs.fit(X_train, y_train)
best_lr = lr_gs.best_estimator_
pd.DataFrame(lr_gs.cv_results_).to_csv(OUT_DIR / 'cv_results_lr.csv', index=False)
print(f"    LR -- params: {lr_gs.best_params_}  auc: {lr_gs.best_score_:.4f}")

# random forest
print("  random forest...")
rf_pipe = Pipeline([
    ('pre', pre_arbol),
    ('clf', RandomForestClassifier(class_weight='balanced',
                                   random_state=RANDOM_STATE, n_jobs=-1))
])
rf_param_dist = {
    'clf__n_estimators':     [200, 400, 600],
    'clf__max_depth':        [None, 6, 10, 16],
    'clf__min_samples_split':[2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 5],
    'clf__max_features':     ['sqrt', 'log2'],
}
rf_rs = RandomizedSearchCV(rf_pipe, rf_param_dist, n_iter=25, cv=cv,
                            scoring='roc_auc', n_jobs=-1, refit=True,
                            random_state=RANDOM_STATE, return_train_score=True)
rf_rs.fit(X_train, y_train)
best_rf = rf_rs.best_estimator_
pd.DataFrame(rf_rs.cv_results_).to_csv(OUT_DIR / 'cv_results_rf.csv', index=False)
print(f"    RF -- params: {rf_rs.best_params_}  auc: {rf_rs.best_score_:.4f}")

# xgboost
print("  xgboost...")
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb_pipe = Pipeline([
    ('pre', pre_arbol),
    ('clf', xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc',
                               tree_method='hist', random_state=RANDOM_STATE,
                               n_jobs=-1, scale_pos_weight=scale_pos,
                               verbosity=0))
])
xgb_param_dist = {
    'clf__n_estimators':    [200, 400, 600],
    'clf__max_depth':       [3, 5, 7],
    'clf__learning_rate':   [0.01, 0.05, 0.1],
    'clf__subsample':       [0.7, 0.85, 1.0],
    'clf__colsample_bytree':[0.7, 0.85, 1.0],
    'clf__min_child_weight':[1, 3, 5],
    'clf__gamma':           [0, 0.1, 0.3],
    'clf__reg_alpha':       [0, 0.1],
    'clf__reg_lambda':      [1, 5],
}
xgb_rs = RandomizedSearchCV(xgb_pipe, xgb_param_dist, n_iter=35, cv=cv,
                              scoring='roc_auc', n_jobs=-1, refit=True,
                              random_state=RANDOM_STATE, return_train_score=True)
xgb_rs.fit(X_train, y_train)
best_xgb = xgb_rs.best_estimator_
pd.DataFrame(xgb_rs.cv_results_).to_csv(OUT_DIR / 'cv_results_xgb.csv', index=False)
print(f"    XGB -- params: {xgb_rs.best_params_}  auc: {xgb_rs.best_score_:.4f}")

# tabla resumen cv
cv_comparison = pd.DataFrame([
    {'Modelo': 'Logistic Regression', 'CV ROC-AUC': round(lr_gs.best_score_, 4),
     'Mejores Hiperparametros': str(lr_gs.best_params_)},
    {'Modelo': 'Random Forest',       'CV ROC-AUC': round(rf_rs.best_score_, 4),
     'Mejores Hiperparametros': str(rf_rs.best_params_)},
    {'Modelo': 'XGBoost',             'CV ROC-AUC': round(xgb_rs.best_score_, 4),
     'Mejores Hiperparametros': str(xgb_rs.best_params_)},
])
cv_comparison.to_csv(OUT_DIR / 'cv_comparison.csv', index=False)
print(cv_comparison.to_string(index=False))

# fig 8 -- comparacion cv
fig, ax = plt.subplots(figsize=(8, 4))
scores  = [lr_gs.best_score_, rf_rs.best_score_, xgb_rs.best_score_]
modelos = ['Logistic\nRegression', 'Random\nForest', 'XGBoost']
colors  = ['#42A5F5', '#66BB6A', '#FFA726']
bars = ax.bar(modelos, scores, color=colors, edgecolor='white', linewidth=0.8)
ax.set_ylim(min(scores) - 0.05, min(max(scores) + 0.05, 1.0))
ax.set_ylabel('CV ROC-AUC (5-fold)')
ax.set_title('Figura 8 -- Comparacion CV ROC-AUC por modelo', fontweight='bold')
for bar, val in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', fontweight='bold')
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Baseline (0.5)')
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / '09_cv_compare.png', bbox_inches='tight')
plt.close()
print("fig 8 ok")


# -------------------------------------------------------
# evaluacion en hold-out (abril)
# -------------------------------------------------------
print("\nevaluacion en test (abril)...")

models_dict = {
    'Logistic Regression': best_lr,
    'Random Forest':       best_rf,
    'XGBoost':             best_xgb,
}

def optimal_threshold_pr(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    precision_arr, recall_arr, thresholds = precision_recall_curve(y, proba)
    f1_arr = np.where(
        (precision_arr + recall_arr) > 0,
        2 * precision_arr * recall_arr / (precision_arr + recall_arr),
        0
    )
    best_idx = np.argmax(f1_arr[:-1])
    return thresholds[best_idx], f1_arr[best_idx]

umbrales = {}
for name, model in models_dict.items():
    thr, f1 = optimal_threshold_pr(model, X_train, y_train)
    umbrales[name] = thr
    print(f"  {name}: umbral={thr:.3f}  f1_val={f1:.4f}")

# metricas en test con umbral 0.5 y umbral optimo
metrics_rows = []
for name, model in models_dict.items():
    proba_test = model.predict_proba(X_test)[:, 1]
    for thr_label, threshold in [('0.50', 0.5), ('Optimo', umbrales[name])]:
        thr_val = 0.5 if thr_label == '0.50' else umbrales[name]
        pred = (proba_test >= thr_val).astype(int)
        metrics_rows.append({
            'Modelo':    name,
            'Umbral':    thr_label,
            'Accuracy':  round(accuracy_score(y_test, pred), 4),
            'Precision': round(precision_score(y_test, pred, zero_division=0), 4),
            'Recall':    round(recall_score(y_test, pred, zero_division=0), 4),
            'F1':        round(f1_score(y_test, pred, zero_division=0), 4),
            'ROC-AUC':   round(roc_auc_score(y_test, proba_test), 4),
            'PR-AUC':    round(average_precision_score(y_test, proba_test), 4),
        })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUT_DIR / 'metricas_test.csv', index=False)
print(metrics_df.to_string(index=False))

# fig 9 -- matrices de confusion
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, models_dict.items()):
    proba_test = model.predict_proba(X_test)[:, 1]
    pred = (proba_test >= umbrales[name]).astype(int)
    cm   = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['CNE (0)', 'COEF (1)'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    roc = roc_auc_score(y_test, proba_test)
    ax.set_title(f'{name}\nROC-AUC={roc:.4f}', fontsize=10)
plt.suptitle('Figura 9 -- Matrices de confusion (umbral optimo, test=abril)', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '11_confusion.png', bbox_inches='tight')
plt.close()
print("fig 9 ok")

# fig 10-11 -- curvas ROC y PR
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors_m = {
    'Logistic Regression': '#42A5F5',
    'Random Forest':       '#66BB6A',
    'XGBoost':             '#FFA726'
}
for name, model in models_dict.items():
    proba_test = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    auc = roc_auc_score(y_test, proba_test)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', color=colors_m[name], linewidth=2)
    prec, rec, _ = precision_recall_curve(y_test, proba_test)
    ap = average_precision_score(y_test, proba_test)
    axes[1].plot(rec, prec, label=f'{name} (AP={ap:.4f})', color=colors_m[name], linewidth=2)

axes[0].plot([0,1],[0,1], 'k--', linewidth=0.8, label='Aleatorio')
axes[0].set_xlabel('Tasa de Falsos Positivos')
axes[0].set_ylabel('Tasa de Verdaderos Positivos')
axes[0].set_title('Curva ROC -- Test (abril)')
axes[0].legend(fontsize=9)

baseline_pr = y_test.mean()
axes[1].axhline(baseline_pr, color='k', linestyle='--', linewidth=0.8,
                label=f'Baseline ({baseline_pr:.2f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Curva Precision-Recall -- Test (abril)')
axes[1].legend(fontsize=9)

plt.suptitle('Figuras 10-11 -- Curvas ROC y Precision-Recall (hold-out abril)', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '13_roc_pr_test.png', bbox_inches='tight')
plt.close()
print("fig 10-11 ok")

# modelo ganador por ROC-AUC
best_model_name = metrics_df.groupby('Modelo')['ROC-AUC'].max().idxmax()
best_model = models_dict[best_model_name]
print(f"\nmodelo ganador: {best_model_name}")

# fig 12 -- importancia de variables
if best_model_name in ('XGBoost', 'Random Forest'):
    clf_step   = best_model.named_steps['clf']
    importances = clf_step.feature_importances_
    pre_step   = best_model.named_steps['pre']
    cat_names_fe = list(pre_step.named_transformers_['cat'].get_feature_names_out(cat_cols))
    fn = num_cols + ord_col_names + cat_names_fe
else:
    clf_step   = best_model.named_steps['clf']
    importances = np.abs(clf_step.coef_[0])
    fn = all_feature_names

imp_df = pd.DataFrame({'Feature': fn, 'Importance': importances})
imp_df = imp_df.sort_values('Importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(range(len(imp_df)), imp_df['Importance'].values,
               color='#FFA726', edgecolor='white')
ax.set_yticks(range(len(imp_df)))
ax.set_yticklabels(imp_df['Feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Importancia')
ax.set_title(f'Figura 12 -- Importancia de variables ({best_model_name})', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '14_feature_importance.png', bbox_inches='tight')
plt.close()
print("fig 12 ok")

# fig 13 -- lift y ganancia acumulada
proba_best = best_model.predict_proba(X_test)[:, 1]
order_idx  = np.argsort(proba_best)[::-1]
y_sorted   = y_test.values[order_idx]
cum_gain   = np.cumsum(y_sorted) / y_sorted.sum()
cum_pct    = np.arange(1, len(y_sorted)+1) / len(y_sorted)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(cum_pct*100, cum_gain*100, color='#FFA726', linewidth=2, label=best_model_name)
axes[0].plot([0,100],[0,100], 'k--', linewidth=0.8, label='Aleatorio')
axes[0].set_xlabel('% Leads contactados (ordenados por probabilidad)')
axes[0].set_ylabel('% COEFs capturados')
axes[0].set_title('Curva de Ganancia Acumulada')
axes[0].legend()
axes[0].fill_between(cum_pct*100, cum_gain*100, cum_pct*100, alpha=0.15, color='#FFA726')

lift = cum_gain / cum_pct
axes[1].plot(cum_pct*100, lift, color='#7E57C2', linewidth=2)
axes[1].axhline(1, color='k', linestyle='--', linewidth=0.8, label='Lift=1')
axes[1].set_xlabel('% Leads contactados')
axes[1].set_ylabel('Lift')
axes[1].set_title('Curva de Lift')
axes[1].legend()
axes[1].set_ylim(0, max(lift[:10]) * 1.1)

plt.suptitle(f'Figura 13 -- Curvas de Ganancia y Lift ({best_model_name})', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '15_lift.png', bbox_inches='tight')
plt.close()
print("fig 13 ok")

# fig 14 -- calibracion
fig, ax = plt.subplots(figsize=(7, 5))
for name, model in models_dict.items():
    CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=10, ax=ax, name=name)
ax.set_title('Figura 14 -- Curva de calibracion de probabilidades', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '16_calibration.png', bbox_inches='tight')
plt.close()
print("fig 14 ok")


# -------------------------------------------------------
# guardar modelo y manifest
# -------------------------------------------------------
print("\nguardando modelo final...")

joblib.dump(best_model, OUT_DIR / 'modelo_final.joblib')
print(f"  {best_model_name} guardado en outputs/modelo_final.joblib")

best_params_table = [
    {'Modelo': 'Logistic Regression', **lr_gs.best_params_,
     'CV ROC-AUC': round(lr_gs.best_score_, 4)},
    {'Modelo': 'Random Forest',
     **{k.replace('clf__',''):v for k,v in rf_rs.best_params_.items()},
     'CV ROC-AUC': round(rf_rs.best_score_, 4)},
    {'Modelo': 'XGBoost',
     **{k.replace('clf__',''):v for k,v in xgb_rs.best_params_.items()},
     'CV ROC-AUC': round(xgb_rs.best_score_, 4)},
]
pd.DataFrame(best_params_table).to_csv(OUT_DIR / 'mejores_hiperparametros.csv', index=False)

manifest = {
    'best_model_name': best_model_name,
    'cv_scores': {
        'Logistic Regression': round(lr_gs.best_score_, 4),
        'Random Forest':       round(rf_rs.best_score_, 4),
        'XGBoost':             round(xgb_rs.best_score_, 4),
    },
    'test_metrics_path': str(OUT_DIR / 'metricas_test.csv'),
    'best_params': {
        'lr':  lr_gs.best_params_,
        'rf':  {k.replace('clf__',''):v for k,v in rf_rs.best_params_.items()},
        'xgb': {k.replace('clf__',''):v for k,v in xgb_rs.best_params_.items()},
    },
    'thresholds':     {k: round(float(v), 4) for k, v in umbrales.items()},
    'train_shape':    list(X_train.shape),
    'test_shape':     list(X_test.shape),
    'scale_pos_weight': round(float(scale_pos), 2),
    'figs': {
        '01_target_dist':        str(FIG_DIR / '01_target_dist.png'),
        '02_hist_numericas':     str(FIG_DIR / '02_hist_numericas.png'),
        '03_box_target':         str(FIG_DIR / '03_box_target.png'),
        '04_cat_target':         str(FIG_DIR / '04_cat_target.png'),
        '05_temporal':           str(FIG_DIR / '05_temporal.png'),
        '06_corr':               str(FIG_DIR / '06_corr.png'),
        '08_mutual_info':        str(FIG_DIR / '08_mutual_info.png'),
        '09_cv_compare':         str(FIG_DIR / '09_cv_compare.png'),
        '11_confusion':          str(FIG_DIR / '11_confusion.png'),
        '13_roc_pr_test':        str(FIG_DIR / '13_roc_pr_test.png'),
        '14_feature_importance': str(FIG_DIR / '14_feature_importance.png'),
        '15_lift':               str(FIG_DIR / '15_lift.png'),
        '16_calibration':        str(FIG_DIR / '16_calibration.png'),
    },
    'outputs': {
        'diccionario_datos':      str(OUT_DIR / 'diccionario_datos.csv'),
        'metricas_test':          str(OUT_DIR / 'metricas_test.csv'),
        'cv_comparison':          str(OUT_DIR / 'cv_comparison.csv'),
        'mutual_info':            str(OUT_DIR / 'mutual_info.csv'),
        'mejores_hiperparametros':str(OUT_DIR / 'mejores_hiperparametros.csv'),
        'tasa_coef':              str(OUT_DIR / 'tasa_coef_por_categoria.csv'),
        'mannwhitney':            str(OUT_DIR / 'test_mannwhitney.csv'),
    }
}

with open(OUT_DIR / 'manifest.json', 'w', encoding='utf-8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("\ntodo listo!")
print(f"figuras en: {FIG_DIR}")
print(f"outputs en: {OUT_DIR}")
