import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# 0) Parâmetros / caminho do arquivo
# -------------------------------
CSV_PATH = CSV_PATH = "data_traduzido.csv"
MAX_DENDRO_SAMPLES = 300                   # reduzir para renderizar o dendrograma
MAX_BOXPLOT_FEATURES = 6                   # quantas features mostrar por boxplot

# -------------------------------
# 1) Descrição do problema
# -------------------------------
print("Problema: Agrupar músicas por similaridade sonora (ritmo, energia, valência, danceability, ...).")
print("Métodos aplicados: K-Means (agrupa por características contínuas) e Hierarchical Clustering (visualiza semelhanças entre gêneros/perfis).\n")

# -------------------------------
# 2) Carregamento & inspeção inicial
# -------------------------------
df = pd.read_csv(CSV_PATH)
print("Arquivo carregado:", CSV_PATH)
print("Dimensões iniciais:", df.shape)
print("\nColunas:")
print(df.columns.tolist())
print("\nPrimeiras linhas:")
print(df.head())

# -------------------------------
# 3) Seleção de features de áudio
# - tenta mapear nomes comuns em PT/EN (ex: 'dançabilidade' / 'danceability', 'energia'/'energy', 'valência'/'valence', 'acusticidade'/'acousticness', 'instrumentalidade'/'instrumentalness', 'vivacidade'/'liveness', 'volume'/'loudness', 'tempo'/'tempo'/'bpm')
# - se nada for encontrado, usa colunas numéricas como fallback
# -------------------------------
possible = ['tempo','bpm','energy','energia','valence','valência','danceability','dançabilidade',
            'loudness','volume','acousticness','acusticidade','instrumentalness','instrumentalidade',
            'liveness','vivacidade','speechiness','speechiness','duration_ms','duração_ms']
cols_lower = {c.lower(): c for c in df.columns}
selected = []
for p in possible:
    if p in cols_lower:
        selected.append(cols_lower[p])

if not selected:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected = numeric_cols[:8]  # pegar até 8 para evitar excesso
    print("Nenhuma feature padrão encontrada — usando colunas numéricas de fallback.")

print("Features selecionadas para clustering:", selected)
print(df[selected].describe().T)

# -------------------------------
# 4) ETL básica: duplicates + NA
# -------------------------------
before = df.shape[0]
df = df.drop_duplicates().dropna(subset=selected).reset_index(drop=True)
after = df.shape[0]
print(f"\nLinhas removidas (duplicatas/NA nas features): {before - after}")
print("Dimensões após limpeza:", df.shape)

# -------------------------------
# 5) Pré-processamento: padronização
# -------------------------------
X = df[selected].values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# -------------------------------
# 6) Redução dimensional para visualização (PCA 2 componentes)
# -------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(Xs)
df['pca1'] = X_pca[:,0]
df['pca2'] = X_pca[:,1]
print("Explained variance ratio (PCA 2 components):", pca.explained_variance_ratio_)

# -------------------------------
# 7) K-Means: escolha de k (elbow + silhouette)
# -------------------------------
K_range = range(2,9)  # 2..8
inertias = []
sils = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labs = km.fit_predict(Xs)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xs, labs))

# Plot Elbow
plt.figure(figsize=(7,4))
plt.plot(list(K_range), inertias, marker='o')
plt.title("Elbow (inertia) — KMeans")
plt.xlabel("k")
plt.ylabel("inertia")
plt.grid(True)
plt.show()

# Plot Silhouette
plt.figure(figsize=(7,4))
plt.plot(list(K_range), sils, marker='o')
plt.title("Silhouette score por k — KMeans")
plt.xlabel("k")
plt.ylabel("silhouette score")
plt.grid(True)
plt.show()

best_k = K_range[int(np.argmax(sils))]
print("Melhor k (maior silhouette):", best_k)

# Treina KMeans com best_k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit(Xs)
df['kmeans_cluster'] = kmeans.labels_

# Visualização PCA (KMeans)
plt.figure(figsize=(7,6))
plt.scatter(df['pca1'], df['pca2'], c=df['kmeans_cluster'])
plt.title(f"KMeans (k={best_k}) — projeção PCA")
plt.xlabel("PCA1"); plt.ylabel("PCA2")
plt.grid(True)
plt.show()

# Centros em escala original
centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers_orig, columns=selected)
centers_df['cluster'] = range(best_k)
print(centers_df)

print("Contagem por cluster (KMeans):")
print(df['kmeans_cluster'].value_counts().sort_index())

# -------------------------------
# 8) Boxplots por feature (até MAX_BOXPLOT_FEATURES)
# -------------------------------
for feat in selected[:MAX_BOXPLOT_FEATURES]:
    plt.figure(figsize=(7,4))
    groups = [df.loc[df['kmeans_cluster']==c, feat].values for c in sorted(df['kmeans_cluster'].unique())]
    plt.boxplot(groups, labels=[f"c{c}" for c in sorted(df['kmeans_cluster'].unique())])
    plt.title(f"Boxplot — {feat} por cluster (KMeans)")
    plt.xlabel("Cluster"); plt.ylabel(feat)
    plt.grid(True)
    plt.show()

# -------------------------------
# 9) Hierarchical Clustering: dendrograma (amostrado) + Agglomerative (Ward)
# -------------------------------
n = df.shape[0]
sample_n = min(n, MAX_DENDRO_SAMPLES)
idx = np.random.RandomState(42).choice(n, sample_n, replace=False)
Xs_sample = Xs[idx]

linked = linkage(Xs_sample, method='ward')
plt.figure(figsize=(12,5))
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Dendrograma (amostra)")
plt.xlabel("Amostras (truncado)")
plt.ylabel("Distância")
plt.show()

# Agglomerative com mesmo número de clusters para comparação
hc = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
df['hc_cluster'] = hc.fit_predict(Xs)

plt.figure(figsize=(7,6))
plt.scatter(df['pca1'], df['pca2'], c=df['hc_cluster'])
plt.title(f"Hierarchical (n_clusters={best_k}) — projeção PCA")
plt.xlabel("PCA1"); plt.ylabel("PCA2")
plt.grid(True)
plt.show()

# -------------------------------
# 10) Métricas e comparação
# -------------------------------
sil_km = silhouette_score(Xs, df['kmeans_cluster'])
sil_hc = silhouette_score(Xs, df['hc_cluster'])
print(f"Silhouette KMeans (k={best_k}): {sil_km:.4f}")
print(f"Silhouette Hierarchical (n={best_k}): {sil_hc:.4f}")

means_km = df.groupby('kmeans_cluster')[selected].mean().reset_index().rename(columns={'kmeans_cluster':'cluster'})
means_hc = df.groupby('hc_cluster')[selected].mean().reset_index().rename(columns={'hc_cluster':'cluster'})
print("\nMédias por cluster (KMeans):")
print(means_km)
print("\nMédias por cluster (Hierarchical):")
print(means_hc)

# -------------------------------
# 11) Interpretação (guia para análise)
# -------------------------------
print("\nInterpretação / como usar os resultados:")
print("- Compare as médias/centros por cluster para entender os 'perfis' (ex.: altos valores de 'dançabilidade' e 'energia' -> cluster dance/party).")
print("- Clusters com alta 'acusticidade' + baixa 'energia' podem representar músicas acústicas/mais calmas.")
print("- Silhouette mais próximo de 1: clusters bem separados; próximo de 0: sobreposição; negativo: possível problema.")
print("- Para recomendação: mapear músicas curtidas de um usuário aos clusters e recomendar músicas dentro do mesmo cluster (ou clusters próximos pelo dendrograma).")
