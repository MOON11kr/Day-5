import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_diabetes

# Set style for better looking plots
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Load datasets
iris = load_iris()
iris_df = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['target']
)
iris_df['species'] = iris_df['target'].map(
    {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
)

wine = load_wine()
wine_df = pd.DataFrame(
    data=np.c_[wine['data'], wine['target']],
    columns=wine['feature_names'] + ['target']
)
wine_df['wine_class'] = wine_df['target'].map(
    {0: 'class_0', 1: 'class_1', 2: 'class_2'}
)

diabetes = load_diabetes()
diabetes_df = pd.DataFrame(
    data=np.c_[diabetes['data'], diabetes['target']],
    columns=[f'feature_{i}' for i in range(10)] + ['target']
)

# 1. Line Plot
plt.figure(figsize=(10, 5))
plt.plot(iris_df['sepal length (cm)'], label='Sepal Length')
plt.plot(iris_df['sepal width (cm)'], label='Sepal Width')
plt.title('Line Plot of Iris Sepal Dimensions')
plt.xlabel('Sample Index')
plt.ylabel('Measurement (cm)')
plt.legend()
plt.show()

# 2. Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='sepal length (cm)',
    y='sepal width (cm)',
    hue='species',
    data=iris_df,
    palette='viridis'
)
plt.title('Scatter Plot of Sepal Length vs Width by Species')
plt.show()

# 3. Histogram
plt.figure(figsize=(10, 6))
sns.histplot(
    iris_df['petal length (cm)'],
    bins=20,
    kde=True,
    color='skyblue'
)
plt.title('Histogram of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Bar Plot
plt.figure(figsize=(10, 6))
iris_df.groupby('species').mean()['petal width (cm)'].plot(
    kind='bar',
    color=['red', 'green', 'blue']
)
plt.title('Average Petal Width by Iris Species')
plt.ylabel('Petal Width (cm)')
plt.show()

# 5. Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='wine_class',
    y='alcohol',
    data=wine_df,
    palette='Set2'
)
plt.title('Box Plot of Alcohol Content by Wine Class')
plt.xlabel('Wine Class')
plt.ylabel('Alcohol Content')
plt.show()

# 6. Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(
    x='species',
    y='petal length (cm)',
    data=iris_df,
    palette='coolwarm'
)
plt.title('Violin Plot of Petal Length by Iris Species')
plt.show()

# 7. Heatmap
plt.figure(figsize=(10, 8))
corr = iris_df.iloc[:, :4].corr()
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    center=0
)
plt.title('Correlation Heatmap of Iris Features')
plt.show()

# 8. Pair Plot
sns.pairplot(
    iris_df.iloc[:, :5],
    hue='species',
    palette='husl'
)
plt.suptitle('Pair Plot of Iris Features by Species', y=1.02)
plt.show()

# 9. Pie Chart
plt.figure(figsize=(8, 8))
iris_df['species'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=['gold', 'lightcoral', 'lightskyblue']
)
plt.title('Distribution of Iris Species')
plt.ylabel('')
plt.show()

# 10. Regression Plot
plt.figure(figsize=(10, 6))
sns.regplot(
    x='feature_2',
    y='target',
    data=diabetes_df,
    scatter_kws={'alpha': 0.3},
    line_kws={'color': 'red'}
)
plt.title('Regression Plot of Diabetes Dataset Feature vs Target')
plt.xlabel('Feature 2')
plt.ylabel('Target')
plt.show()

# 11. Area Plot
plt.figure(figsize=(10, 6))
iris_df.iloc[:30, :4].plot.area(alpha=0.4)
plt.title('Area Plot of Iris Features (First 30 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Measurement (cm)')
plt.show()

# 12. Hexbin Plot
plt.figure(figsize=(10, 6))
plt.hexbin(
    diabetes_df['feature_2'],
    diabetes_df['target'],
    gridsize=25,
    cmap='Blues'
)
plt.colorbar(label='Count in bin')
plt.title('Hexbin Plot of Diabetes Feature vs Target')
plt.xlabel('Feature 2')
plt.ylabel('Target')
plt.show()
