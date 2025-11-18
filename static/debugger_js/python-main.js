// Python Main application logic
console.log('🟢 python-main.js loaded - Version 4 - ' + new Date().toISOString());

document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners
    setupEventListeners();
    
    console.log('✓ Python Programming Editor initialized');
    console.log('✓ loadPythonExample available:', typeof loadPythonExample);
    console.log('✓ setEditorCode available:', typeof setEditorCode);
    console.log('✓ window.setEditorCode available:', typeof window.setEditorCode);
});

function setupEventListeners() {
    // Run button
    const runBtn = document.getElementById('runBtn');
    if (runBtn) {
        runBtn.addEventListener('click', runPythonCode);
    }
    
    // Clear button
    const clearBtn = document.getElementById('clearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            clearEditor();
            clearOutput();
            clearPlots();
        });
    }
    
    // Example button
    const exampleBtn = document.getElementById('exampleBtn');
    if (exampleBtn) {
        exampleBtn.addEventListener('click', function() {
            console.log('🔘 [USER ACTION] Example button clicked');
            showPythonExampleMenu();
        });
    }
    
    // Clear output button
    const clearOutputBtn = document.getElementById('clearOutputBtn');
    if (clearOutputBtn) {
        clearOutputBtn.addEventListener('click', clearOutput);
    }
    
    // Clear plots button
    const clearPlotsBtn = document.getElementById('clearPlotsBtn');
    if (clearPlotsBtn) {
        clearPlotsBtn.addEventListener('click', clearPlots);
    }
}

function showPythonExampleMenu() {
    console.log('📋 [MENU] Opening example menu');
    const exampleCategories = [
        {
            name: 'Machine Learning Basics',
            examples: [
                { name: 'Linear Regression', key: 'ml_linear_regression' },
                { name: 'Logistic Regression', key: 'ml_logistic_regression' },
                { name: 'Decision Trees', key: 'ml_decision_trees' },
                { name: 'Random Forest', key: 'ml_random_forest' },
                { name: 'K-Means Clustering', key: 'ml_kmeans' },
                { name: 'SVM Classification', key: 'ml_svm' },
                { name: 'Naive Bayes', key: 'ml_naive_bayes' },
                { name: 'K-Nearest Neighbors', key: 'ml_knn' },
                { name: 'Gradient Boosting', key: 'ml_gradient_boosting' },
                { name: 'AdaBoost', key: 'ml_adaboost' },
                { name: 'Principal Component Analysis', key: 'ml_pca' },
                { name: 'DBSCAN Clustering', key: 'ml_dbscan' }
            ]
        },
        {
            name: 'Advanced Machine Learning',
            examples: [
                { name: 'Cross-Validation', key: 'ml_cross_validation' },
                { name: 'Grid Search Hyperparameters', key: 'ml_grid_search' },
                { name: 'Feature Selection', key: 'ml_feature_selection' },
                { name: 'Ensemble Methods', key: 'ml_ensemble' },
                { name: 'Pipeline Creation', key: 'ml_pipeline' },
                { name: 'Model Evaluation Metrics', key: 'ml_metrics' },
                { name: 'Imbalanced Data Handling', key: 'ml_imbalanced' },
                { name: 'Anomaly Detection', key: 'ml_anomaly' }
            ]
        },
        {
            name: 'Regression Models',
            examples: [
                { name: 'Ridge Regression', key: 'ml_ridge' },
                { name: 'Lasso Regression', key: 'ml_lasso' },
                { name: 'Polynomial Regression', key: 'ml_polynomial' },
                { name: 'ElasticNet Regression', key: 'ml_elasticnet' }
            ]
        },
        {
            name: 'Deep Learning',
            examples: [
                { name: 'Neural Network (Basic)', key: 'dl_neural_network' },
                { name: 'CNN - Image Classification', key: 'dl_cnn' },
                { name: 'RNN - Time Series', key: 'dl_rnn' },
                { name: 'Transfer Learning', key: 'dl_transfer_learning' }
            ]
        },
        {
            name: 'Data Science',
            examples: [
                { name: 'Pandas DataFrame Basics', key: 'ds_pandas_basics' },
                { name: 'Data Cleaning', key: 'ds_data_cleaning' },
                { name: 'Data Visualization', key: 'ds_visualization' },
                { name: 'Statistical Analysis', key: 'ds_statistics' },
                { name: 'Feature Engineering', key: 'ds_feature_engineering' }
            ]
        },
        {
            name: 'NumPy & Arrays',
            examples: [
                { name: 'Array Operations', key: 'np_array_ops' },
                { name: 'Matrix Operations', key: 'np_matrix_ops' },
                { name: 'Broadcasting', key: 'np_broadcasting' },
                { name: 'Linear Algebra', key: 'np_linear_algebra' }
            ]
        },
        {
            name: 'Python Basics',
            examples: [
                { name: 'Variables & Types', key: 'py_basics' },
                { name: 'Lists & Dictionaries', key: 'py_collections' },
                { name: 'Functions', key: 'py_functions' },
                { name: 'Classes & OOP', key: 'py_classes' },
                { name: 'File I/O', key: 'py_file_io' },
                { name: 'List Comprehensions', key: 'py_list_comp' },
                { name: 'Decorators', key: 'py_decorators' },
                { name: 'Generators', key: 'py_generators' },
                { name: 'Context Managers', key: 'py_context_managers' },
                { name: 'Exception Handling', key: 'py_exceptions' }
            ]
        },
        {
            name: 'Advanced Python',
            examples: [
                { name: 'Regular Expressions', key: 'py_regex' },
                { name: 'JSON & APIs', key: 'py_json_api' },
                { name: 'Multi-threading', key: 'py_threading' },
                { name: 'Multi-processing', key: 'py_multiprocessing' },
                { name: 'Async/Await', key: 'py_async' },
                { name: 'Type Hints', key: 'py_type_hints' },
                { name: 'Property Decorators', key: 'py_property' },
                { name: 'Magic Methods', key: 'py_magic_methods' }
            ]
        },
        {
            name: 'Data Structures & Algorithms',
            examples: [
                { name: 'Binary Search', key: 'algo_binary_search' },
                { name: 'Sorting Algorithms', key: 'algo_sorting' },
                { name: 'Linked Lists', key: 'algo_linked_list' },
                { name: 'Stacks & Queues', key: 'algo_stack_queue' },
                { name: 'Trees & BST', key: 'algo_trees' },
                { name: 'Graph Algorithms', key: 'algo_graphs' },
                { name: 'Dynamic Programming', key: 'algo_dp' },
                { name: 'Hash Tables', key: 'algo_hash' }
            ]
        },
        {
            name: 'Time Series & Finance',
            examples: [
                { name: 'Time Series Analysis', key: 'ts_analysis' },
                { name: 'Moving Averages', key: 'ts_moving_avg' },
                { name: 'Stock Price Analysis', key: 'finance_stocks' },
                { name: 'Portfolio Optimization', key: 'finance_portfolio' }
            ]
        },
        {
            name: 'Text Processing & NLP',
            examples: [
                { name: 'Text Preprocessing', key: 'nlp_preprocess' },
                { name: 'Word Frequency', key: 'nlp_word_freq' },
                { name: 'Sentiment Analysis', key: 'nlp_sentiment' },
                { name: 'TF-IDF Vectorization', key: 'nlp_tfidf' }
            ]
        },
        {
            name: 'Web Scraping & Automation',
            examples: [
                { name: 'Web Scraping Basics', key: 'web_scraping' },
                { name: 'CSV Processing', key: 'data_csv' },
                { name: 'Excel Operations', key: 'data_excel' },
                { name: 'Email Automation', key: 'auto_email' }
            ]
        }
    ];
    
    const menu = document.createElement('div');
    menu.className = 'example-menu';
    
    const header = document.createElement('h4');
    header.innerHTML = '<i class="fas fa-brain"></i> Python & Machine Learning Examples';
    menu.appendChild(header);
    
    const categoriesContainer = document.createElement('div');
    categoriesContainer.className = 'example-categories';
    
    exampleCategories.forEach(category => {
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'example-category';
        
        const categoryHeader = document.createElement('div');
        categoryHeader.className = 'category-header';
        categoryHeader.innerHTML = `${category.name} <i class="fas fa-chevron-down"></i>`;
        
        const exampleGrid = document.createElement('div');
        exampleGrid.className = 'example-grid';
        exampleGrid.style.display = 'grid'; // Show by default
        
        categoryHeader.onclick = function() {
            this.classList.toggle('collapsed');
            exampleGrid.style.display = this.classList.contains('collapsed') ? 'none' : 'grid';
        };
        
        category.examples.forEach(ex => {
            const btn = document.createElement('button');
            btn.className = 'btn btn-secondary btn-sm example-item';
            btn.textContent = ex.name;
            btn.onclick = () => {
                console.log('=== Example button clicked ===');
                console.log('Example name:', ex.name);
                console.log('Example key:', ex.key);
                console.log('loadPythonExample function exists:', typeof loadPythonExample);
                loadPythonExample(ex.key);
                menu.remove();
            };
            exampleGrid.appendChild(btn);
        });
        
        categoryDiv.appendChild(categoryHeader);
        categoryDiv.appendChild(exampleGrid);
        categoriesContainer.appendChild(categoryDiv);
    });
    
    menu.appendChild(categoriesContainer);
    
    const closeBtn = document.createElement('button');
    closeBtn.className = 'btn btn-secondary';
    closeBtn.textContent = 'Cancel';
    closeBtn.onclick = () => menu.remove();
    menu.appendChild(closeBtn);
    
    document.body.appendChild(menu);
}

function loadPythonExample(key) {
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('📥 [REQUEST RECEIVED] loadPythonExample called');
    console.log('   Method: Direct function call (not HTTP)');
    console.log('   Trigger: onclick event handler');
    console.log('   Parameter key:', key);
    console.log('   Timestamp:', new Date().toISOString());
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    
    console.log('loadPythonExample called with key:', key);
    const examples = {
        'ml_linear_regression': `# Linear Regression Example with Visualization
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + np.random.randn(100, 1) * 2

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model R² Score: {score:.4f}")
print(f"Coefficients: {model.coef_[0][0]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training data')
plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Linear Regression (R² = {score:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
`,
        'ml_logistic_regression': `# Logistic Regression Example
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Use only 2 classes for binary classification
X = X[y != 2]
y = y[y != 2]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names[:2]))
`,
        'ml_decision_trees': `# Decision Tree Classifier Example
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# Predict
y_pred = dt.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Tree Depth: {dt.get_depth()}")
print(f"Number of Leaves: {dt.get_n_leaves()}")
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
`,
        'ml_random_forest': `# Random Forest Classifier Example
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Number of Trees: {rf.n_estimators}")

# Feature importance
print("\\nTop 5 Important Features:")
feature_importance = sorted(zip(wine.feature_names, rf.feature_importances_), 
                           key=lambda x: x[1], reverse=True)[:5]
for name, importance in feature_importance:
    print(f"{name}: {importance:.4f}")
`,
        'ml_kmeans': `# K-Means Clustering Example with Visualization
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# Evaluate
silhouette_avg = silhouette_score(X, y_pred)
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"\\nCluster Centers:")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i}: [{center[0]:.2f}, {center[1]:.2f}]")

# Count samples per cluster
unique, counts = np.unique(y_pred, return_counts=True)
print(f"\\nSamples per cluster: {dict(zip(unique, counts))}")

# Visualize clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=300, edgecolors='black', linewidths=2, 
            label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'K-Means Clustering (Silhouette Score = {silhouette_avg:.4f})')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
`,
        'ml_svm': `# Support Vector Machine (SVM) Example
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)

# Predict
y_pred = svm.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Support Vectors: {len(svm.support_vectors_)}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
`,
        'dl_neural_network': `# Simple Neural Network with TensorFlow/Keras
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Note: TensorFlow may not be installed
# This is a demonstration of the code structure

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Dataset prepared:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")

# Uncomment to use with TensorFlow:
# import tensorflow as tf
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)
`,
    'dl_cnn': `# CNN Example (Image Classification) using TensorFlow/Keras
# Note: TensorFlow required; this is a minimal illustrative example
import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers

# Example structure only — real training requires image data
# model = keras.Sequential([
#     layers.Input(shape=(64,64,3)),
#     layers.Conv2D(32, (3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# print(model.summary())
print('CNN example (structure) — uncomment TensorFlow imports to run')
`,
    'dl_rnn': `# RNN / LSTM Example for Time Series (Keras)
# Note: TensorFlow required for real runs
import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers

# Simple synthetic sequence example
def create_sequences(n_samples=200, seq_len=20):
    X = np.random.randn(n_samples, seq_len, 1)
    y = (X.mean(axis=1) > 0).astype(int)
    return X, y

X, y = create_sequences()
print('Prepared synthetic sequences:', X.shape)
# model = keras.Sequential([
#     layers.LSTM(32, input_shape=(X.shape[1], X.shape[2])),
#     layers.Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(X, y, epochs=10, validation_split=0.2, verbose=0)
print('RNN example (structure) — uncomment TensorFlow imports to run')
`,
    'dl_transfer_learning': `# Transfer Learning Example (MobileNetV2) with Keras
# Note: TensorFlow required. This snippet shows the pattern.
# from tensorflow import keras
# from tensorflow.keras import layers, applications

# base_model = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(160,160,3))
# base_model.trainable = False
# model = keras.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print('Transfer learning example (MobileNetV2) — uncomment to use TensorFlow')
`,
        'ds_pandas_basics': `# Pandas DataFrame Basics
import pandas as pd
import numpy as np

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'],
    'Salary': [50000, 60000, 55000, 65000, 58000]
}
df = pd.DataFrame(data)

print("DataFrame:")
print(df)
print("\\nDataFrame Info:")
print(df.info())
print("\\nBasic Statistics:")
print(df.describe())
print("\\nColumn Names:")
print(df.columns.tolist())
print("\\nFirst 3 rows:")
print(df.head(3))
print("\\nFiltering (Age > 28):")
print(df[df['Age'] > 28])
`,
        'ds_data_cleaning': `# Data Cleaning Example
import pandas as pd
import numpy as np

# Create dataset with missing values
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, np.nan, 5],
    'D': ['a', 'b', 'c', 'd', 'e']
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print(f"\\nMissing values per column:")
print(df.isnull().sum())

# Fill missing values
df_filled = df.copy()
df_filled['A'].fillna(df_filled['A'].mean(), inplace=True)
df_filled['B'].fillna(df_filled['B'].median(), inplace=True)
df_filled['C'].fillna(0, inplace=True)

print("\\nCleaned DataFrame:")
print(df_filled)

# Remove duplicates
df_no_duplicates = df_filled.drop_duplicates()
print(f"\\nShape after removing duplicates: {df_no_duplicates.shape}")
`,
        'ds_visualization': `# Data Visualization Example
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100)
data = pd.DataFrame({
    'Date': dates,
    'Sales': np.random.randint(100, 500, 100),
    'Revenue': np.random.randint(1000, 5000, 100)
})

print("Sales Data:")
print(data.head(10))
print("\\nSummary Statistics:")
print(data[['Sales', 'Revenue']].describe())

# Calculate correlations
correlation = data[['Sales', 'Revenue']].corr()
print("\\nCorrelation Matrix:")
print(correlation)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Line plot
axes[0, 0].plot(data['Date'], data['Sales'], label='Sales', color='blue')
axes[0, 0].set_title('Sales Over Time')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Sales')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Bar plot for monthly average
data['Month'] = data['Date'].dt.to_period('M')
monthly = data.groupby('Month')[['Sales', 'Revenue']].mean()
monthly.plot(kind='bar', ax=axes[0, 1], color=['skyblue', 'lightcoral'])
axes[0, 1].set_title('Average Monthly Sales & Revenue')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Value')
axes[0, 1].legend()

# Scatter plot
axes[1, 0].scatter(data['Sales'], data['Revenue'], alpha=0.5, color='green')
axes[1, 0].set_title('Sales vs Revenue')
axes[1, 0].set_xlabel('Sales')
axes[1, 0].set_ylabel('Revenue')
axes[1, 0].grid(True, alpha=0.3)

# Histogram
axes[1, 1].hist(data['Sales'], bins=20, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Sales Distribution')
axes[1, 1].set_xlabel('Sales')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\nVisualization created successfully!")
`,
        'ds_statistics': `# Statistical Analysis Example
import pandas as pd
import numpy as np
from scipy import stats

# Generate sample data
np.random.seed(42)
group_a = np.random.normal(100, 15, 50)
group_b = np.random.normal(105, 15, 50)

# Descriptive statistics
print("Group A Statistics:")
print(f"Mean: {np.mean(group_a):.2f}")
print(f"Median: {np.median(group_a):.2f}")
print(f"Std Dev: {np.std(group_a):.2f}")
print(f"Variance: {np.var(group_a):.2f}")

print("\\nGroup B Statistics:")
print(f"Mean: {np.mean(group_b):.2f}")
print(f"Median: {np.median(group_b):.2f}")
print(f"Std Dev: {np.std(group_b):.2f}")

# T-test
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"\\nT-test Results:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
`,
        'ds_feature_engineering': `# Feature Engineering Example
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Create sample dataset
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 75000, 80000, 90000],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
    'years_exp': [2, 5, 10, 15, 20]
})

print("Original Data:")
print(data)

# Create new features
data['income_per_year'] = data['income'] / data['years_exp']
data['age_group'] = pd.cut(data['age'], bins=[0, 30, 40, 100], labels=['Young', 'Middle', 'Senior'])

print("\\nWith New Features:")
print(data)

# Encode categorical variables
le = LabelEncoder()
data['education_encoded'] = le.fit_transform(data['education'])

# Scale numerical features
scaler = StandardScaler()
data[['age_scaled', 'income_scaled']] = scaler.fit_transform(data[['age', 'income']])

print("\\nWith Encoding and Scaling:")
print(data)
`,
        'np_array_ops': `# NumPy Array Operations
import numpy as np

# Create arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])

print("Array 1:", arr1)
print("Array 2:", arr2)

# Basic operations
print("\\nAddition:", arr1 + arr2)
print("Multiplication:", arr1 * arr2)
print("Power:", arr1 ** 2)

# Statistical operations
print("\\nMean:", arr1.mean())
print("Sum:", arr1.sum())
print("Min:", arr1.min())
print("Max:", arr1.max())
print("Std Dev:", arr1.std())

# Array manipulation
print("\\nReshaped (5,1):")
print(arr1.reshape(5, 1))

# 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\\n2D Array:")
print(arr_2d)
print("Shape:", arr_2d.shape)
print("Sum of columns:", arr_2d.sum(axis=0))
print("Sum of rows:", arr_2d.sum(axis=1))
`,
        'np_matrix_ops': `# NumPy Matrix Operations
import numpy as np

# Create matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix A:")
print(A)
print("\\nMatrix B:")
print(B)

# Matrix operations
print("\\nMatrix Multiplication (A @ B):")
print(A @ B)

print("\\nElement-wise Multiplication (A * B):")
print(A * B)

print("\\nTranspose of A:")
print(A.T)

print("\\nDeterminant of A:")
print(np.linalg.det(A))

print("\\nInverse of A:")
print(np.linalg.inv(A))

print("\\nEigenvalues and Eigenvectors of A:")
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
`,
        'np_broadcasting': `# NumPy Broadcasting Example
import numpy as np

# Broadcasting with 1D arrays
a = np.array([1, 2, 3])
b = 10
print("Array:", a)
print("Scalar:", b)
print("Broadcasting (a + b):", a + b)

# Broadcasting with 2D arrays
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row = np.array([10, 20, 30])

print("\\nMatrix:")
print(matrix)
print("\\nRow vector:", row)
print("\\nBroadcasting (matrix + row):")
print(matrix + row)

# Column broadcasting
col = np.array([[100], [200], [300]])
print("\\nColumn vector:")
print(col)
print("\\nBroadcasting (matrix + col):")
print(matrix + col)

# Complex broadcasting
result = matrix * row + col
print("\\nComplex broadcasting (matrix * row + col):")
print(result)
`,
        'np_linear_algebra': `# Linear Algebra with NumPy
import numpy as np

# System of linear equations: Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

print("Matrix A:")
print(A)
print("\\nVector b:", b)

# Solve linear system
x = np.linalg.solve(A, b)
print("\\nSolution x:", x)
print("Verification (A @ x):", A @ x)

# Matrix properties
print("\\nMatrix Properties:")
print("Determinant:", np.linalg.det(A))
print("Rank:", np.linalg.matrix_rank(A))
print("Condition number:", np.linalg.cond(A))

# Norms
print("\\nNorms:")
print("L1 norm of b:", np.linalg.norm(b, 1))
print("L2 norm of b:", np.linalg.norm(b, 2))
print("Infinity norm of b:", np.linalg.norm(b, np.inf))

# SVD
U, s, Vt = np.linalg.svd(A)
print("\\nSingular Values:", s)
`,
        'py_basics': `# Python Basics - Variables & Types
# This is a comment

# Variables and basic types
name = "Python"
version = 3.11
is_awesome = True
pi = 3.14159

print(f"Language: {name}")
print(f"Version: {version}")
print(f"Is awesome? {is_awesome}")
print(f"Pi value: {pi}")

# Type checking
print(f"\\nType of name: {type(name)}")
print(f"Type of version: {type(version)}")
print(f"Type of is_awesome: {type(is_awesome)}")

# Basic operations
x = 10
y = 3
print(f"\\nMath Operations:")
print(f"{x} + {y} = {x + y}")
print(f"{x} - {y} = {x - y}")
print(f"{x} * {y} = {x * y}")
print(f"{x} / {y} = {x / y:.2f}")
print(f"{x} // {y} = {x // y}")
print(f"{x} % {y} = {x % y}")
print(f"{x} ** {y} = {x ** y}")

# String operations
greeting = "Hello"
target = "World"
message = greeting + " " + target + "!"
print(f"\\n{message}")
print(f"Length: {len(message)}")
print(f"Uppercase: {message.upper()}")
print(f"Lowercase: {message.lower()}")
`,
        'py_collections': `# Python Collections - Lists & Dictionaries

# Lists
fruits = ["apple", "banana", "orange", "grape"]
print("Fruits:", fruits)
print("First fruit:", fruits[0])
print("Last fruit:", fruits[-1])

# List operations
fruits.append("mango")
print("After append:", fruits)
fruits.remove("banana")
print("After remove:", fruits)
print("Length:", len(fruits))

# List comprehension
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
print("\\nNumbers:", numbers)
print("Squares:", squares)

# Dictionaries
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York",
    "skills": ["Python", "Machine Learning", "Data Science"]
}
print("\\nPerson:", person)
print("Name:", person["name"])
print("Skills:", person["skills"])

# Dictionary operations
person["email"] = "alice@example.com"
print("\\nAfter adding email:", person)

# Iterating
print("\\nIterating over dictionary:")
for key, value in person.items():
    print(f"{key}: {value}")
`,
        'py_functions': `# Python Functions

# Simple function
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
print(greet("Bob"))

# Function with default parameters
def power(base, exponent=2):
    return base ** exponent

print(f"\\n2^3 = {power(2, 3)}")
print(f"5^2 = {power(5)}")

# Function with multiple return values
def calculate_stats(numbers):
    total = sum(numbers)
    average = total / len(numbers)
    maximum = max(numbers)
    minimum = min(numbers)
    return total, average, maximum, minimum

data = [10, 20, 30, 40, 50]
total, avg, max_val, min_val = calculate_stats(data)
print(f"\\nData: {data}")
print(f"Total: {total}")
print(f"Average: {avg}")
print(f"Max: {max_val}")
print(f"Min: {min_val}")

# Lambda functions
square = lambda x: x ** 2
add = lambda x, y: x + y

print(f"\\nSquare of 5: {square(5)}")
print(f"Add 3 and 7: {add(3, 7)}")

# Map and filter
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

print(f"\\nOriginal: {numbers}")
print(f"Doubled: {doubled}")
print(f"Even numbers: {evens}")
`,
        'py_classes': `# Python Classes and OOP

# Define a class
class Dog:
    # Class variable
    species = "Canis familiaris"
    
    def __init__(self, name, age):
        # Instance variables
        self.name = name
        self.age = age
    
    def bark(self):
        return f"{self.name} says Woof!"
    
    def get_info(self):
        return f"{self.name} is {self.age} years old"

# Create objects
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

print(dog1.bark())
print(dog2.bark())
print(dog1.get_info())
print(dog2.get_info())
print(f"Species: {Dog.species}")

# Inheritance
class GoldenRetriever(Dog):
    def __init__(self, name, age, color):
        super().__init__(name, age)
        self.color = color
    
    def fetch(self):
        return f"{self.name} is fetching the ball!"
    
    def get_info(self):
        base_info = super().get_info()
        return f"{base_info} and has {self.color} fur"

golden = GoldenRetriever("Charlie", 4, "golden")
print(f"\\n{golden.get_info()}")
print(golden.bark())
print(golden.fetch())
`,
        'py_file_io': `# Python File I/O Example

# Writing to a file
content = """This is a sample text file.
It contains multiple lines.
Python makes file handling easy!"""

try:
    # Write to file
    with open('sample.txt', 'w') as file:
        file.write(content)
    print("File written successfully!")
    
    # Read from file
    with open('sample.txt', 'r') as file:
        data = file.read()
    print("\\nFile contents:")
    print(data)
    
    # Read line by line
    print("\\nReading line by line:")
    with open('sample.txt', 'r') as file:
        for i, line in enumerate(file, 1):
            print(f"Line {i}: {line.strip()}")
    
    # Append to file
    with open('sample.txt', 'a') as file:
        file.write("\\nThis line was appended!")
    
    print("\\nAfter appending:")
    with open('sample.txt', 'r') as file:
        print(file.read())
    
except IOError as e:
    print(f"An error occurred: {e}")

print("\\nNote: File operations work with proper permissions.")
`,
        'ml_naive_bayes': `# Naive Bayes Classification
import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predict
y_pred = nb.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"\\nClass Priors: {nb.class_prior_}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\\nConfusion Matrix:\\n{cm}")

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title(f'Naive Bayes Confusion Matrix (Accuracy: {accuracy:.4f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
`,
        'ml_knn': `# K-Nearest Neighbors (KNN) Classification
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Use only first 4 classes for simplicity
mask = y < 4
X, y = X[mask], y[mask]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test different K values
k_values = range(1, 21)
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

# Best K
best_k = k_values[np.argmax(test_scores)]
print(f"Best K: {best_k}")
print(f"Best Test Accuracy: {max(test_scores):.4f}")

# Train final model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot K vs Accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, 'o-', label='Training Score', linewidth=2)
plt.plot(k_values, test_scores, 's-', label='Test Score', linewidth=2)
plt.axvline(best_k, color='red', linestyle='--', label=f'Best K = {best_k}')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('KNN: Model Performance vs K Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
`,
        'ml_gradient_boosting': `# Gradient Boosting Classifier
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                max_depth=3, random_state=42)
gb.fit(X_train, y_train)

# Predict
y_pred = gb.predict(X_test)
y_pred_proba = gb.predict_proba(X_test)[:, 1]

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"\\nFeature Importance (Top 5):")
feature_importance = sorted(zip(range(20), gb.feature_importances_), 
                           key=lambda x: x[1], reverse=True)[:5]
for idx, importance in feature_importance:
    print(f"Feature {idx}: {importance:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('Receiver Operating Characteristic (ROC)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Feature Importance
importances = gb.feature_importances_
indices = np.argsort(importances)[-10:]
axes[1].barh(range(len(indices)), importances[indices], color='skyblue')
axes[1].set_yticks(range(len(indices)))
axes[1].set_yticklabels([f'Feature {i}' for i in indices])
axes[1].set_xlabel('Importance')
axes[1].set_title('Top 10 Feature Importances')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
`,
        'ml_adaboost': `# AdaBoost Classifier
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train AdaBoost with different number of estimators
n_estimators_list = [10, 25, 50, 100, 200]
train_scores = []
test_scores = []

for n in n_estimators_list:
    ada = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n,
        random_state=42
    )
    ada.fit(X_train, y_train)
    train_scores.append(ada.score(X_train, y_train))
    test_scores.append(ada.score(X_test, y_test))

# Best model
best_n = n_estimators_list[np.argmax(test_scores)]
ada_best = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=best_n,
    random_state=42
)
ada_best.fit(X_train, y_train)
y_pred = ada_best.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Best n_estimators: {best_n}")
print(f"Best Accuracy: {accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\\nConfusion Matrix:\\n{cm}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Performance vs n_estimators
axes[0].plot(n_estimators_list, train_scores, 'o-', label='Training', linewidth=2)
axes[0].plot(n_estimators_list, test_scores, 's-', label='Testing', linewidth=2)
axes[0].axvline(best_n, color='red', linestyle='--', label=f'Best n={best_n}')
axes[0].set_xlabel('Number of Estimators')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('AdaBoost Performance')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
axes[1].set_title(f'Confusion Matrix (Acc: {accuracy:.4f})')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()
`,
        'ml_pca': `# Principal Component Analysis (PCA)
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_var = pca.explained_variance_ratio_
cumsum_var = np.cumsum(explained_var)

print("Explained Variance Ratio:")
for i, var in enumerate(explained_var):
    print(f"PC{i+1}: {var:.4f} ({cumsum_var[i]:.4f} cumulative)")

# 2D PCA for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, len(explained_var) + 1), explained_var, 
           alpha=0.7, label='Individual', color='skyblue')
axes[0].plot(range(1, len(explained_var) + 1), cumsum_var, 
            'ro-', linewidth=2, label='Cumulative')
axes[0].axhline(y=0.95, color='green', linestyle='--', label='95% threshold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('PCA Scree Plot')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2D projection
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    mask = y == i
    axes[1].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                   c=color, label=iris.target_names[i], 
                   alpha=0.7, edgecolors='k')
axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
axes[1].set_title('2D PCA Projection')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
`,
        'ml_dbscan': `# DBSCAN Clustering
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate sample data (non-linear clusters)
X, y_true = make_moons(n_samples=300, noise=0.1, random_state=42)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Statistics
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"\\nCluster sizes:")
unique, counts = np.unique(labels[labels != -1], return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"Cluster {cluster_id}: {count} points")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original data
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', 
               alpha=0.6, edgecolors='k')
axes[0].set_title('Original Data (True Labels)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].grid(True, alpha=0.3)

# DBSCAN results
scatter = axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                         alpha=0.6, edgecolors='k')
# Highlight noise points
noise_mask = labels == -1
if n_noise > 0:
    axes[1].scatter(X[noise_mask, 0], X[noise_mask, 1], 
                   c='red', marker='x', s=100, label='Noise')
axes[1].set_title(f'DBSCAN Clustering ({n_clusters} clusters, {n_noise} noise)')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes[1], label='Cluster')
plt.tight_layout()
plt.show()
`,
        'ml_cross_validation': `# Cross-Validation Example
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target

# Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Different CV strategies
cv_strategies = {
    'KFold (5)': KFold(n_splits=5, shuffle=True, random_state=42),
    'KFold (10)': KFold(n_splits=10, shuffle=True, random_state=42),
    'StratifiedKFold (5)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
}

results = {}
for name, cv in cv_strategies.items():
    scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    results[name] = scores
    print(f"{name}:")
    print(f"  Scores: {scores}")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print()

# Detailed cross-validation
cv_results = cross_validate(rf, X, y, cv=5, 
                           scoring=['accuracy', 'precision_macro', 'recall_macro'],
                           return_train_score=True)

print("Detailed 5-Fold CV Results:")
print(f"Train Accuracy: {cv_results['train_accuracy'].mean():.4f}")
print(f"Test Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Test Precision: {cv_results['test_precision_macro'].mean():.4f}")
print(f"Test Recall: {cv_results['test_recall_macro'].mean():.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot of CV scores
axes[0].boxplot([results[name] for name in results.keys()], 
               labels=list(results.keys()))
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Cross-Validation Score Distribution')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].tick_params(axis='x', rotation=15)

# Train vs Test scores
fold_nums = range(1, 6)
axes[1].plot(fold_nums, cv_results['train_accuracy'], 'o-', 
            label='Training', linewidth=2, markersize=8)
axes[1].plot(fold_nums, cv_results['test_accuracy'], 's-', 
            label='Testing', linewidth=2, markersize=8)
axes[1].axhline(cv_results['test_accuracy'].mean(), 
               color='red', linestyle='--', label='Mean Test')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Train vs Test Accuracy per Fold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
`,
        'ml_grid_search': `# Grid Search for Hyperparameter Tuning
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

# Grid Search
print("Performing Grid Search...")
grid_search = GridSearchCV(SVC(), param_grid, cv=5, 
                          scoring='accuracy', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

# Results
print(f"\\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")

# Top 5 combinations
results = grid_search.cv_results_
top_indices = np.argsort(results['mean_test_score'])[-5:][::-1]
print("\\nTop 5 Parameter Combinations:")
for i, idx in enumerate(top_indices, 1):
    print(f"{i}. Score: {results['mean_test_score'][idx]:.4f}, "
          f"Params: {results['params'][idx]}")

# Visualize for RBF kernel
rbf_results = [(res['params'], res['mean_test_score']) 
               for res in [dict(zip(results.keys(), values)) 
                          for values in zip(*results.values())]
               if res['params']['kernel'] == 'rbf']

C_values = sorted(set(r[0]['C'] for r in rbf_results))
gamma_values = sorted(set(r[0]['gamma'] for r in rbf_results))

# Create heatmap data
heatmap_data = np.zeros((len(gamma_values), len(C_values)))
for params, score in rbf_results:
    i = gamma_values.index(params['gamma'])
    j = C_values.index(params['C'])
    heatmap_data[i, j] = score

# Plot heatmap
plt.figure(figsize=(10, 8))
im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
plt.colorbar(im, label='Mean CV Accuracy')
plt.xticks(range(len(C_values)), C_values)
plt.yticks(range(len(gamma_values)), gamma_values)
plt.xlabel('C (Regularization)')
plt.ylabel('Gamma')
plt.title('Grid Search Results (RBF Kernel)\\nCV Accuracy Heatmap')

# Annotate cells with values
for i in range(len(gamma_values)):
    for j in range(len(C_values)):
        text = plt.text(j, i, f'{heatmap_data[i, j]:.3f}',
                       ha="center", va="center", color="white", fontsize=9)

plt.tight_layout()
plt.show()
`,
        'ml_feature_selection': `# Feature Selection Techniques
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Original features: {X.shape[1]}")

# Method 1: SelectKBest (Univariate)
selector_kbest = SelectKBest(f_classif, k=10)
X_train_kbest = selector_kbest.fit_transform(X_train, y_train)
X_test_kbest = selector_kbest.transform(X_test)

# Method 2: RFE (Recursive Feature Elimination)
rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Method 3: Random Forest Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
top_10_idx = np.argsort(importances)[-10:]
X_train_rf = X_train[:, top_10_idx]
X_test_rf = X_test[:, top_10_idx]

# Compare performance
methods = {
    'All Features': (X_train, X_test),
    'SelectKBest': (X_train_kbest, X_test_kbest),
    'RFE': (X_train_rfe, X_test_rfe),
    'RF Importance': (X_train_rf, X_test_rf)
}

results = {}
for name, (X_tr, X_te) in methods.items():
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_tr, y_train)
    score = clf.score(X_te, y_test)
    results[name] = score
    print(f"{name}: {score:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Method comparison
axes[0].bar(results.keys(), results.values(), color=['gray', 'skyblue', 'lightgreen', 'coral'])
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Feature Selection Methods Comparison')
axes[0].tick_params(axis='x', rotation=15)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(results['All Features'], color='red', linestyle='--', label='Baseline')
axes[0].legend()

# Feature importance
top_indices = np.argsort(importances)[-15:]
axes[1].barh(range(len(top_indices)), importances[top_indices], color='steelblue')
axes[1].set_yticks(range(len(top_indices)))
axes[1].set_yticklabels([cancer.feature_names[i] for i in top_indices], fontsize=8)
axes[1].set_xlabel('Importance')
axes[1].set_title('Top 15 Feature Importances (Random Forest)')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
`,
        'ml_ensemble': `# Ensemble Methods - Voting Classifier
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

# Load data
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Individual classifiers
clf1 = LogisticRegression(max_iter=1000, random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = SVC(kernel='rbf', probability=True, random_state=42)
clf4 = DecisionTreeClassifier(max_depth=5, random_state=42)

# Hard voting ensemble
voting_hard = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3), ('dt', clf4)],
    voting='hard'
)

# Soft voting ensemble
voting_soft = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3), ('dt', clf4)],
    voting='soft'
)

# Train and evaluate all models
models = {
    'Logistic Regression': clf1,
    'Random Forest': clf2,
    'SVM': clf3,
    'Decision Tree': clf4,
    'Hard Voting': voting_hard,
    'Soft Voting': voting_soft
}

results = {}
cv_scores = {}

print("Model Performance:")
for name, model in models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    
    results[name] = {'train': train_score, 'test': test_score, 'cv': cv_score}
    print(f"{name}:")
    print(f"  Train: {train_score:.4f}, Test: {test_score:.4f}, CV: {cv_score:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Performance comparison
model_names = list(results.keys())
test_scores = [results[name]['test'] for name in model_names]
cv_scores = [results[name]['cv'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

axes[0].bar(x - width/2, test_scores, width, label='Test Score', color='skyblue')
axes[0].bar(x + width/2, cv_scores, width, label='CV Score', color='lightcoral')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Improvement over individual classifiers
individual_scores = [results[name]['test'] for name in list(results.keys())[:4]]
ensemble_scores = [results[name]['test'] for name in ['Hard Voting', 'Soft Voting']]
mean_individual = np.mean(individual_scores)

axes[1].bar(['Mean Individual', 'Hard Voting', 'Soft Voting'], 
           [mean_individual] + ensemble_scores,
           color=['gray', 'lightgreen', 'lightblue'])
axes[1].axhline(mean_individual, color='red', linestyle='--', linewidth=2, 
               label='Mean Individual')
axes[1].set_ylabel('Test Accuracy')
axes[1].set_title('Ensemble vs Individual Classifiers')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
`,
        'ml_pipeline': `# ML Pipeline Creation
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameter grid for pipeline
param_grid = {
    'feature_selection__k': [5, 10, 15, 20],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7, None]
}

# Grid search on pipeline
print("Performing Grid Search on Pipeline...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, 
                          scoring='accuracy', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

# Results
print(f"\\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")

# Get feature selection results
best_pipeline = grid_search.best_estimator_
selected_features = best_pipeline.named_steps['feature_selection'].get_support()
print(f"\\nNumber of selected features: {selected_features.sum()}")
print(f"Selected feature names:")
for i, selected in enumerate(selected_features):
    if selected:
        print(f"  - {cancer.feature_names[i]}")

# Compare with different k values
k_values = [5, 10, 15, 20, 30]
scores = []

for k in k_values:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=min(k, X.shape[1]))),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    scores.append(pipe.score(X_test, y_test))

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(k_values, scores, 'o-', linewidth=2, markersize=10, color='steelblue')
best_k = grid_search.best_params_['feature_selection__k']
best_score = grid_search.score(X_test, y_test)
plt.axvline(best_k, color='red', linestyle='--', linewidth=2, 
           label=f'Best k={best_k}')
plt.axhline(best_score, color='green', linestyle='--', linewidth=1, alpha=0.5)
plt.xlabel('Number of Features (k)')
plt.ylabel('Test Accuracy')
plt.title('Pipeline Performance vs Number of Features')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\\nPipeline steps: {[name for name, _ in pipeline.steps]}")
`,
        'ml_ridge': `# Ridge Regression (L2 Regularization)
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate data with noise
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_scores = []
test_scores = []
coefficients = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    train_scores.append(ridge.score(X_train, y_train))
    test_scores.append(ridge.score(X_test, y_test))
    coefficients.append(ridge.coef_[0])

# Best alpha
best_alpha = alphas[np.argmax(test_scores)]
print(f"Best alpha: {best_alpha}")
print(f"Best R² score: {max(test_scores):.4f}")

# Compare with Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
print(f"\\nLinear Regression R²: {lr_score:.4f}")
print(f"Ridge Regression R²: {max(test_scores):.4f}")

# Best Ridge model
best_ridge = Ridge(alpha=best_alpha)
best_ridge.fit(X_train, y_train)
y_pred_ridge = best_ridge.predict(X_test)
y_pred_lr = lr.predict(X_test)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Alpha vs Score
axes[0].semilogx(alphas, train_scores, 'o-', label='Training', linewidth=2)
axes[0].semilogx(alphas, test_scores, 's-', label='Testing', linewidth=2)
axes[0].axvline(best_alpha, color='red', linestyle='--', label=f'Best α={best_alpha}')
axes[0].set_xlabel('Alpha (Regularization Strength)')
axes[0].set_ylabel('R² Score')
axes[0].set_title('Ridge Regression: Alpha Tuning')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Predictions comparison
axes[1].scatter(X_test, y_test, color='black', alpha=0.5, label='True values')
sorted_idx = X_test.flatten().argsort()
axes[1].plot(X_test[sorted_idx], y_pred_lr[sorted_idx], 
            color='blue', linewidth=2, label='Linear Regression')
axes[1].plot(X_test[sorted_idx], y_pred_ridge[sorted_idx], 
            color='red', linewidth=2, label=f'Ridge (α={best_alpha})')
axes[1].set_xlabel('X')
axes[1].set_ylabel('y')
axes[1].set_title('Predictions Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
`,
        'ml_lasso': `# Lasso Regression (L1 Regularization)
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate data with multiple features
X, y = make_regression(n_samples=100, n_features=10, n_informative=5, 
                      noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test different alpha values
alphas = np.logspace(-4, 2, 50)
coefs = []
train_scores = []
test_scores = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
    train_scores.append(lasso.score(X_train, y_train))
    test_scores.append(lasso.score(X_test, y_test))

# Best alpha
best_alpha = alphas[np.argmax(test_scores)]
best_lasso = Lasso(alpha=best_alpha, max_iter=10000)
best_lasso.fit(X_train, y_train)

print(f"Best alpha: {best_alpha:.4f}")
print(f"Best R² score: {max(test_scores):.4f}")
print(f"\\nFeature Coefficients:")
for i, coef in enumerate(best_lasso.coef_):
    print(f"Feature {i}: {coef:.4f}" + (" (eliminated)" if abs(coef) < 0.001 else ""))

# Compare with Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"\\nLinear Regression R²: {lr.score(X_test, y_test):.4f}")
print(f"Lasso R²: {max(test_scores):.4f}")
print(f"Number of non-zero coefficients: {np.sum(np.abs(best_lasso.coef_) > 0.001)}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Coefficient paths
coefs = np.array(coefs)
for i in range(coefs.shape[1]):
    axes[0].semilogx(alphas, coefs[:, i], label=f'Feature {i}')
axes[0].axvline(best_alpha, color='red', linestyle='--', linewidth=2, 
               label=f'Best α={best_alpha:.4f}')
axes[0].set_xlabel('Alpha (Regularization Strength)')
axes[0].set_ylabel('Coefficient Value')
axes[0].set_title('Lasso Coefficient Paths')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[0].grid(True, alpha=0.3)

# Alpha vs Score
axes[1].semilogx(alphas, train_scores, label='Training', linewidth=2)
axes[1].semilogx(alphas, test_scores, label='Testing', linewidth=2)
axes[1].axvline(best_alpha, color='red', linestyle='--', 
               label=f'Best α={best_alpha:.4f}')
axes[1].set_xlabel('Alpha (Regularization Strength)')
axes[1].set_ylabel('R² Score')
axes[1].set_title('Lasso Performance vs Alpha')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
`,
        'ml_polynomial': `# Polynomial Regression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Generate non-linear data
np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = 0.5 * X**2 - 3 * X + np.random.randn(100, 1) * 5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test different polynomial degrees
degrees = [1, 2, 3, 4, 5, 8, 10]
train_scores = []
test_scores = []
models = []

for degree in degrees:
    # Create pipeline
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)
    models.append(model)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"Degree {degree}: Train R²={train_score:.4f}, Test R²={test_score:.4f}")

# Best degree
best_degree = degrees[np.argmax(test_scores)]
best_model = models[np.argmax(test_scores)]
print(f"\\nBest degree: {best_degree}")
print(f"Best test R² score: {max(test_scores):.4f}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Polynomial fits
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
for i, degree in enumerate([1, 2, 3, 5]):
    ax = axes[i // 2, i % 2]
    model = models[degrees.index(degree)]
    y_plot = model.predict(X_plot)
    
    ax.scatter(X_train, y_train, alpha=0.5, label='Training data', s=30)
    ax.scatter(X_test, y_test, alpha=0.5, label='Test data', s=30, color='red')
    ax.plot(X_plot, y_plot, linewidth=2, label=f'Degree {degree}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'Polynomial Degree {degree} (R²={test_scores[degrees.index(degree)]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_scores, 'o-', label='Training Score', linewidth=2, markersize=8)
plt.plot(degrees, test_scores, 's-', label='Test Score', linewidth=2, markersize=8)
plt.axvline(best_degree, color='red', linestyle='--', label=f'Best degree={best_degree}')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Model Performance vs Polynomial Degree')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
`,
        'ml_elasticnet': `# ElasticNet Regression (L1 + L2)
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate data
X, y = make_regression(n_samples=100, n_features=20, n_informative=10,
                      noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different l1_ratio values (0=Ridge, 1=Lasso)
l1_ratios = np.linspace(0, 1, 11)
alpha = 1.0

results = {'train': [], 'test': [], 'n_nonzero': []}

for l1_ratio in l1_ratios:
    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    en.fit(X_train_scaled, y_train)
    
    results['train'].append(en.score(X_train_scaled, y_train))
    results['test'].append(en.score(X_test_scaled, y_test))
    results['n_nonzero'].append(np.sum(np.abs(en.coef_) > 0.001))

# Best l1_ratio
best_l1_ratio = l1_ratios[np.argmax(results['test'])]
print(f"Best l1_ratio: {best_l1_ratio:.2f}")
print(f"Best test R² score: {max(results['test']):.4f}")

# Compare with pure Lasso and Ridge
lasso = Lasso(alpha=alpha, max_iter=10000)
ridge = Ridge(alpha=alpha)

lasso.fit(X_train_scaled, y_train)
ridge.fit(X_train_scaled, y_train)

print(f"\\nComparison (alpha={alpha}):")
print(f"Ridge R²: {ridge.score(X_test_scaled, y_test):.4f}")
print(f"ElasticNet R²: {max(results['test']):.4f}")
print(f"Lasso R²: {lasso.score(X_test_scaled, y_test):.4f}")

# Train best model
best_en = ElasticNet(alpha=alpha, l1_ratio=best_l1_ratio, max_iter=10000)
best_en.fit(X_train_scaled, y_train)

print(f"\\nNon-zero coefficients: {np.sum(np.abs(best_en.coef_) > 0.001)}/{len(best_en.coef_)}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Performance vs l1_ratio
axes[0].plot(l1_ratios, results['train'], 'o-', label='Training', linewidth=2)
axes[0].plot(l1_ratios, results['test'], 's-', label='Testing', linewidth=2)
axes[0].axvline(best_l1_ratio, color='red', linestyle='--', 
               label=f'Best l1_ratio={best_l1_ratio:.2f}')
axes[0].axvline(0, color='blue', linestyle=':', alpha=0.5, label='Ridge')
axes[0].axvline(1, color='green', linestyle=':', alpha=0.5, label='Lasso')
axes[0].set_xlabel('L1 Ratio (0=Ridge, 1=Lasso)')
axes[0].set_ylabel('R² Score')
axes[0].set_title(f'ElasticNet Performance (alpha={alpha})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Feature sparsity vs l1_ratio
ax2 = axes[0].twinx()
ax2.plot(l1_ratios, results['n_nonzero'], 'o--', color='orange', 
        label='Non-zero features', alpha=0.7)
ax2.set_ylabel('Number of Non-zero Features', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Coefficient comparison
coef_ridge = ridge.coef_
coef_en = best_en.coef_
coef_lasso = lasso.coef_

x = np.arange(len(coef_ridge))
width = 0.25

axes[1].bar(x - width, coef_ridge, width, label='Ridge', alpha=0.8)
axes[1].bar(x, coef_en, width, label=f'ElasticNet (l1={best_l1_ratio:.2f})', alpha=0.8)
axes[1].bar(x + width, coef_lasso, width, label='Lasso', alpha=0.8)
axes[1].set_xlabel('Feature Index')
axes[1].set_ylabel('Coefficient Value')
axes[1].set_title('Coefficient Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
`,
        'py_list_comp': `# List Comprehensions & Generator Expressions
import time

# Basic list comprehension
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squares = [x**2 for x in numbers]
print("Squares:", squares)

# With condition
evens = [x for x in numbers if x % 2 == 0]
print("Even numbers:", evens)

# Nested list comprehension
matrix = [[i+j for j in range(3)] for i in range(3)]
print("\\nMatrix:")
for row in matrix:
    print(row)

# Flatten matrix
flattened = [item for row in matrix for item in row]
print("\\nFlattened:", flattened)

# Dictionary comprehension
squared_dict = {x: x**2 for x in range(1, 6)}
print("\\nSquared dict:", squared_dict)

# Set comprehension
unique_chars = {char.lower() for char in "Hello World"}
print("Unique chars:", unique_chars)

# Performance comparison
print("\\nPerformance Comparison:")
# Traditional loop
start = time.time()
result1 = []
for i in range(100000):
    result1.append(i**2)
time1 = time.time() - start

# List comprehension
start = time.time()
result2 = [i**2 for i in range(100000)]
time2 = time.time() - start

print(f"Traditional loop: {time1:.4f}s")
print(f"List comprehension: {time2:.4f}s")
print(f"Speedup: {time1/time2:.2f}x faster")

# Generator expression (memory efficient)
gen = (x**2 for x in range(1, 11))
print("\\nGenerator:", type(gen))
print("Generator values:", list(gen))
`,
        'py_decorators': `# Python Decorators
import time
import functools

# Simple decorator
def timer_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.4f} seconds")
        return result
    return wrapper

# Decorator with arguments
def repeat(times):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

# Caching decorator
def memoize(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args in cache:
            print(f"Cache hit for {args}")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

# Using decorators
@timer_decorator
def slow_function(n):
    total = sum(range(n))
    return total

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test decorators
print("Testing timer decorator:")
result = slow_function(1000000)
print(f"Result: {result}")

print("\\nTesting repeat decorator:")
greet("Alice")

print("\\nTesting memoize decorator:")
print("First call:")
print(f"fib(10) = {fibonacci(10)}")
print("\\nSecond call (cached):")
print(f"fib(10) = {fibonacci(10)}")

# Class decorator
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self):
        print("Database initialized")

db1 = Database()
db2 = Database()
print(f"\\nSame instance? {db1 is db2}")
`,
        'py_generators': `# Python Generators
import sys

# Simple generator
def countdown(n):
    print(f"Starting countdown from {n}")
    while n > 0:
        yield n
        n -= 1

# Using generator
print("Countdown generator:")
for num in countdown(5):
    print(num)

# Generator expression
squares = (x**2 for x in range(10))
print("\\nSquares generator:", type(squares))
print("First 5 squares:", [next(squares) for _ in range(5)])

# Infinite generator
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1

# Use with limit
gen = infinite_sequence()
print("\\nFirst 10 numbers:")
for _ in range(10):
    print(next(gen), end=" ")
print()

# Fibonacci generator
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()
print("\\nFirst 15 Fibonacci numbers:")
print([next(fib) for _ in range(15)])

# Generator pipeline
def read_data():
    for i in range(100):
        yield i

def filter_evens(numbers):
    for n in numbers:
        if n % 2 == 0:
            yield n

def square_numbers(numbers):
    for n in numbers:
        yield n ** 2

# Chain generators
data = read_data()
evens = filter_evens(data)
squares = square_numbers(evens)

print("\\nPipeline result (first 10):")
print([next(squares) for _ in range(10)])

# Memory efficiency comparison
print("\\nMemory comparison:")
list_comp = [x**2 for x in range(10000)]
gen_exp = (x**2 for x in range(10000))

print(f"List size: {sys.getsizeof(list_comp)} bytes")
print(f"Generator size: {sys.getsizeof(gen_exp)} bytes")
print(f"Memory saved: {sys.getsizeof(list_comp) - sys.getsizeof(gen_exp)} bytes")
`,
        'py_context_managers': `# Context Managers
import time
from contextlib import contextmanager

# Basic file context manager (built-in)
print("Writing to file:")
with open('test_file.txt', 'w') as f:
    f.write("Hello, World!\\n")
    f.write("This is a test file.")
print("File written and automatically closed")

# Custom context manager (class-based)
class Timer:
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        print(f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print(f"{self.name} took {self.end - self.start:.4f} seconds")
        return False

# Using class-based context manager
print("\\nUsing Timer context manager:")
with Timer("Operation 1"):
    total = sum(range(1000000))
    print(f"Sum: {total}")

# Function-based context manager
@contextmanager
def temporary_value(obj, attr, value):
    """Temporarily change an attribute value"""
    original = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield obj
    finally:
        setattr(obj, attr, original)

class Config:
    debug = False

config = Config()
print(f"\\nOriginal debug: {config.debug}")

with temporary_value(config, 'debug', True):
    print(f"Inside context: {config.debug}")

print(f"After context: {config.debug}")

# Database connection simulator
@contextmanager
def database_connection(db_name):
    print(f"Connecting to {db_name}...")
    connection = {"connected": True, "db": db_name}
    try:
        yield connection
    finally:
        print(f"Closing connection to {db_name}")
        connection["connected"] = False

print("\\nDatabase connection example:")
with database_connection("MyDatabase") as db:
    print(f"Connected: {db['connected']}")
    print(f"Database: {db['db']}")

# Multiple context managers
@contextmanager
def managed_resource(name):
    print(f"Acquiring {name}")
    try:
        yield name
    finally:
        print(f"Releasing {name}")

print("\\nMultiple context managers:")
with managed_resource("Resource1") as r1, managed_resource("Resource2") as r2:
    print(f"Using {r1} and {r2}")

print("\\nAll resources released!")
`,
        'py_exceptions': `# Exception Handling
import sys

# Basic try-except
print("Basic Exception Handling:")
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exceptions
print("\\nMultiple Exception Types:")
try:
    x = int("not a number")
except ValueError as e:
    print(f"ValueError: {e}")
except TypeError as e:
    print(f"TypeError: {e}")

# Try-except-else-finally
print("\\nComplete exception structure:")
try:
    number = int("42")
except ValueError:
    print("Invalid number")
else:
    print(f"Successfully converted: {number}")
finally:
    print("Cleanup always runs")

# Custom exceptions
class InsufficientFundsError(Exception):
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Insufficient funds: balance={balance}, needed={amount}")

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance
    
    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(self.balance, amount)
        self.balance -= amount
        return self.balance

print("\\nCustom Exception:")
account = BankAccount(100)
try:
    account.withdraw(150)
    except InsufficientFundsError as e:
    print(f"Error: {e}")
    print("Your balance: $", e.balance)
    print("Amount requested: $", e.amount)

# Exception chaining
print("\\nException Chaining:")
try:
    try:
        value = 1 / 0
    except ZeroDivisionError as e:
        raise ValueError("Invalid operation") from e
except ValueError as e:
    print(f"Caught: {e}")
    print(f"Original: {e.__cause__}")

# Context manager with exceptions
class ErrorLogger:
    def __enter__(self):
        print("Entering error logging context")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Exception caught: {exc_type.__name__}: {exc_val}")
            return True  # Suppress exception
        return False

print("\\nContext Manager Exception Handling:")
with ErrorLogger():
    print("About to raise an error...")
    raise ValueError("Test error")

print("Execution continued!")

# Assert statements
print("\\nAssert statements:")
def calculate_average(numbers):
    assert len(numbers) > 0, "List cannot be empty"
    assert all(isinstance(n, (int, float)) for n in numbers), "All elements must be numbers"
    return sum(numbers) / len(numbers)

try:
    avg = calculate_average([1, 2, 3, 4, 5])
    print(f"Average: {avg}")
    
    avg = calculate_average([])
except AssertionError as e:
    print(f"Assertion failed: {e}")
`,
        'py_regex': `# Regular Expressions
import re

text = """
Contact us at: support@example.com or sales@company.org
Phone: (123) 456-7890 or 987-654-3210
Website: https://www.example.com
Dates: 2024-01-15, 03/20/2024
"""

print("Sample Text:")
print(text)

# Find all emails
print("\\n1. Email Addresses:")
emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}', text)
print(emails)

# Find phone numbers
print("\\n2. Phone Numbers:")
phones = re.findall(r'\\(?\\d{3}\\)?[\\s-]?\\d{3}[\\s-]?\\d{4}', text)
print(phones)

# Find URLs
print("\\n3. URLs:")
urls = re.findall(r'https?://[^\\s]+', text)
print(urls)

# Find dates (multiple formats)
print("\\n4. Dates:")
dates = re.findall(r'\\d{4}-\\d{2}-\\d{2}|\\d{2}/\\d{2}/\\d{4}', text)
print(dates)

# Pattern matching with groups
print("\\n5. Pattern Groups:")
email_pattern = r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+)\\.([a-zA-Z]{2,})'
for match in re.finditer(email_pattern, text):
    username, domain, tld = match.groups()
    print(f"Username: {username}, Domain: {domain}, TLD: {tld}")

# Search and match
print("\\n6. Search vs Match:")
pattern = r'\\d+'
print(f"Search (finds anywhere): {re.search(pattern, 'abc123def')}")
print(f"Match (from start): {re.match(pattern, 'abc123def')}")
print(f"Match (from start): {re.match(pattern, '123abc')}")

# Substitution
print("\\n7. Substitution:")
censored = re.sub(r'\\d{3}-\\d{3}-\\d{4}', 'XXX-XXX-XXXX', text)
print("Censored phones:", censored[:200])

# Split
print("\\n8. Split by pattern:")
data = "apple,banana;orange:grape|mango"
fruits = re.split(r'[,;:|]', data)
print(fruits)

# Validation examples
print("\\n9. Validation:")
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_password(password):
    # At least 8 chars, 1 uppercase, 1 lowercase, 1 digit
    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d).{8,}$'
    return bool(re.match(pattern, password))

test_email = "user@example.com"
test_password = "SecurePass123"

print(f"Email '{test_email}' valid: {validate_email(test_email)}")
print(f"Password valid: {validate_password(test_password)}")

# Lookahead and lookbehind
print("\\n10. Advanced patterns:")
text2 = "Price: $100, $200, $300"
prices = re.findall(r'(?<=\\$)\\d+', text2)
print(f"Prices without $: {prices}")
`,
        'py_json_api': `# JSON and API Simulation
import json
from datetime import datetime

# Create sample data
data = {
    "users": [
        {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 35}
    ],
    "timestamp": datetime.now().isoformat(),
    "status": "success"
}

# Convert to JSON
print("1. Python to JSON:")
json_string = json.dumps(data, indent=2)
print(json_string[:200] + "...")

# Parse JSON
print("\\n2. JSON to Python:")
parsed_data = json.loads(json_string)
print(f"Type: {type(parsed_data)}")
print(f"Number of users: {len(parsed_data['users'])}")

# Access nested data
print("\\n3. Accessing nested data:")
for user in parsed_data['users']:
    print(f"- {user['name']} ({user['age']} years old)")

# Save to file
filename = 'users_data.json'
with open(filename, 'w') as f:
    json.dump(data, f, indent=2)
print(f"\\n4. Saved to {filename}")

# Load from file
with open(filename, 'r') as f:
    loaded_data = json.load(f)
print(f"5. Loaded from {filename}")
print(f"First user: {loaded_data['users'][0]}")

# Custom JSON encoder for complex objects
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

complex_data = {
    "date": datetime.now(),
    "tags": {"python", "json", "api"},
    "count": 42
}

print("\\n6. Custom encoder:")
print(json.dumps(complex_data, cls=CustomEncoder, indent=2))

# Simulated API response
def simulate_api_request(endpoint, method="GET", data=None):
    """Simulate an API request"""
    responses = {
        "/api/users": {
            "users": parsed_data['users'],
            "count": len(parsed_data['users'])
        },
        "/api/user/1": parsed_data['users'][0],
        "/api/status": {"status": "online", "version": "1.0"}
    }
    
    return {
        "status_code": 200,
        "data": responses.get(endpoint, {"error": "Not found"}),
        "headers": {"Content-Type": "application/json"}
    }

print("\\n7. Simulated API requests:")
# GET all users
response = simulate_api_request("/api/users")
print(f"GET /api/users: {response['status_code']}")
print(f"Users count: {response['data']['count']}")

# GET specific user
response = simulate_api_request("/api/user/1")
print(f"\\nGET /api/user/1:")
print(json.dumps(response['data'], indent=2))

# Pretty printing
print("\\n8. Pretty printing JSON:")
nested = {
    "level1": {
        "level2": {
            "level3": {
                "data": [1, 2, 3, 4, 5]
            }
        }
    }
}
print(json.dumps(nested, indent=4, sort_keys=True))
`,
        'py_threading': `# Multi-threading in Python
import threading
import time
import queue

# Simple thread example
def worker(name, duration):
    print(f"Thread {name} starting")
    time.sleep(duration)
    print(f"Thread {name} finished after {duration}s")

print("1. Basic Threading:")
threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(f"Worker-{i}", i+1))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("All threads completed!\\n")

# Thread with return value using Queue
def calculate_sum(numbers, result_queue):
    thread_name = threading.current_thread().name
    result = sum(numbers)
    print(f"{thread_name} calculated sum: {result}")
    result_queue.put(result)

print("2. Threads with return values:")
result_queue = queue.Queue()
chunks = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

threads = []
for i, chunk in enumerate(chunks):
    t = threading.Thread(target=calculate_sum, args=(chunk, result_queue), 
                        name=f"SumThread-{i}")
    threads.append(t)
    t.start()

for t in threads:
    t.join()

total = sum([result_queue.get() for _ in range(len(chunks))])
print(f"Total sum: {total}\\n")

# Thread synchronization with Lock
counter = 0
counter_lock = threading.Lock()

def increment_counter(times):
    global counter
    for _ in range(times):
        with counter_lock:
            counter += 1

print("3. Thread Synchronization (Lock):")
threads = []
for i in range(5):
    t = threading.Thread(target=increment_counter, args=(1000,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Final counter value: {counter}")
print(f"Expected: {5 * 1000}\\n")

# Producer-Consumer pattern
def producer(q, items):
    for item in items:
        print(f"Producing: {item}")
        q.put(item)
        time.sleep(0.1)
    q.put(None)  # Signal completion

def consumer(q, name):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"{name} consuming: {item}")
        time.sleep(0.15)
        q.task_done()

print("4. Producer-Consumer Pattern:")
work_queue = queue.Queue()
items = ['item1', 'item2', 'item3', 'item4', 'item5']

# Start producer and consumers
producer_thread = threading.Thread(target=producer, args=(work_queue, items))
consumer_threads = [
    threading.Thread(target=consumer, args=(work_queue, f"Consumer-{i}"))
    for i in range(2)
]

producer_thread.start()
for t in consumer_threads:
    t.start()

producer_thread.join()
work_queue.join()
for t in consumer_threads:
    work_queue.put(None)
for t in consumer_threads:
    t.join()

print("Producer-Consumer completed!\\n")

# Thread pool simulation
print("5. Thread Information:")
print(f"Active threads: {threading.active_count()}")
print(f"Current thread: {threading.current_thread().name}")
print(f"Main thread: {threading.main_thread().name}")
`,
        'py_multiprocessing': `# Multi-processing in Python
import multiprocessing as mp
import time
import os

# Simple process
def worker_process(name, duration):
    pid = os.getpid()
    print(f"Process {name} (PID: {pid}) starting")
    time.sleep(duration)
    result = duration ** 2
    print(f"Process {name} (PID: {pid}) finished")
    return result

print("1. Basic Multiprocessing:")
print(f"Main process PID: {os.getpid()}")
print(f"CPU count: {mp.cpu_count()}")

# Create and start processes
processes = []
for i in range(3):
    p = mp.Process(target=worker_process, args=(f"Worker-{i}", i+1))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

print("All processes completed!\\n")

# Process with return value using Queue
def calculate_factorial(n, result_queue):
    result = 1
    for i in range(1, n + 1):
        result *= i
    result_queue.put((n, result))
    print(f"Factorial of {n} = {result}")

print("2. Processes with return values:")
result_queue = mp.Queue()
numbers = [5, 6, 7, 8]

processes = []
for num in numbers:
    p = mp.Process(target=calculate_factorial, args=(num, result_queue))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

results = {}
for _ in range(len(numbers)):
    n, fact = result_queue.get()
    results[n] = fact

print(f"Results: {results}\\n")

# Pool of workers
def square(x):
    return x ** 2

print("3. Process Pool:")
with mp.Pool(processes=4) as pool:
    numbers = list(range(1, 11))
    results = pool.map(square, numbers)
    print(f"Squares: {results}")

# Parallel computation
def compute_sum(start, end):
    return sum(range(start, end))

print("\\n4. Parallel computation:")
ranges = [(0, 250000), (250000, 500000), (500000, 750000), (750000, 1000000)]

start_time = time.time()
with mp.Pool(processes=4) as pool:
    results = pool.starmap(compute_sum, ranges)
    total = sum(results)
parallel_time = time.time() - start_time

print(f"Parallel result: {total}")
print(f"Time taken: {parallel_time:.4f}s")

# Sequential comparison
start_time = time.time()
sequential_total = sum(range(1000000))
sequential_time = time.time() - start_time

print(f"Sequential result: {sequential_total}")
print(f"Time taken: {sequential_time:.4f}s")
print(f"Speedup: {sequential_time/parallel_time:.2f}x\\n")

# Shared memory (Value and Array)
def increment_shared(shared_value, shared_array, lock):
    with lock:
        shared_value.value += 1
        for i in range(len(shared_array)):
            shared_array[i] += 1

print("5. Shared Memory:")
shared_val = mp.Value('i', 0)
shared_arr = mp.Array('i', [0, 0, 0, 0, 0])
lock = mp.Lock()

processes = []
for _ in range(5):
    p = mp.Process(target=increment_shared, args=(shared_val, shared_arr, lock))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

print(f"Shared value: {shared_val.value}")
print(f"Shared array: {list(shared_arr)}")
`,
        'py_async': `# Async/Await Programming
import asyncio
import time

# Basic async function
async def greet(name, delay):
    print(f"Hello {name}, waiting {delay}s...")
    await asyncio.sleep(delay)
    print(f"Goodbye {name}!")
    return f"Completed: {name}"

# Run single async function
print("1. Basic Async/Await:")
async def main1():
    result = await greet("Alice", 1)
    print(result)

asyncio.run(main1())

# Multiple concurrent tasks
print("\\n2. Concurrent Execution:")
async def main2():
    start = time.time()
    
    # Run concurrently
    results = await asyncio.gather(
        greet("Bob", 1),
        greet("Charlie", 2),
        greet("David", 1.5)
    )
    
    elapsed = time.time() - start
    print(f"All completed in {elapsed:.2f}s")
    print(f"Results: {results}")

asyncio.run(main2())

# Async with task creation
print("\\n3. Creating Tasks:")
async def fetch_data(id, delay):
    print(f"Fetching data {id}...")
    await asyncio.sleep(delay)
    return {"id": id, "data": f"Data-{id}"}

async def main3():
    # Create tasks
    tasks = [
        asyncio.create_task(fetch_data(i, i * 0.5))
        for i in range(1, 4)
    ]
    
    # Wait for all tasks
    results = await asyncio.gather(*tasks)
    print("Fetched data:")
    for result in results:
        print(f"  {result}")

asyncio.run(main3())

# Async generator
print("\\n4. Async Generator:")
async def async_range(count):
    for i in range(count):
        await asyncio.sleep(0.1)
        yield i

async def main4():
    print("Async numbers:")
    async for number in async_range(5):
        print(f"  {number}")

asyncio.run(main4())

# Timeout handling
print("\\n5. Timeout Handling:")
async def slow_operation():
    await asyncio.sleep(3)
    return "Completed"

async def main5():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=1.0)
        print(result)
    except asyncio.TimeoutError:
        print("Operation timed out!")

asyncio.run(main5())

# Producer-Consumer async pattern
print("\\n6. Async Producer-Consumer:")
async def producer(queue, items):
    for item in items:
        print(f"Producing: {item}")
        await queue.put(item)
        await asyncio.sleep(0.1)
    await queue.put(None)

async def consumer(queue, name):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        print(f"{name} consuming: {item}")
        await asyncio.sleep(0.15)
        queue.task_done()

async def main6():
    queue = asyncio.Queue()
    items = ['item1', 'item2', 'item3', 'item4']
    
    # Create producer and consumer tasks
    producer_task = asyncio.create_task(producer(queue, items))
    consumer_tasks = [
        asyncio.create_task(consumer(queue, f"Consumer-{i}"))
        for i in range(2)
    ]
    
    await producer_task
    await queue.join()
    
    for task in consumer_tasks:
        await queue.put(None)
    
    await asyncio.gather(*consumer_tasks)
    print("Producer-Consumer completed!")

asyncio.run(main6())

# Performance comparison
print("\\n7. Performance Comparison:")
async def io_task(n):
    await asyncio.sleep(0.1)
    return n ** 2

async def async_version():
    start = time.time()
    results = await asyncio.gather(*[io_task(i) for i in range(10)])
    return time.time() - start, results

def sync_version():
    start = time.time()
    results = []
    for i in range(10):
        time.sleep(0.1)
        results.append(i ** 2)
    return time.time() - start, results

async_time, async_results = asyncio.run(async_version())
sync_time, sync_results = sync_version()

print(f"Async time: {async_time:.2f}s")
print(f"Sync time: {sync_time:.2f}s")
print(f"Speedup: {sync_time/async_time:.2f}x")
`,
        'py_type_hints': `# Type Hints and Annotations
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from typing import TypeVar, Generic

# Basic type hints
def greet(name: str, age: int) -> str:
    return f"Hello {name}, you are {age} years old"

result = greet("Alice", 30)
print(result)

# List and Dict types
def process_numbers(numbers: List[int]) -> Dict[str, float]:
    return {
        "sum": sum(numbers),
        "average": sum(numbers) / len(numbers),
        "max": max(numbers),
        "min": min(numbers)
    }

stats = process_numbers([1, 2, 3, 4, 5])
print(f"\\nStatistics: {stats}")

# Optional and Union types
def find_user(user_id: int) -> Optional[Dict[str, Any]]:
    users = {
        1: {"name": "Alice", "email": "alice@example.com"},
        2: {"name": "Bob", "email": "bob@example.com"}
    }
    return users.get(user_id)

user = find_user(1)
print(f"\\nUser: {user}")

def process_data(data: Union[int, str, List[int]]) -> str:
    if isinstance(data, int):
        return f"Integer: {data}"
    elif isinstance(data, str):
        return f"String: {data}"
    else:
        return f"List: {data}"

print("\\nUnion types:")
print(process_data(42))
print(process_data("hello"))
print(process_data([1, 2, 3]))

# Function types
def apply_operation(x: int, y: int, operation: Callable[[int, int], int]) -> int:
    return operation(x, y)

def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

print(f"\\nApply add: {apply_operation(5, 3, add)}")
print(f"Apply multiply: {apply_operation(5, 3, multiply)}")

# Tuple types
def get_coordinates() -> Tuple[float, float, float]:
    return (10.5, 20.3, 30.7)

x, y, z = get_coordinates()
print(f"\\nCoordinates: x={x}, y={y}, z={z}")

# Generic types
T = TypeVar('T')

def get_first_element(items: List[T]) -> Optional[T]:
    return items[0] if items else None

print(f"\\nFirst element: {get_first_element([1, 2, 3])}")
print(f"First element: {get_first_element(['a', 'b', 'c'])}")

# Generic class
class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> Optional[T]:
        return self._items.pop() if self._items else None
    
    def peek(self) -> Optional[T]:
        return self._items[-1] if self._items else None
    
    def size(self) -> int:
        return len(self._items)

print("\\nGeneric Stack:")
int_stack: Stack[int] = Stack()
int_stack.push(1)
int_stack.push(2)
int_stack.push(3)
print(f"Stack size: {int_stack.size()}")
print(f"Peek: {int_stack.peek()}")
print(f"Pop: {int_stack.pop()}")

# Class with type hints
class Person:
    def __init__(self, name: str, age: int, skills: List[str]) -> None:
        self.name: str = name
        self.age: int = age
        self.skills: List[str] = skills
    
    def add_skill(self, skill: str) -> None:
        self.skills.append(skill)
    
    def get_info(self) -> Dict[str, Union[str, int, List[str]]]:
        return {
            "name": self.name,
            "age": self.age,
            "skills": self.skills
        }

person = Person("Alice", 30, ["Python", "JavaScript"])
person.add_skill("Machine Learning")
print(f"\\nPerson info: {person.get_info()}")

print("\\nNote: Type hints are checked by tools like mypy, not at runtime")
`,
        'py_property': `# Property Decorators
import math

# Basic property
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        """Get the radius"""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Set the radius with validation"""
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        """Calculate area (read-only)"""
        return math.pi * self._radius ** 2
    
    @property
    def circumference(self):
        """Calculate circumference (read-only)"""
        return 2 * math.pi * self._radius

print("1. Basic Property:")
circle = Circle(5)
print(f"Radius: {circle.radius}")
print(f"Area: {circle.area:.2f}")
print(f"Circumference: {circle.circumference:.2f}")

# Update radius
circle.radius = 10
print(f"\\nNew radius: {circle.radius}")
print(f"New area: {circle.area:.2f}")

# Validation
try:
    circle.radius = -5
except ValueError as e:
    print(f"Error: {e}")

# Temperature converter
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9
    
    @property
    def kelvin(self):
        return self._celsius + 273.15
    
    @kelvin.setter
    def kelvin(self, value):
        self.celsius = value - 273.15

print("\\n2. Temperature Converter:")
temp = Temperature(25)
print(f"Celsius: {temp.celsius}°C")
print(f"Fahrenheit: {temp.fahrenheit}°F")
print(f"Kelvin: {temp.kelvin}K")

temp.fahrenheit = 98.6
print(f"\\nSet to 98.6°F:")
print(f"Celsius: {temp.celsius:.2f}°C")
print(f"Kelvin: {temp.kelvin:.2f}K")

# Cached property
class DataProcessor:
    def __init__(self, data):
        self._data = data
        self._sum_cache = None
        self._avg_cache = None
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value
        # Invalidate cache
        self._sum_cache = None
        self._avg_cache = None
    
    @property
    def sum(self):
        if self._sum_cache is None:
            print("Computing sum...")
            self._sum_cache = sum(self._data)
        return self._sum_cache
    
    @property
    def average(self):
        if self._avg_cache is None:
            print("Computing average...")
            self._avg_cache = self.sum / len(self._data)
        return self._avg_cache

print("\\n3. Cached Properties:")
processor = DataProcessor([1, 2, 3, 4, 5])
print(f"Sum: {processor.sum}")  # Computes
print(f"Sum again: {processor.sum}")  # Uses cache
print(f"Average: {processor.average}")  # Computes
print(f"Average again: {processor.average}")  # Uses cache

# Bank account with balance property
class BankAccount:
    def __init__(self, initial_balance=0):
        self._balance = initial_balance
        self._transactions = []
    
    @property
    def balance(self):
        return self._balance
    
    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            self._transactions.append("Deposit: +$" + str(amount))
    
    def withdraw(self, amount):
        if amount > 0 and amount <= self._balance:
            self._balance -= amount
            self._transactions.append("Withdrawal: -$" + str(amount))
        else:
            raise ValueError("Insufficient funds or invalid amount")
    
    @property
    def transaction_history(self):
        return self._transactions.copy()

print("\\n4. Bank Account:")
account = BankAccount(1000)
print("Initial balance: $", account.balance)

account.deposit(500)
account.withdraw(200)
print("Current balance: $", account.balance)
print("Transactions:")
for transaction in account.transaction_history:
    print(f"  {transaction}")
`,
        'py_magic_methods': `# Magic Methods (Dunder Methods)
import math

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __str__(self):
        return f"<{self.x}, {self.y}>"
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __abs__(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def __len__(self):
        return 2
    
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Index out of range")

print("1. Vector Operations:")
v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"v1 + v2 = {v1 + v2}")
print(f"v1 - v2 = {v1 - v2}")
print(f"v1 * 3 = {v1 * 3}")
print(f"|v1| = {abs(v1):.2f}")
print(f"v1[0] = {v1[0]}, v1[1] = {v1[1]}")
print(f"v1 == v2: {v1 == v2}")

# Context manager magic methods
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing {self.filename}")
        if self.file:
            self.file.close()
        return False

print("\\n2. Context Manager:")
with FileManager('test.txt', 'w') as f:
    f.write("Hello, Magic Methods!")

# Callable class
class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, x):
        return x * self.factor

print("\\n3. Callable Object:")
times_three = Multiplier(3)
print(f"Multiplier(3)(5) = {times_three(5)}")
print(f"Multiplier(3)(10) = {times_three(10)}")

# Container magic methods
class CustomList:
    def __init__(self, items=None):
        self.items = items or []
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
    
    def __setitem__(self, index, value):
        self.items[index] = value
    
    def __delitem__(self, index):
        del self.items[index]
    
    def __contains__(self, item):
        return item in self.items
    
    def __iter__(self):
        return iter(self.items)
    
    def __reversed__(self):
        return reversed(self.items)
    
    def __repr__(self):
        return f"CustomList({self.items})"

print("\\n4. Container Operations:")
cl = CustomList([1, 2, 3, 4, 5])
print(f"List: {cl}")
print(f"Length: {len(cl)}")
print(f"cl[2] = {cl[2]}")
print(f"3 in cl: {3 in cl}")
print(f"10 in cl: {10 in cl}")
print(f"Iteration: {[x * 2 for x in cl]}")
print(f"Reversed: {list(reversed(cl))}")

# Comparison magic methods
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __lt__(self, other):
        return self.age < other.age
    
    def __le__(self, other):
        return self.age <= other.age
    
    def __gt__(self, other):
        return self.age > other.age
    
    def __ge__(self, other):
        return self.age >= other.age
    
    def __eq__(self, other):
        return self.age == other.age
    
    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

print("\\n5. Comparison Operations:")
alice = Person("Alice", 30)
bob = Person("Bob", 25)
charlie = Person("Charlie", 30)

print(f"{alice} > {bob}: {alice > bob}")
print(f"{alice} == {charlie}: {alice == charlie}")
print(f"Sorted: {sorted([alice, bob, charlie])}")

# String representation
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
    
    def __str__(self):
        return f"'{self.title}' by {self.author}"
    
    def __repr__(self):
        return f"Book('{self.title}', '{self.author}', {self.pages})"
    
    def __format__(self, format_spec):
        if format_spec == 'short':
            return self.title
        elif format_spec == 'full':
            return f"'{self.title}' by {self.author} ({self.pages} pages)"
        return str(self)

print("\\n6. String Representation:")
book = Book("Python Guide", "John Doe", 350)
print(f"str(book): {str(book)}")
print(f"repr(book): {repr(book)}")
print(f"Short format: {book:short}")
print(f"Full format: {book:full}")
`,
        // Additional algorithm examples
        'algo_binary_search': `# Binary Search Algorithm
def binary_search(arr, target):
    """Binary search on sorted array"""
    left, right = 0, len(arr) - 1
    iterations = 0
    
    while left <= right:
        iterations += 1
        mid = (left + right) // 2
        print(f"Iteration {iterations}: Checking index {mid} (value: {arr[mid]})")
        
        if arr[mid] == target:
            return mid, iterations
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1, iterations

# Test binary search
print("Binary Search:")
sorted_array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
print(f"Array: {sorted_array}")

target = 13
index, iterations = binary_search(sorted_array, target)
print(f"\\nSearching for {target}:")
print(f"Found at index: {index}, Iterations: {iterations}")
`,
        'algo_sorting': `# Sorting Algorithms
import random, time

arr = random.sample(range(100), 20)
print(f"Original: {arr}")

# Bubble Sort
def bubble_sort(a):
    a = a.copy()
    for i in range(len(a)):
        for j in range(len(a)-i-1):
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
    return a

print(f"Bubble sorted: {bubble_sort(arr)}")

# Quick Sort
def quick_sort(a):
    if len(a) <= 1: return a
    pivot = a[len(a)//2]
    return quick_sort([x for x in a if x < pivot]) + [x for x in a if x == pivot] + quick_sort([x for x in a if x > pivot])

print(f"Quick sorted: {quick_sort(arr)}")
print(f"Python sorted: {sorted(arr)}")
`,
        'algo_linked_list': `# Linked List
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        if not self.head:
            self.head = Node(data)
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = Node(data)
    
    def display(self):
        items = []
        curr = self.head
        while curr:
            items.append(str(curr.data))
            curr = curr.next
        return ' -> '.join(items)

ll = LinkedList()
for i in [10, 20, 30, 40, 50]:
    ll.append(i)
print(f"Linked List: {ll.display()}")
`,
        'algo_stack_queue': `# Stack and Queue
class Stack:
    def __init__(self):
        self.items = []
    def push(self, x):
        self.items.append(x)
    def pop(self):
        return self.items.pop() if self.items else None
    def peek(self):
        return self.items[-1] if self.items else None

class Queue:
    def __init__(self):
        self.items = []
    def enqueue(self, x):
        self.items.append(x)
    def dequeue(self):
        return self.items.pop(0) if self.items else None

print("Stack (LIFO):")
s = Stack()
for i in [1,2,3,4,5]:
    s.push(i)
print(f"Push 1-5: {s.items}")
print(f"Pop: {s.pop()}, {s.pop()}")
print(f"Remaining: {s.items}")

print("\\nQueue (FIFO):")
q = Queue()
for i in [1,2,3,4,5]:
    q.enqueue(i)
print(f"Enqueue 1-5: {q.items}")
print(f"Dequeue: {q.dequeue()}, {q.dequeue()}")
print(f"Remaining: {q.items}")
`,
        'algo_trees': `# Binary Tree
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        if not self.root:
            self.root = Node(val)
        else:
            self._insert(self.root, val)
    
    def _insert(self, node, val):
        if val < node.val:
            if node.left:
                self._insert(node.left, val)
            else:
                node.left = Node(val)
        else:
            if node.right:
                self._insert(node.right, val)
            else:
                node.right = Node(val)
    
    def inorder(self, node, result=[]):
        if node:
            self.inorder(node.left, result)
            result.append(node.val)
            self.inorder(node.right, result)
        return result

bt = BinaryTree()
for val in [50,30,70,20,40,60,80]:
    bt.insert(val)
print(f"Tree (inorder): {bt.inorder(bt.root, [])}")
`,
        'algo_graphs': `# Graph (Adjacency List)
class Graph:
    def __init__(self):
        self.graph = {}
    
    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)
    
    def bfs(self, start):
        visited, queue = set(), [start]
        result = []
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                queue.extend([v for v in self.graph.get(vertex, []) if v not in visited])
        return result
    
    def dfs(self, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        result = [start]
        for neighbor in self.graph.get(start, []):
            if neighbor not in visited:
                result.extend(self.dfs(neighbor, visited))
        return result

g = Graph()
edges = [(0,1), (0,2), (1,3), (2,3), (3,4)]
for u,v in edges:
    g.add_edge(u, v)

print(f"Graph: {g.graph}")
print(f"BFS from 0: {g.bfs(0)}")
print(f"DFS from 0: {g.dfs(0)}")
`,
        'algo_dp': `# Dynamic Programming - Fibonacci
def fib_recursive(n):
    if n <= 1: return n
    return fib_recursive(n-1) + fib_recursive(n-2)

def fib_dp(n):
    if n <= 1: return n
    dp = [0] * (n+1)
    dp[1] = 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

def fib_memo(n, memo={}):
    if n in memo: return memo[n]
    if n <= 1: return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

n = 20
print(f"Fibonacci({n}):")
print(f"Recursive: {fib_recursive(n)}")
print(f"DP (tabulation): {fib_dp(n)}")
print(f"DP (memoization): {fib_memo(n)}")

# Coin Change
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i-coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

print(f"\\nCoin Change (coins=[1,5,10], amount=30): {coin_change([1,5,10], 30)} coins")
`,
        'algo_hash': `# Hash Table
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def put(self, key, val):
        idx = self._hash(key)
        for i, (k, v) in enumerate(self.table[idx]):
            if k == key:
                self.table[idx][i] = (key, val)
                return
        self.table[idx].append((key, val))
    
    def get(self, key):
        idx = self._hash(key)
        for k, v in self.table[idx]:
            if k == key:
                return v
        return None

ht = HashTable(5)
ht.put("Alice", 25)
ht.put("Bob", 30)
ht.put("Charlie", 35)

print("Hash Table:")
print(f"Alice: {ht.get('Alice')}")
print(f"Bob: {ht.get('Bob')}")
print(f"Charlie: {ht.get('Charlie')}")

# Two Sum using hash
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

nums = [2, 7, 11, 15]
target = 9
print(f"\\nTwo Sum({nums}, {target}): {two_sum(nums, target)}")
`,
        'ts_analysis': `# Time Series Analysis
import numpy as np
import matplotlib.pyplot as plt

# Generate time series data
np.random.seed(42)
time = np.arange(100)
trend = 0.5 * time
seasonal = 10 * np.sin(2 * np.pi * time / 12)
noise = np.random.normal(0, 3, 100)
ts = trend + seasonal + noise

# Moving average
window = 5
ma = np.convolve(ts, np.ones(window)/window, mode='valid')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(time, ts, label='Original', alpha=0.7)
plt.plot(time[window-1:], ma, label=f'MA({window})', linewidth=2)
plt.title('Time Series with Moving Average')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(ts, bins=20, edgecolor='black')
plt.title('Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

print(f"Time series stats:")
print(f"Mean: {np.mean(ts):.2f}")
print(f"Std: {np.std(ts):.2f}")
print(f"Min: {np.min(ts):.2f}")
print(f"Max: {np.max(ts):.2f}")
`,
        'ts_moving_avg': `# Moving Averages
import numpy as np
import matplotlib.pyplot as plt

# Stock price simulation
np.random.seed(42)
days = 200
price = 100 + np.cumsum(np.random.randn(days) * 2)

# Calculate moving averages
def moving_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

ma_5 = moving_average(price, 5)
ma_20 = moving_average(price, 20)
ma_50 = moving_average(price, 50)

plt.figure(figsize=(12, 6))
plt.plot(price, label='Price', alpha=0.6)
plt.plot(range(4, len(price)), ma_5, label='MA(5)')
plt.plot(range(19, len(price)), ma_20, label='MA(20)')
plt.plot(range(49, len(price)), ma_50, label='MA(50)')
plt.title('Stock Price with Moving Averages')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Price stats:")
print(f"Start: {price[0]:.2f}")
print(f"End: {price[-1]:.2f}")
print(f"Change: {((price[-1]/price[0]-1)*100):.2f}%")
`,
        'finance_stocks': `# Stock Analysis
import numpy as np
import matplotlib.pyplot as plt

# Simulate stock returns
np.random.seed(42)
days = 252
returns = np.random.normal(0.001, 0.02, days)
price = 100 * np.exp(np.cumsum(returns))

# Calculate metrics
daily_return = np.diff(price) / price[:-1]
cumulative_return = (price[-1] / price[0] - 1) * 100
volatility = np.std(daily_return) * np.sqrt(252) * 100
sharpe_ratio = (np.mean(daily_return) * 252) / (np.std(daily_return) * np.sqrt(252))

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(price)
plt.title('Stock Price')
plt.ylabel('Price')

plt.subplot(2, 2, 2)
plt.plot(daily_return)
plt.title('Daily Returns')
plt.ylabel('Return')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)

plt.subplot(2, 2, 3)
plt.hist(daily_return, bins=30, edgecolor='black')
plt.title('Return Distribution')
plt.xlabel('Daily Return')

plt.subplot(2, 2, 4)
cumulative = (1 + daily_return).cumprod() - 1
plt.plot(cumulative * 100)
plt.title('Cumulative Returns')
plt.ylabel('Return (%)')

plt.tight_layout()
plt.show()

print(f"Stock Metrics:")
print(f"Cumulative Return: {cumulative_return:.2f}%")
print(f"Annual Volatility: {volatility:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
`,
        'finance_portfolio': `# Portfolio Optimization
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Portfolio with 3 stocks
n_stocks = 3
n_days = 252
returns = np.random.normal([0.001, 0.0008, 0.0012], [0.02, 0.015, 0.025], (n_days, n_stocks))

# Simulate different portfolios
n_portfolios = 1000
results = np.zeros((3, n_portfolios))

for i in range(n_portfolios):
    weights = np.random.random(n_stocks)
    weights /= np.sum(weights)
    
    portfolio_return = np.sum(returns.mean(axis=0) * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.T @ returns / n_days, weights))) * np.sqrt(252)
    sharpe = portfolio_return / portfolio_std
    
    results[0,i] = portfolio_return
    results[1,i] = portfolio_std
    results[2,i] = sharpe

plt.figure(figsize=(10, 6))
scatter = plt.scatter(results[1,:]*100, results[0,:]*100, c=results[2,:], cmap='viridis')
plt.colorbar(scatter, label='Sharpe Ratio')
plt.xlabel('Volatility (%)')
plt.ylabel('Return (%)')
plt.title('Efficient Frontier')
plt.grid(True, alpha=0.3)
plt.show()

best_idx = np.argmax(results[2,:])
print(f"Best Portfolio:")
print(f"Return: {results[0,best_idx]*100:.2f}%")
print(f"Volatility: {results[1,best_idx]*100:.2f}%")
print(f"Sharpe Ratio: {results[2,best_idx]:.2f}")
`,
        'nlp_preprocess': `# Text Preprocessing
import re
from collections import Counter

text = """
Natural Language Processing (NLP) is a field of AI. It helps computers
understand, interpret, and generate human language. NLP applications include
chatbots, translation, and sentiment analysis!
"""

print("Original text:")
print(text)

# Lowercase
text_lower = text.lower()
print(f"\\nLowercase: {text_lower[:100]}...")

# Remove punctuation
text_no_punct = re.sub(r'[^\\w\\s]', '', text_lower)
print(f"\\nNo punctuation: {text_no_punct[:100]}...")

# Tokenization
tokens = text_no_punct.split()
print(f"\\nTokens ({len(tokens)}): {tokens[:10]}")

# Remove stopwords
stopwords = {'is', 'a', 'of', 'it', 'and', 'the'}
filtered = [w for w in tokens if w not in stopwords]
print(f"\\nFiltered ({len(filtered)}): {filtered[:10]}")

# Word frequency
freq = Counter(filtered)
print(f"\\nTop 5 words:")
for word, count in freq.most_common(5):
    print(f"  {word}: {count}")

# N-grams
bigrams = [f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered)-1)]
print(f"\\nBigrams (first 5): {bigrams[:5]}")
`,
        'nlp_word_freq': `# Word Frequency Analysis
from collections import Counter
import matplotlib.pyplot as plt

text = """
Python is a high-level programming language. Python is widely used for web development,
data science, artificial intelligence, and automation. Python has a simple syntax that
makes it easy to learn. Python is one of the most popular programming languages today.
"""

# Tokenize
words = text.lower().split()
words = [w.strip('.,!?') for w in words]

# Count frequencies
word_freq = Counter(words)

print(f"Total words: {len(words)}")
print(f"Unique words: {len(word_freq)}")

print("\\nTop 10 words:")
for word, count in word_freq.most_common(10):
    print(f"  {word}: {count}")

# Visualize
top_words = dict(word_freq.most_common(10))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.barh(list(top_words.keys()), list(top_words.values()))
plt.xlabel('Frequency')
plt.title('Top 10 Words')

plt.subplot(1, 2, 2)
all_freqs = sorted(word_freq.values(), reverse=True)
plt.plot(all_freqs)
plt.xlabel('Word Rank')
plt.ylabel('Frequency')
plt.title('Word Frequency Distribution (Zipf\\'s Law)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
`,
        'nlp_sentiment': `# Sentiment Analysis (Simple)
from collections import Counter

# Simple sentiment lexicon
positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best', 'happy', 'perfect'}
negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor', 'disappointing', 'sad', 'angry'}

def analyze_sentiment(text):
    words = text.lower().split()
    words = [w.strip('.,!?') for w in words]
    
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    
    score = pos_count - neg_count
    
    if score > 0:
        sentiment = "Positive"
    elif score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return sentiment, score, pos_count, neg_count

reviews = [
    "This product is amazing! I love it. Best purchase ever!",
    "Terrible quality. Very disappointing and bad.",
    "It's okay, nothing special.",
    "Excellent service! Very happy with this.",
    "Worst experience. Hate it. Awful product."
]

print("Sentiment Analysis Results:\\n")
for i, review in enumerate(reviews, 1):
    sentiment, score, pos, neg = analyze_sentiment(review)
    print(f"Review {i}: {review[:50]}...")
    print(f"  Sentiment: {sentiment} (score: {score}, pos: {pos}, neg: {neg})\\n")
`,
        'nlp_tfidf': `# TF-IDF (Term Frequency-Inverse Document Frequency)
import math
from collections import Counter

documents = [
    "Python is a programming language",
    "Machine learning uses Python",
    "Python is popular for data science",
    "Data science and machine learning"
]

def compute_tf(doc):
    words = doc.lower().split()
    word_count = Counter(words)
    total_words = len(words)
    return {word: count/total_words for word, count in word_count.items()}

def compute_idf(documents):
    N = len(documents)
    idf = {}
    all_words = set(word for doc in documents for word in doc.lower().split())
    
    for word in all_words:
        count = sum(1 for doc in documents if word in doc.lower())
        idf[word] = math.log(N / count)
    
    return idf

def compute_tfidf(documents):
    idf = compute_idf(documents)
    tfidf_scores = []
    
    for doc in documents:
        tf = compute_tf(doc)
        tfidf = {word: tf_val * idf[word] for word, tf_val in tf.items()}
        tfidf_scores.append(tfidf)
    
    return tfidf_scores

print("Documents:")
for i, doc in enumerate(documents):
    print(f"{i+1}. {doc}")

tfidf = compute_tfidf(documents)

print("\\nTF-IDF Scores:\\n")
for i, scores in enumerate(tfidf):
    print(f"Document {i+1}:")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_scores[:5]:
        print(f"  {word}: {score:.3f}")
    print()
`,
        'web_scraping': `# Web Scraping Basics (Simulated)
# Note: This is a simulation as actual web scraping requires external libraries

class HTMLParser:
    def __init__(self, html):
        self.html = html
    
    def find_tags(self, tag):
        import re
        pattern = f'<{tag}[^>]*>(.*?)</{tag}>'
        return re.findall(pattern, self.html, re.DOTALL)
    
    def find_links(self):
        import re
        return re.findall(r'href=["\\']([^\"\\']+)["\\'']', self.html)

# Simulated HTML
html = """
<html>
<head><title>Sample Page</title></head>
<body>
    <h1>Welcome</h1>
    <p>This is a paragraph.</p>
    <a href="https://example.com">Link 1</a>
    <p>Another paragraph with <a href="/page2">Link 2</a></p>
    <div>Some content</div>
</body>
</html>
"""

print("HTML Parsing Example:\\n")
parser = HTMLParser(html)

# Extract titles
titles = parser.find_tags('title')
print(f"Titles: {titles}")

# Extract headings
headings = parser.find_tags('h1')
print(f"Headings: {headings}")

# Extract paragraphs
paragraphs = parser.find_tags('p')
print(f"Paragraphs: {paragraphs}")

# Extract links
links = parser.find_links()
print(f"Links: {links}")

# Simulated data extraction
data = {
    'title': titles[0] if titles else None,
    'heading': headings[0] if headings else None,
    'num_paragraphs': len(paragraphs),
    'num_links': len(links),
    'links': links
}

print(f"\\nExtracted Data:")
for key, value in data.items():
    print(f"  {key}: {value}")
`,
        'data_csv': `# CSV Data Processing
import csv
from io import StringIO

# Simulated CSV data
csv_data = """name,age,city,salary
Alice,25,New York,75000
Bob,30,San Francisco,90000
Charlie,35,Boston,85000
David,28,Seattle,80000
Eve,32,Austin,78000"""

print("CSV Data Processing\\n")

# Read CSV
file = StringIO(csv_data)
reader = csv.DictReader(file)
data = list(reader)

print("Data:")
for row in data:
    print(row)

# Calculate statistics
ages = [int(row['age']) for row in data]
salaries = [int(row['salary']) for row in data]

print(f"\\nStatistics:")
print(f"Average age: {sum(ages)/len(ages):.1f}")
print(f"Average salary: {sum(salaries)/len(salaries):,.0f}")

# Filter data
high_earners = [row for row in data if int(row['salary']) >= 80000]
print(f"\\nHigh earners (>=80k): {len(high_earners)}")
for row in high_earners:
    print(f"  {row['name']}: {row['salary']}")

# Group by city
cities = {}
for row in data:
    city = row['city']
    if city not in cities:
        cities[city] = []
    cities[city].append(row['name'])

print(f"\\nBy city:")
for city, people in cities.items():
    print(f"  {city}: {', '.join(people)}")
`,
        'data_excel': `# Excel-like Data Operations
import numpy as np

# Simulated spreadsheet data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Q1': [1000, 1500, 1200, 1800, 1300],
    'Q2': [1100, 1600, 1300, 1900, 1400],
    'Q3': [1200, 1700, 1400, 2000, 1500],
    'Q4': [1300, 1800, 1500, 2100, 1600]
}

print("Quarterly Sales Data\\n")

# Display data
print(f"{'Name':<10} {'Q1':>8} {'Q2':>8} {'Q3':>8} {'Q4':>8} {'Total':>10} {'Avg':>10}")
print("-" * 70)

totals = []
for i, name in enumerate(data['Name']):
    q1, q2, q3, q4 = data['Q1'][i], data['Q2'][i], data['Q3'][i], data['Q4'][i]
    total = q1 + q2 + q3 + q4
    avg = total / 4
    totals.append(total)
    print(f"{name:<10} {q1:>8} {q2:>8} {q3:>8} {q4:>8} {total:>10} {avg:>10.0f}")

# Summary statistics
print("\\n" + "="*70)
print(f"{'TOTALS':<10} {sum(data['Q1']):>8} {sum(data['Q2']):>8} {sum(data['Q3']):>8} {sum(data['Q4']):>8} {sum(totals):>10}")

print(f"\\nTop performer: {data['Name'][np.argmax(totals)]} ({max(totals):,})")
print(f"Best quarter: Q{np.argmax([sum(data[f'Q{i}']) for i in range(1,5)])+1}")
`,
        'auto_email': `# Email Automation (Simulated)
from datetime import datetime

class Email:
    def __init__(self, to, subject, body):
        self.to = to
        self.subject = subject
        self.body = body
        self.sent_at = None
    
    def send(self):
        self.sent_at = datetime.now()
        print(f"[EMAIL SENT]")
        print(f"To: {self.to}")
        print(f"Subject: {self.subject}")
        print(f"Body:\\n{self.body}")
        print(f"Sent at: {self.sent_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)

class EmailCampaign:
    def __init__(self):
        self.emails = []
    
    def add_recipient(self, email, name):
        subject = f"Hello {name}!"
        body = f"""
Dear {name},

Thank you for your interest in our services. We're excited to have you!

Best regards,
The Team
"""
        self.emails.append(Email(email, subject, body))
    
    def send_all(self):
        print(f"Sending {len(self.emails)} emails...\\n")
        for email in self.emails:
            email.send()
        print(f"Campaign complete! {len(self.emails)} emails sent.")

# Create campaign
campaign = EmailCampaign()
recipients = [
    ("alice@example.com", "Alice"),
    ("bob@example.com", "Bob"),
    ("charlie@example.com", "Charlie")
]

for email, name in recipients:
    campaign.add_recipient(email, name)

campaign.send_all()

# Email template with placeholders
template = "Hello {name}, your order #{order_id} totaling $" + "{amount} is confirmed!"
orders = [
    {"name": "Alice", "order_id": 1001, "amount": 99.99},
    {"name": "Bob", "order_id": 1002, "amount": 149.50}
]

print("\\nOrder Confirmations:")
for order in orders:
    message = template.format(**order)
    print(f"  {message}")
`
    };
    
    const code = examples[key] || '# Example not found';
    console.log('=== loadPythonExample ===');
    console.log('Key:', key);
    console.log('Found:', code !== '# Example not found');
    console.log('Code length:', code.length, 'characters');
    setEditorCode(code);
}

async function runPythonCode() {
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('🚀 [RUN BUTTON CLICKED] Executing Python code');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    
    const code = getEditorCode();
    
    console.log('📝 Code length:', code.length, 'characters');
    
    if (!code.trim()) {
        console.warn('⚠️  No code to execute');
        showOutput('Error: No code to execute', 'error');
        return;
    }
    
    // Show loading
    showLoading(true);
    clearOutput();
    clearPlots();
    
    try {
        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
        console.log('📤 [HTTP REQUEST] Sending POST to server');
        console.log('   URL: /coding-practices/python/execute/');
        console.log('   Method: POST');
        console.log('   Content-Type: application/json');
        console.log('   Timestamp:', new Date().toISOString());
        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
        
        const response = await fetch('/coding-practices/python/execute/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ code: code })
        });
        
        console.log('📥 [HTTP RESPONSE] Received from server');
        console.log('   Status:', response.status, response.statusText);
        console.log('   OK:', response.ok);
        
        const result = await response.json();
        
        console.log('📋 Response data:', {
            success: result.success,
            stdout_length: result.stdout?.length || 0,
            stderr_length: result.stderr?.length || 0,
            plots_count: result.plots?.length || 0
        });
        
        if (response.ok) {
            handlePythonExecutionResult(result);
        } else {
            console.error('❌ Server returned error:', result.error);
            showOutput(`Error: ${result.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error('❌ Network/Fetch error:', error);
        showOutput(`Network Error: ${error.message}`, 'error');
        console.error('Execution error:', error);
    } finally {
        showLoading(false);
        console.log('✅ Python execution flow completed');
    }
}

function handlePythonExecutionResult(result) {
    let output = '';
    
    if (result.stdout) {
        output += `<div class="output-section output-stdout">
            <div class="output-label">Output:</div>
            <pre>${escapeHtml(result.stdout)}</pre>
        </div>`;
    }
    
    if (result.stderr) {
        output += `<div class="output-section output-stderr">
            <div class="output-label">Errors:</div>
            <pre>${escapeHtml(result.stderr)}</pre>
        </div>`;
    }
    
    if (!result.stdout && !result.stderr) {
        output += '<div class="output-section"><em>No output</em></div>';
    }
    
    showOutput(output, result.success ? 'success' : 'error');
    
    // Display plots if any
    if (result.plots && result.plots.length > 0) {
        displayPlots(result.plots);
    } else {
        showPlotsPlaceholder();
    }
}

function displayPlots(plots) {
    const plotsDiv = document.getElementById('plots');
    plotsDiv.className = 'plots-content';
    
    let plotsHtml = '';
    plots.forEach((plot, index) => {
        plotsHtml += `
            <div class="plot-container">
                <div class="plot-title">Plot ${index + 1}</div>
                <img src="data:image/png;base64,${plot.data}" alt="Plot ${index + 1}">
            </div>
        `;
    });
    
    plotsDiv.innerHTML = plotsHtml;
}

function showPlotsPlaceholder() {
    const plotsDiv = document.getElementById('plots');
    plotsDiv.className = 'plots-content';
    plotsDiv.innerHTML = `
        <div class="plots-placeholder">
            <i class="fas fa-chart-bar"></i>
            <p>No plots generated</p>
        </div>
    `;
}

function clearPlots() {
    const plotsDiv = document.getElementById('plots');
    plotsDiv.className = 'plots-content';
    plotsDiv.innerHTML = `
        <div class="plots-placeholder">
            <i class="fas fa-chart-bar"></i>
            <p>Plots will appear here when you use matplotlib</p>
        </div>
    `;
}

function showOutput(content, type = 'info') {
    const outputDiv = document.getElementById('output');
    outputDiv.className = 'output-content';
    outputDiv.classList.add(`output-${type}`);
    outputDiv.innerHTML = content;
}

function clearOutput() {
    const outputDiv = document.getElementById('output');
    outputDiv.className = 'output-content';
    outputDiv.innerHTML = `
        <div class="output-placeholder">
            <i class="fas fa-info-circle"></i>
            <p>Output will appear here after running your code</p>
        </div>
    `;
}

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
    
    const runBtn = document.getElementById('runBtn');
    if (runBtn) {
        runBtn.disabled = show;
        if (show) {
            runBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running...';
        } else {
            runBtn.innerHTML = '<i class="fas fa-play"></i> Run Code';
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to run
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        runPythonCode();
    }
    
    // Ctrl/Cmd + K to clear
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        clearEditor();
        clearOutput();
        clearPlots();
    }
});

// Make loadPythonExample globally available
window.loadPythonExample = loadPythonExample;
console.log('✓ loadPythonExample registered globally');

// Test function - call this from browser console to test if it works
window.testLoadExample = function() {
    console.log('🧪 TEST: Calling loadPythonExample with test key');
    loadPythonExample('ml_linear_regression');
};
console.log('✓ Test function available: Type window.testLoadExample() in console to test');
