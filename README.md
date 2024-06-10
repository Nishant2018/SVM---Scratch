## Support Vector Machine (SVM)

### Introduction

Support Vector Machine (SVM) is a supervised machine learning algorithm commonly used for classification and regression tasks. It works by finding the hyperplane that best separates the data into different classes.

### Why Use SVM?

- **Effective in High Dimensions**: SVM is effective in cases where the number of dimensions is greater than the number of samples.
- **Memory Efficient**: Uses a subset of training points (support vectors) in the decision function, making it memory efficient.
- **Versatile**: Can be used for both linear and non-linear data through the kernel trick.

### How SVM Works

1. **Hyperplane and Margin**: The main objective of SVM is to find the hyperplane that best separates the classes. The hyperplane is chosen to maximize the margin, which is the distance between the hyperplane and the nearest data points from either class (support vectors).

2. **Linear SVM**: For a linearly separable dataset, SVM finds the hyperplane:
   \[
   \mathbf{w} \cdot \mathbf{x} + b = 0
   \]
   Where \( \mathbf{w} \) is the weight vector, \( \mathbf{x} \) is the feature vector, and \( b \) is the bias.

   The optimization problem is:
   \[
   \min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2
   \]
   Subject to:
   \[
   y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \quad \forall i
   \]
   Where \( y_i \) is the class label of \( \mathbf{x}_i \).

3. **Non-Linear SVM and Kernel Trick**: For non-linearly separable data, SVM uses kernel functions to map the data into a higher-dimensional space where it becomes linearly separable. Common kernels include:
   - Polynomial Kernel:
     \[
     K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i \cdot \mathbf{x}_j + c)^d
     \]
   - Radial Basis Function (RBF) Kernel:
     \[
     K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)
     \]

### Example Code

Here is a simple example of how to use SVM for classification using Python's `scikit-learn` library:

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
