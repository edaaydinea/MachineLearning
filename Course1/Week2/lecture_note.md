# Multiple Linear Regression

## Multiple features

1. **Introduction to Multiple Linear Regression:**
   - **Original Model:** For a single feature \( x \), the model is \( f_{w,b}(x) = wx + b \).
   - **Extended Model:** With multiple features, the model becomes \( f_{w,b}(x) = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + b \).

2. **Features and Notation:**
   - **Features:** Let \( X_1, X_2, X_3, X_4 \) denote four features.
   - **Vector Notation:**
     - \( X^{(i)} \): Vector of features for the \( i \)-th training example.
     - Example: \( X^{(2)} = [1416, 3, 2, 40] \).
   - **Specific Feature:** \( X^{(i)}_j \) denotes the value of the \( j \)-th feature in the \( i \)-th training example. Example: \( X^{(2)}_3 = 2 \).

3. **Model Representation:**
   - **Example Model:** \( f_{w,b}(x) = 0.1X_1 + 4X_2 + 10X_3 - 2X_4 + 80 \).
   - **Interpretation of Parameters:**
     - \( b = 80 \): Base price of $80,000.
     - \( w_1 = 0.1 \): Price increases by $100 per square foot.
     - \( w_2 = 4 \): Price increases by $4,000 per bedroom.
     - \( w_3 = 10 \): Price increases by $10,000 per floor.
     - \( w_4 = -2 \): Price decreases by $2,000 per year of age.

4. **Vector Notation for Multiple Linear Regression:**
   - **Parameters Vector:** \( \mathbf{W} = [w_1, w_2, w_3, w_4] \).
   - **Feature Vector:** \( \mathbf{X} = [X_1, X_2, X_3, X_4] \).
   - **Model Expression:** \( f(\mathbf{X}) = \mathbf{W} \cdot \mathbf{X} + b \).

5. **Dot Product Calculation:**
   - **Definition:** Dot product of vectors \( \mathbf{W} \) and \( \mathbf{X} \) is \( w_1X_1 + w_2X_2 + w_3X_3 + w_4X_4 \).
   - **Model in Compact Form:** \( f(\mathbf{X}) = \mathbf{W} \cdot \mathbf{X} + b \).

6. **Terminology:**
   - **Multiple Linear Regression:** Refers to linear regression with multiple features.
   - **Univariate Regression:** Refers to linear regression with a single feature.
   - **Multivariate Regression:** A different concept not used in this context.

## Vectorization part 1

1. **Introduction to Vectorization:**
   - **Purpose:** Vectorization speeds up code execution and reduces code length. It leverages numerical linear algebra libraries and hardware acceleration (e.g., GPUs).

2. **Example with Vectorization:**
   - **Parameters and Features:** Consider \( \mathbf{w} \) and \( \mathbf{x} \) vectors with three elements each (where \( n = 3 \)).
   - **Indexing in Python:**
     - **1-based Indexing:** \( w_1, w_2, w_3 \) in mathematical notation.
     - **0-based Indexing:** \( w[0], w[1], w[2] \) in Python (NumPy).

3. **Non-Vectorized Implementation:**
   - **Manual Computation:** Multiply each parameter \( w_j \) by its associated feature \( x_j \) and sum up.
   - **For Loop Example:**
     ```python
     f = 0
     for j in range(n):
         f += w[j] * x[j]
     f += b
     ```
   - **Inefficiency:** Not practical for large \( n \) (e.g., \( n = 100,000 \)) due to computational overhead.

4. **Vectorized Implementation:**
   - **Mathematical Expression:** \( f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b \)
   - **NumPy Implementation:**
     ```python
     fp = np.dot(w, x) + b
     ```
   - **Advantages:**
     - **Shorter Code:** Only one line for the dot product.
     - **Faster Execution:** Uses parallel hardware acceleration (CPU or GPU).
     - **Efficient for Large \( n \):** Handles large vectors efficiently.

5. **Benefits of Vectorization:**
   - **Code Efficiency:** Reduces code length and complexity.
   - **Execution Speed:** Faster execution due to parallel processing capabilities.

6. **Hardware Utilization:**
   - **CPU/GPU Acceleration:** Vectorized operations can leverage parallel hardware to speed up computation, making them much more efficient compared to iterative methods. 

## Vectorization part 2

**Concept Overview:**
- **Vectorization** is a technique used to make algorithms more efficient by performing operations on whole vectors or matrices at once, rather than element by element.
- This approach leverages modern hardware capabilities, such as CPUs and GPUs, to execute operations in parallel, significantly speeding up computations.

**Key Points:**

1. **Comparison of Computation Methods:**
   - **Without Vectorization:**
     - **For Loop Example:**
       ```python
       for j in range(n):
           result += w[j] * x[j]
       result += b
       ```
     - **Time Complexity:** Sequential processing of each element, leading to longer execution times, especially as `n` increases.
   
   - **With Vectorization:**
     - **Vectorized Code Example:**
       ```python
       result = np.dot(w, x) + b
       ```
     - **Time Complexity:** Parallel processing, resulting in faster execution times, particularly beneficial for large datasets.

2. **Vectorization in NumPy:**
   - NumPy, a numerical linear algebra library in Python, supports vectorized operations.
   - **Dot Product Example:**
     ```python
     fp = np.dot(w, x) + b
     ```
   - **Benefits:**
     - **Shorter Code:** Reduced lines of code compared to for loops.
     - **Faster Execution:** Efficient use of parallel hardware, resulting in quicker computations.

3. **Understanding Parallel Processing:**
   - **Non-Vectorized Operations:** Perform calculations sequentially, step by step.
   - **Vectorized Operations:** Utilize parallel processing to compute all values simultaneously, then aggregate results efficiently.

4. **Example with Multiple Features:**
   - **Without Vectorization:**
     ```python
     for j in range(16):
         w[j] = w[j] - 0.1 * d[j]
     ```
   - **With Vectorization:**
     ```python
     w = w - 0.1 * d
     ```
   - **Benefits of Vectorized Implementation:** Drastically reduces computation time, especially with large datasets.

5. **Application in Machine Learning:**
   - Vectorization is crucial for efficient implementation of machine learning algorithms, such as multiple linear regression and gradient descent.
   - It is essential for handling large-scale problems and datasets efficiently.

## Gradient descent for multiple linear regression

### Implementing Gradient Descent for Multiple Linear Regression with Vectorization

#### **1. Recap of Multiple Linear Regression:**

- **Model Equation:**
  In multiple linear regression, we predict a target \( y \) using a linear combination of features \( x \):
  \[
  f_w, b(x) = w \cdot x + b
  \]
  where:
  - \( w \) is a vector of weights (one per feature),
  - \( x \) is a vector of features,
  - \( b \) is the bias term (intercept).

#### **2. Cost Function:**

- **Cost Function \( J(w, b) \):**
  The cost function measures how well our model is performing by comparing the predicted values with the actual values:
  \[
  J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (f_w, b(x^{(i)}) - y^{(i)})^2
  \]
  where \( m \) is the number of training examples, \( x^{(i)} \) is the \(i\)-th training example, and \( y^{(i)} \) is the corresponding target value.

#### **3. Gradient Descent for Multiple Linear Regression:**

- **Gradient Descent Update Rule:**
  We iteratively adjust the weights \( w \) and bias \( b \) to minimize the cost function:
  \[
  w := w - \alpha \frac{\partial J(w, b)}{\partial w}
  \]
  \[
  b := b - \alpha \frac{\partial J(w, b)}{\partial b}
  \]
  where \( \alpha \) is the learning rate, and the derivatives are calculated as follows:

- **Derivative Calculations:**
  For weights \( w_j \):
  \[
  \frac{\partial J(w, b)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (f_w, b(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}
  \]
  For bias \( b \):
  \[
  \frac{\partial J(w, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (f_w, b(x^{(i)}) - y^{(i)})
  \]

#### **4. Vectorized Implementation:**

- **Vectorized Cost Function:**
  Using vector notation:
  \[
  J(w, b) = \frac{1}{2m} \| Xw + b - y \|^2
  \]
  where \( X \) is the matrix of features, \( w \) is the vector of weights, \( b \) is the bias term, and \( y \) is the vector of target values.

- **Vectorized Gradient Descent:**
  Update rules for weights and bias are:
  \[
  w := w - \alpha \frac{1}{m} X^T (Xw + b - y)
  \]
  \[
  b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (Xw + b - y)_{i}
  \]

#### **5. Implementation:**

- **Initialize Parameters:**
  ```python
  import numpy as np

  def initialize_parameters(n):
      w = np.zeros((n, 1))
      b = 0
      return w, b
  ```

- **Compute Cost Function:**
  ```python
  def compute_cost(X, y, w, b):
      m = X.shape[0]
      predictions = np.dot(X, w) + b
      cost = (1/(2*m)) * np.sum(np.square(predictions - y))
      return cost
  ```

- **Gradient Descent Update:**
  ```python
  def gradient_descent(X, y, w, b, alpha, num_iterations):
      m = X.shape[0]
      for i in range(num_iterations):
          predictions = np.dot(X, w) + b
          dw = (1/m) * np.dot(X.T, (predictions - y))
          db = (1/m) * np.sum(predictions - y)
          w -= alpha * dw
          b -= alpha * db
      return w, b
  ```

#### **6. Alternative: The Normal Equation:**

- **Normal Equation:**
  The normal equation provides a closed-form solution to find \( w \) and \( b \):
  \[
  w = (X^T X)^{-1} X^T y
  \]
  \[
  b = \text{mean}(y - Xw)
  \]

- **Pros and Cons:**
  - **Pros:** Provides a direct solution without iterative optimization.
  - **Cons:** Computationally expensive for large datasets or when the number of features is very large.


# Gradient descent in practice

## Feature scaling part 1

### Feature Scaling for Gradient Descent Efficiency

#### **1. Understanding the Problem:**

When you have features with very different ranges, the cost function’s contours can become elongated in one direction, which complicates the gradient descent process. For instance:

- **Feature Example:**
  - **Feature 1 (Size of house):** Ranges from 300 to 2000 square feet.
  - **Feature 2 (Number of bedrooms):** Ranges from 0 to 5.

If you apply gradient descent directly to these features, the cost function’s contour plot might look elongated along one axis, leading to inefficient convergence of gradient descent.

#### **2. The Impact of Feature Scaling:**

- **Unscaled Features:**
  Features with different scales result in a contour plot that looks like an ellipse or an oval. Gradient descent might take a longer path to converge because it “bounces” more in the direction of the feature with a larger range.

- **Scaled Features:**
  By scaling features so they all fall within a similar range (e.g., 0 to 1), the cost function’s contour plot becomes more circular. This makes gradient descent more efficient as it can converge more quickly.

#### **3. Example of Feature Scaling:**

- **Original Features:**
  - **Size (x1):** 300 to 2000
  - **Bedrooms (x2):** 0 to 5

- **Scaled Features:**
  - **Size (x1):** Scaled to range from 0 to 1.
  - **Bedrooms (x2):** Scaled to range from 0 to 1.

#### **4. Implementing Feature Scaling:**

Feature scaling typically involves normalization or standardization:

- **Normalization (Min-Max Scaling):**
  Scales features to a fixed range, usually [0, 1].
  \[
  x_{\text{scaled}} = \frac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)}
  \]

- **Standardization (Z-score Scaling):**
  Scales features to have zero mean and unit variance.
  \[
  x_{\text{standardized}} = \frac{x - \mu}{\sigma}
  \]
  where \( \mu \) is the mean and \( \sigma \) is the standard deviation of the feature.

#### **5. Example Python Code for Feature Scaling:**

Here’s how you can implement feature scaling in Python using NumPy:

```python
import numpy as np

# Min-Max Normalization
def min_max_scaling(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_scaled = (X - X_min) / (X_max - X_min)
    return X_scaled

# Z-score Standardization
def z_score_scaling(X):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standardized = (X - X_mean) / X_std
    return X_standardized

# Example usage
X = np.array([[300, 0], [2000, 5]])  # Example feature matrix
X_normalized = min_max_scaling(X)
X_standardized = z_score_scaling(X)

print("Normalized Features:\n", X_normalized)
print("Standardized Features:\n", X_standardized)
```

#### **6. Benefits of Feature Scaling:**

- **Faster Convergence:**
  Gradient descent converges more quickly due to more balanced feature scales.
  
- **Improved Performance:**
  More reliable and consistent results as the algorithm handles all features more equally.


## Feature scaling part 2

### Implementing Feature Scaling

Feature scaling ensures that all features are on a comparable scale, which can significantly improve the performance of gradient descent. Here’s a step-by-step guide on how to implement different feature scaling techniques:

#### **1. Min-Max Normalization**

**Purpose:** Rescales features to fall within a specific range, typically [0, 1].

**Formula:**
\[
x_{\text{scaled}} = \frac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)}
\]

**Example:**
- **Feature 1 (Size of house):** Ranges from 300 to 2000.
- **Feature 2 (Number of bedrooms):** Ranges from 0 to 5.

To normalize:
- **For x1:** \(\text{min}(x_1) = 300\), \(\text{max}(x_1) = 2000\)
  \[
  x_{1_{\text{scaled}}} = \frac{x_1 - 300}{2000 - 300}
  \]
  
- **For x2:** \(\text{min}(x_2) = 0\), \(\text{max}(x_2) = 5\)
  \[
  x_{2_{\text{scaled}}} = \frac{x_2 - 0}{5 - 0}
  \]

#### **2. Mean Normalization**

**Purpose:** Centers features around zero, often with values between -1 and 1.

**Formula:**
\[
x_{\text{normalized}} = \frac{x - \text{mean}(x)}{\text{max}(x) - \text{min}(x)}
\]

**Example:**
- **Feature 1 (Size of house):** Mean (\(\mu_1\)) = 600
  \[
  x_{1_{\text{normalized}}} = \frac{x_1 - 600}{2000 - 300}
  \]

- **Feature 2 (Number of bedrooms):** Mean (\(\mu_2\)) = 2.3
  \[
  x_{2_{\text{normalized}}} = \frac{x_2 - 2.3}{5 - 0}
  \]

#### **3. Z-Score Normalization (Standardization)**

**Purpose:** Rescales features to have zero mean and unit variance, making them comparable across different scales.

**Formula:**
\[
x_{\text{standardized}} = \frac{x - \mu}{\sigma}
\]
where \(\mu\) is the mean and \(\sigma\) is the standard deviation.

**Example:**
- **Feature 1 (Size of house):** Mean (\(\mu_1\)) = 600, Standard Deviation (\(\sigma_1\)) = 450
  \[
  x_{1_{\text{standardized}}} = \frac{x_1 - 600}{450}
  \]

- **Feature 2 (Number of bedrooms):** Mean (\(\mu_2\)) = 2.3, Standard Deviation (\(\sigma_2\)) = 1.4
  \[
  x_{2_{\text{standardized}}} = \frac{x_2 - 2.3}{1.4}
  \]

#### **4. Python Code for Feature Scaling**

Here’s a Python implementation for normalization and standardization:

```python
import numpy as np

# Min-Max Normalization
def min_max_scaling(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_scaled = (X - X_min) / (X_max - X_min)
    return X_scaled

# Mean Normalization
def mean_normalization(X):
    X_mean = X.mean(axis=0)
    X_max = X.max(axis=0)
    X_min = X.min(axis=0)
    X_normalized = (X - X_mean) / (X_max - X_min)
    return X_normalized

# Z-Score Standardization
def z_score_standardization(X):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standardized = (X - X_mean) / X_std
    return X_standardized

# Example usage
X = np.array([[300, 0], [2000, 5]])  # Example feature matrix

# Apply different scaling methods
X_min_max_scaled = min_max_scaling(X)
X_mean_normalized = mean_normalization(X)
X_z_score_standardized = z_score_standardization(X)

print("Min-Max Scaled Features:\n", X_min_max_scaled)
print("Mean Normalized Features:\n", X_mean_normalized)
print("Z-Score Standardized Features:\n", X_z_score_standardized)
```

#### **5. When to Use Feature Scaling:**

- **Features with different ranges:** Always scale features with different ranges to ensure comparable gradients during optimization.
- **Large or small values:** Features with very large or small values relative to other features can benefit from scaling.

#### **6. Summary:**

Feature scaling improves gradient descent efficiency by normalizing or standardizing feature values, making the optimization process faster and more reliable. Choose the appropriate scaling method based on the characteristics of your features and the specific needs of your model.

## Checking gradient descent for convergence 

### Understanding Gradient Descent and Learning Curves

**Gradient Descent Overview:**

Gradient descent is an optimization algorithm used to minimize a cost function \( J \) by iteratively updating parameters \( w \) and \( b \). The goal is to find the values of \( w \) and \( b \) that result in the smallest possible cost \( J \). A key aspect of gradient descent is the choice of the learning rate \( \alpha \), which controls the size of the steps taken towards the minimum.

**1. Learning Curve:**

A learning curve plots the cost function \( J \) against the number of iterations of gradient descent. Here's how to interpret it:

- **Horizontal Axis:** Number of iterations of gradient descent.
- **Vertical Axis:** Value of the cost function \( J \) for the parameters \( w \) and \( b \) obtained after each iteration.

**Key Characteristics of a Well-Running Gradient Descent:**

- **Decreasing Cost:** Ideally, the cost \( J \) should decrease with each iteration. A well-functioning gradient descent will show a downward trend in the learning curve.
- **Convergence:** When the curve flattens out, it indicates that the algorithm has converged, meaning the cost is no longer decreasing significantly with further iterations.

**What to Look For:**

1. **Cost Decreasing:** Ensure that the cost \( J \) is decreasing after each iteration. If the cost increases, it could be due to a large learning rate \( \alpha \) or a bug in the code.

2. **Convergence:** Check if the cost \( J \) plateaus or levels off. If the cost does not decrease significantly after a certain number of iterations, gradient descent has likely converged.

**Handling Learning Rate \( \alpha \):**

- **Too Large \( \alpha \):** If the learning rate is too large, the cost \( J \) might oscillate or increase rather than decrease, indicating that the steps are too big and the algorithm is not converging.

- **Too Small \( \alpha \):** If the learning rate is too small, gradient descent may converge very slowly, taking a long time to decrease the cost function.

**Automatic Convergence Test:**

In addition to visual inspection, you can use an automatic convergence test:

- **Epsilon (\( \epsilon \)):** Define a small threshold value, such as \( \epsilon = 0.001 \). If the cost \( J \) decreases by less than \( \epsilon \) between iterations, the algorithm can be considered to have converged.

**Python Example of Gradient Descent and Learning Curve Plotting:**

Here is an example of how to implement gradient descent and plot the learning curve in Python:

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_cost(X, y, w, b):
    m = len(y)
    predictions = X.dot(w) + b
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, w, b, alpha, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)

    for i in range(num_iterations):
        predictions = X.dot(w) + b
        error = predictions - y
        w_gradient = (1/m) * X.T.dot(error)
        b_gradient = (1/m) * np.sum(error)
        
        w -= alpha * w_gradient
        b -= alpha * b_gradient
        
        cost_history[i] = compute_cost(X, y, w, b)
    
    return w, b, cost_history

# Example data
X = np.array([[1, 1], [2, 2], [3, 3]])  # Feature matrix
y = np.array([1, 2, 3])  # Target values
w = np.array([0.1, 0.1])  # Initial weights
b = 0.1  # Initial bias
alpha = 0.01  # Learning rate
num_iterations = 1000  # Number of iterations

w, b, cost_history = gradient_descent(X, y, w, b, alpha, num_iterations)

# Plotting the learning curve
plt.plot(range(num_iterations), cost_history, 'b-')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('Learning Curve')
plt.show()
```

## Choosing the learning rate

### Choosing an Appropriate Learning Rate for Gradient Descent

Choosing the right learning rate \( \alpha \) is crucial for the effective training of gradient descent algorithms. An inappropriate learning rate can lead to slow convergence or even prevent convergence altogether. Here's a detailed guide on how to choose a good learning rate:

**1. Effects of Learning Rate on Gradient Descent:**

- **Too Large Learning Rate:** If \( \alpha \) is too large, the updates to the parameters may overshoot the minimum of the cost function. This results in the cost function \( J \) oscillating or even increasing, rather than consistently decreasing. The algorithm may never converge to the optimal solution.

  **Illustration:**
  - With a large learning rate, the update step might overshoot the minimum. For example, if you start at a point on the cost function curve, the update might move you to a higher cost region before moving back down.

- **Too Small Learning Rate:** If \( \alpha \) is too small, the algorithm will converge very slowly, taking a large number of iterations to make noticeable progress. While it may eventually converge, the process can be inefficient.

  **Illustration:**
  - With a very small learning rate, the updates are so tiny that the cost function decreases very slowly. This might result in many iterations before reaching the minimum.

**2. Debugging Tips:**

- **Consistent Decrease with Small \( \alpha \):** Set \( \alpha \) to a very small value (e.g., 0.001) and check if the cost function \( J \) consistently decreases with each iteration. If the cost function sometimes increases even with a small \( \alpha \), it usually indicates a bug in the code rather than an issue with the learning rate.

- **Correct Update Formula:** Ensure the update step uses the correct formula. The update should be:
  \[
  w_i = w_i - \alpha \times \frac{\partial J}{\partial w_i}
  \]
  If you accidentally use \( w_i = w_i + \alpha \times \frac{\partial J}{\partial w_i} \), the cost function will move further away from the minimum.

**3. Choosing the Learning Rate:**

- **Experimentation:** Start with a small learning rate (e.g., 0.001) and gradually increase it. Try different values like 0.01, 0.1, and so on. Plot the cost function \( J \) for different values of \( \alpha \) to see which learning rate leads to the fastest and most consistent decrease in \( J \).

- **Scaling Approach:** Use a systematic approach to scale the learning rate. For example, if you start with 0.001, try 0.003 (3 times larger), then 0.01 (again roughly 3 times larger), and so on. This helps in finding an optimal range for \( \alpha \).

**4. Practical Steps for Choosing \( \alpha \):**

1. **Initial Small Value:** Begin with a small learning rate and observe the behavior of the cost function.

2. **Increase Gradually:** Increase the learning rate incrementally and plot the cost function \( J \) after a few iterations for each value.

3. **Find Optimal Range:** Identify the range where the learning rate starts to become too large, leading to oscillations or divergence. Then, select a learning rate that is just below this threshold.

4. **Feature Scaling:** Feature scaling can also impact the performance of gradient descent. Scaling features to a similar range can help in choosing a more effective learning rate.

**Python Example of Learning Rate Experimentation:**

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, y, w, b, alpha, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)

    for i in range(num_iterations):
        predictions = X.dot(w) + b
        error = predictions - y
        w_gradient = (1/m) * X.T.dot(error)
        b_gradient = (1/m) * np.sum(error)
        
        w -= alpha * w_gradient
        b -= alpha * b_gradient
        
        cost_history[i] = compute_cost(X, y, w, b)
    
    return w, b, cost_history

# Example data
X = np.array([[1, 1], [2, 2], [3, 3]])  # Feature matrix
y = np.array([1, 2, 3])  # Target values
w = np.array([0.1, 0.1])  # Initial weights
b = 0.1  # Initial bias
num_iterations = 1000  # Number of iterations

# List of learning rates to test
learning_rates = [0.001, 0.01, 0.1]

for alpha in learning_rates:
    w, b, cost_history = gradient_descent(X, y, w, b, alpha, num_iterations)
    
    # Plotting the learning curve
    plt.plot(range(num_iterations), cost_history, label=f'alpha={alpha}')

plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('Learning Curves for Different Learning Rates')
plt.legend()
plt.show()
```

## Feature Engineering

### Feature Engineering: Enhancing Your Learning Algorithm

Feature engineering is a critical step in building effective machine learning models. By creating or transforming features, you can often improve your model’s performance significantly. Here’s an overview of the process and an example to illustrate how it works:

**1. What is Feature Engineering?**

Feature engineering involves creating new features or modifying existing ones to better capture the underlying patterns in the data. It can involve:

- **Creating New Features:** Deriving new features from existing ones based on domain knowledge or intuition.
- **Transforming Features:** Applying mathematical transformations to features to highlight relationships or patterns.
- **Combining Features:** Combining multiple features into a single feature that may be more predictive.

**2. Example of Feature Engineering:**

Consider the example of predicting the price of a house based on the following features:

- \( x_1 \): Width of the lot (frontage)
- \( x_2 \): Depth of the lot

Initially, you might use a simple model like:
\[ f(x) = w_1 x_1 + w_2 x_2 + b \]

However, you might realize that the area of the lot (width times depth) could be a more informative feature for predicting the price. Therefore, you create a new feature:
\[ x_3 = x_1 \times x_2 \]

With this new feature, you can modify your model to:
\[ f(x) = w_1 x_1 + w_2 x_2 + w_3 x_3 + b \]

Here, \( x_3 \) represents the area of the lot. The model can now learn the importance of \( x_3 \) in predicting the house price, potentially leading to better performance.

**3. Benefits of Feature Engineering:**

- **Enhanced Model Performance:** By introducing more relevant features, you provide the model with additional information that can improve its predictive accuracy.
- **Simplified Learning Process:** New features can make patterns in the data more apparent, which can make it easier for the algorithm to learn from the data.
- **Domain Knowledge Utilization:** Feature engineering allows you to incorporate your understanding of the problem domain into the model, potentially leading to more meaningful features.

**4. Non-Linear Feature Engineering:**

In addition to creating features that represent linear relationships, you can also engineer features that capture non-linear relationships. For example:

- **Polynomial Features:** Create features that represent polynomial terms (e.g., \( x_1^2 \), \( x_2^2 \), or \( x_1 \times x_2 \)).
- **Interaction Terms:** Capture interactions between features that may be important for prediction.
- **Logarithmic or Exponential Transformations:** Apply transformations to capture non-linear patterns.

**5. Practical Tips for Feature Engineering:**

- **Explore Data Thoroughly:** Understand the relationships between features and the target variable. This can guide you in creating meaningful features.
- **Experiment with Different Features:** Try various feature combinations and transformations. Use model performance metrics to evaluate their impact.
- **Visualize Feature Relationships:** Use plots and visualizations to understand how different features relate to each other and to the target variable.

**Python Example of Feature Engineering:**

Here’s a simple Python example of how to implement feature engineering:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Example data
data = {
    'width': [10, 20, 30, 40, 50],
    'depth': [15, 25, 35, 45, 55],
    'price': [150000, 250000, 350000, 450000, 550000]
}
df = pd.DataFrame(data)

# Create new feature: area
df['area'] = df['width'] * df['depth']

# Features and target
X = df[['width', 'depth', 'area']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plotting feature importance
feature_importance = model.coef_
plt.bar(X.columns, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
```


## Polynomial regression

### Polynomial Regression: Fitting Curves to Your Data

**Polynomial regression** is an extension of linear regression that allows you to fit non-linear functions to your data. This approach can capture more complex relationships between features and the target variable. Here’s a summary of the key concepts and steps for using polynomial regression:

#### **1. Understanding Polynomial Regression**

Polynomial regression involves adding polynomial terms (like \(x^2\), \(x^3\), etc.) to the model to capture non-linear relationships. For example:

- **Linear Model:** \( f(x) = w_1 x + b \)
- **Quadratic Model:** \( f(x) = w_1 x + w_2 x^2 + b \)
- **Cubic Model:** \( f(x) = w_1 x + w_2 x^2 + w_3 x^3 + b \)

In polynomial regression, you transform the feature \( x \) to include higher-order terms (e.g., \( x^2 \), \( x^3 \)) and then fit a linear model to these transformed features.

#### **2. Implementing Polynomial Regression**

To fit a polynomial regression model, follow these steps:

**a. Choose Polynomial Features:**
   - Determine which polynomial terms to include based on your intuition about the data. For example, use \( x \), \( x^2 \), and \( x^3 \) if you suspect a cubic relationship.

**b. Feature Scaling:**
   - Apply feature scaling, especially when using polynomial features. Polynomial features can have very different ranges (e.g., \( x^2 \) ranges from 1 to 1,000,000), which can affect the performance of gradient descent. Scaling ensures that all features are on a comparable scale.

**c. Train the Model:**
   - Fit a linear regression model to the polynomial features.

**d. Evaluate the Model:**
   - Assess the performance of the polynomial model using metrics like Mean Squared Error (MSE) and visualize the fit to ensure it captures the underlying pattern.

#### **3. Python Example of Polynomial Regression**

Here’s how you can implement polynomial regression using Python and `Scikit-learn`:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Example data
data = {
    'size': [100, 200, 300, 400, 500],
    'price': [150000, 250000, 350000, 450000, 550000]
}
df = pd.DataFrame(data)

# Feature and target
X = df[['size']]
y = df['price']

# Create polynomial features
poly = PolynomialFeatures(degree=3)  # Degree of the polynomial
X_poly = poly.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plotting
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, model.predict(poly.transform(X)), color='red', label='Polynomial Fit')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price')
plt.title('Polynomial Regression Fit')
plt.legend()
plt.show()
```

#### **4. Choosing Features**

When choosing features for polynomial regression:

- **Analyze Data:** Use domain knowledge and data analysis to decide which polynomial terms might be relevant.
- **Experiment:** Try different polynomial degrees and combinations of features to find the best fit.
- **Validate:** Use cross-validation to ensure that the polynomial model generalizes well to unseen data.

#### **5. Using Scikit-learn**

`Scikit-learn` provides a straightforward way to implement polynomial regression with tools like `PolynomialFeatures` and `LinearRegression`. Familiarize yourself with these tools as they are widely used in practice.
