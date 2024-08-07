# Classification with logistic regression

## Motivations

### Classification Overview

- **Classification** involves predicting discrete categories or classes, as opposed to continuous values.
- **Binary Classification** is a type of classification problem where the output is one of two possible values (e.g., spam or not spam, fraudulent or not fraudulent).

### Linear Regression vs. Logistic Regression

- **Linear Regression**:
  - Predicts continuous values.
  - When used for classification, it predicts values between 0 and 1 but can also output values less than 0 or greater than 1, which aren't ideal for categorical outputs.
  - The decision boundary might shift with new data points, making it less reliable for classification.

- **Logistic Regression**:
  - Aimed at binary classification.
  - Outputs probabilities that are constrained between 0 and 1, which is suitable for classification problems.
  - The name "logistic regression" might be confusing as it's used for classification, but it stems from the historical naming conventions.

### Key Concepts

- **Decision Boundary**: The line or surface that separates different classes in the feature space.
- **Thresholding**: In binary classification, a threshold (like 0.5) is used to decide the class based on the predicted probability.

### Practical Considerations

- **Linear Regression for Classification**: Often inadequate because it doesn’t handle probabilities properly and can lead to incorrect decision boundaries.
- **Logistic Regression**: Designed to address these issues and is more suitable for binary classification tasks.

## Logistic Regression

### Understanding Logistic Regression

Logistic regression is a powerful and widely-used algorithm for binary classification tasks. Let's break down the key components and concepts:

#### **The Sigmoid Function**

- **Sigmoid Function Formula**: \( g(z) = \frac{1}{1 + e^{-z}} \)
- **Characteristics**:
  - **Output Range**: The function outputs values between 0 and 1, making it suitable for probability estimation.
  - **Shape**: The function has an S-shaped curve, smoothly transitioning from 0 to 1.
  - **Behavior**:
    - When \( z \) is very large, \( g(z) \) approaches 1.
    - When \( z \) is very small, \( g(z) \) approaches 0.
    - When \( z = 0 \), \( g(z) = 0.5 \).

#### **Logistic Regression Model**

- **Model Formula**: \( f(x) = g(wx + b) \)
  - **Inputs**: \( x \) represents the features.
  - **Parameters**: \( w \) (weights) and \( b \) (bias) are learned during training.
  - **Output**: The function \( g \) outputs a probability between 0 and 1.

#### **Interpreting the Output**

- **Probability Interpretation**:
  - The output of the logistic regression model represents the probability that the output label \( y \) is 1 given the input \( x \).
  - For example, if the output probability is 0.7, it means there's a 70% chance that \( y \) is 1 (e.g., a tumor is malignant).

#### **Probability Relationship**

- The probabilities of \( y \) being 0 and 1 must sum to 1.
  - If the probability of \( y \) being 1 is 0.7, then the probability of \( y \) being 0 is 0.3.

#### **Notation**

- Sometimes, you might see the probability notation as \( P(y = 1 | x) \), where the vertical bar denotes conditional probability given \( x \), and \( w \) and \( b \) are parameters.

#### **Practical Considerations**

- Logistic regression is often used in applications like internet advertising, fraud detection, and medical diagnosis due to its effectiveness in binary classification tasks.

## Decision boundary

### Logistic Regression and Decision Boundaries

To understand how logistic regression makes predictions, we need to dive into the concept of the decision boundary. Let's recap and build upon what you've learned about the logistic regression model.

#### Logistic Regression Recap

The logistic regression model outputs predictions in two steps:

1. **Compute \( z \):**
   \[
   z = w \cdot x + b
   \]
   Here, \( w \) is the weight vector, \( x \) is the feature vector, and \( b \) is the bias term.

2. **Apply the Sigmoid Function \( g(z) \):**
   \[
   g(z) = \frac{1}{1 + e^{-z}}
   \]
   This transforms \( z \) into a value between 0 and 1, which can be interpreted as a probability.

Combining these steps, the logistic regression model \( f(x) \) is:
\[
f(x) = g(w \cdot x + b)
\]

This can be interpreted as the probability that the output \( y \) is 1 given the input \( x \) and parameters \( w \) and \( b \):
\[
f(x) = P(y = 1 \mid x; w, b)
\]

#### Decision Boundary

The decision boundary is the threshold that the model uses to classify the data points.

1. **Threshold of 0.5:**
   \[
   \text{If } f(x) \geq 0.5, \text{ predict } y = 1
   \]
   \[
   \text{If } f(x) < 0.5, \text{ predict } y = 0
   \]

2. **Threshold Applied to Sigmoid Function:**
   \[
   f(x) = g(z) \geq 0.5
   \]
   \[
   g(z) = \frac{1}{1 + e^{-z}} \geq 0.5 \implies z \geq 0
   \]

3. **Decision Rule:**
   \[
   z = w \cdot x + b \geq 0
   \]
   \[
   \text{Predict } y = 1 \text{ if } w \cdot x + b \geq 0
   \]
   \[
   \text{Predict } y = 0 \text{ if } w \cdot x + b < 0
   \]

#### Visualizing the Decision Boundary

Consider a classification problem with two features \( x_1 \) and \( x_2 \). The decision boundary can be visualized in a 2D plane.

1. **Example with Linear Decision Boundary:**
   - Suppose \( w_1 = 1 \), \( w_2 = 1 \), and \( b = -3 \):
     \[
     z = w_1 x_1 + w_2 x_2 + b = x_1 + x_2 - 3
     \]
   - The decision boundary is:
     \[
     x_1 + x_2 = 3
     \]
   - This line divides the plane into two regions: one where the model predicts \( y = 1 \) and another where it predicts \( y = 0 \).

2. **Non-linear Decision Boundaries:**
   - By using polynomial features, logistic regression can create more complex decision boundaries.
   - Example with quadratic features:
     \[
     z = w_1 x_1^2 + w_2 x_2^2 + b
     \]
     - Suppose \( w_1 = 1 \), \( w_2 = 1 \), and \( b = -1 \):
       \[
       z = x_1^2 + x_2^2 - 1
       \]
     - The decision boundary is:
       \[
       x_1^2 + x_2^2 = 1
       \]
     - This forms a circle in the feature space, predicting \( y = 1 \) outside the circle and \( y = 0 \) inside.

3. **Higher-order Polynomial Features:**
   - By including higher-order terms, more complex decision boundaries can be achieved, allowing logistic regression to fit more intricate patterns in the data.

#### Implementing and Visualizing Logistic Regression

In practice, you can use software libraries such as Python's scikit-learn to implement logistic regression and visualize the decision boundary. Here’s an example code snippet to visualize decision boundaries using logistic regression:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Create a synthetic dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
```

This code generates a synthetic dataset, fits a logistic regression model, and visualizes the decision boundary. The decision boundary is the line (or curve) that separates the two classes predicted by the model.

### Conclusion

Logistic regression is a powerful and widely-used algorithm for binary classification problems. By understanding the decision boundary and how logistic regression uses the sigmoid function to make predictions, you can better interpret and implement this model in your work. In the next steps, you'll explore the cost function for logistic regression and learn how to apply gradient descent to train the model.

# Cost function for logistic regression

## Cost function for logistic regression

1. **Training Set and Features**:
   - Each row in the training set corresponds to a patient visit with features like tumor size and patient's age.
   - \( m \) denotes the number of training examples, and each example has \( n \) features.
   - The target label \( y \) is binary (0 or 1).

2. **Logistic Regression Model**:
   - Defined by the equation \( f(x) = \frac{1}{1 + e^{-(w \cdot x + b)}} \).
   - Parameters \( w \) and \( b \) need to be chosen to fit the training data.

3. **Squared Error Cost Function**:
   - For linear regression, the squared error cost function works well because it is convex.
   - However, using it for logistic regression results in a non-convex cost function with many local minima, making gradient descent unreliable.

4. **New Cost Function for Logistic Regression**:
   - Instead of the squared error, we use a different form of the loss function to ensure convexity.
   - The loss function for logistic regression:
     - If \( y = 1 \), \( \text{Loss} = -\log(f(x)) \).
     - If \( y = 0 \), \( \text{Loss} = -\log(1 - f(x)) \).

5. **Understanding the Loss Function**:
   - For \( y = 1 \):
     - The loss is low when \( f(x) \) is close to 1, meaning the prediction is accurate.
     - As \( f(x) \) decreases, the loss increases, penalizing incorrect predictions.
   - For \( y = 0 \):
     - The loss is low when \( f(x) \) is close to 0.
     - As \( f(x) \) increases, the loss increases, again penalizing incorrect predictions.

6. **Convex Cost Function**:
   - The overall cost function is the average of the loss functions across all training examples.
   - With the new loss function, the cost function becomes convex, allowing gradient descent to reliably find the global minimum.

## Simplified Cost Function for Logistic Regression

1. **Simplified Loss Function**:
   - The original loss function for logistic regression is:
     - If \( y = 1 \): \( -\log(f(x)) \)
     - If \( y = 0 \): \( -\log(1 - f(x)) \)
   - We can combine these cases into a single formula:
     \[
     \text{Loss} = -[y \log(f(x)) + (1 - y) \log(1 - f(x))]
     \]
   - This formula simplifies implementation by handling both cases (when \( y = 1 \) or \( y = 0 \)) in one line.

2. **Cost Function**:
   - The cost function \( J(w, b) \) is the average of the loss over all training examples:
     \[
     J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \text{Loss}(f(x_i), y_i)
     \]
   - Substituting the simplified loss function into this equation, we get:
     \[
     J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log(f(x_i)) + (1 - y_i) \log(1 - f(x_i))]
     \]
   - This is the cost function used for training logistic regression models.

3. **Why This Cost Function?**:
   - This cost function is derived from maximum likelihood estimation, a statistical principle for efficiently estimating model parameters.
   - It has the property of being convex, which means gradient descent can reliably find the global minimum.


# Gradient Descent for logistic regression

## Gradient Descent Implementation

We focus on applying gradient descent to fit the parameters \( w \) and \( b \) of a logistic regression model. Here's a summary of the key points:

1. **Gradient Descent for Logistic Regression**:
   - To fit the logistic regression model, we minimize the cost function \( J(w, b) \) using gradient descent.
   - Gradient descent updates the parameters iteratively to reduce the cost. The update rules are:
     - For each parameter \( w_j \):
       \[
       w_j \leftarrow w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (f(x_i) - y_i) \cdot x_{ij}
       \]
     - For the parameter \( b \):
       \[
       b \leftarrow b - \alpha \frac{1}{m} \sum_{i=1}^{m} (f(x_i) - y_i)
       \]
   - Here, \( f(x_i) \) is the prediction for the \( i \)-th training example, \( y_i \) is the actual label, and \( x_{ij} \) is the \( j \)-th feature of the \( i \)-th training example.

2. **Difference from Linear Regression**:
   - The update equations for \( w_j \) and \( b \) in logistic regression look similar to those in linear regression.
   - The key difference is that in linear regression, \( f(x) = wx + b \), whereas in logistic regression, \( f(x) \) is the sigmoid function applied to \( wx + b \). This makes logistic regression suited for classification tasks.

3. **Gradient Descent Implementation**:
   - Updates are done simultaneously for all parameters, but the equations for gradient descent are the same as those used in linear regression, despite the difference in \( f(x) \).

4. **Vectorization**:
   - Vectorized implementations can speed up gradient descent by performing operations on entire vectors or matrices instead of individual elements. This approach is similar to what was used in linear regression.

5. **Feature Scaling**:
   - Scaling features to a similar range (e.g., between -1 and 1) can help gradient descent converge faster. This principle applies to logistic regression as well.

# The problem of overfitting

## The Problem of Overfitting

### **Understanding Overfitting and Underfitting**

1. **Underfitting**:
   - **Definition**: When a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and new examples.
   - **Example**: A linear model used to predict housing prices might not fit the data well if the true relationship is non-linear (e.g., if the price flattens out as the size increases).
   - **Technical Term**: High bias, as the model makes strong assumptions that do not fit the data well.

2. **Overfitting**:
   - **Definition**: When a model is too complex and fits the training data very well but performs poorly on new examples because it has learned noise or specificities of the training set.
   - **Example**: A fourth-order polynomial model might fit the training data perfectly but would likely perform poorly on new data due to its high variance and sensitivity to noise.
   - **Technical Term**: High variance, as the model is overly flexible and adjusts too closely to the training data.

3. **Just Right**:
   - **Definition**: A model that balances complexity and simplicity, fitting the training data well and generalizing effectively to new examples.
   - **Example**: A quadratic model for predicting housing prices or a logistic regression model with added quadratic terms for classification tasks.

### **Key Points to Address**

- **Bias vs. Variance**: 
  - **High Bias**: Model is too simple (underfitting) and cannot capture the data's complexity.
  - **High Variance**: Model is too complex (overfitting) and fits the training data too closely.

- **Generalization**: The goal is to create a model that performs well not only on the training data but also on new, unseen examples.

### **Future Steps**

1. **Regularization**:
   - **Definition**: A technique used to reduce overfitting by adding a penalty for more complex models. This can help in minimizing the variance of the model.
   - **Method**: We'll explore how regularization can improve model performance by discouraging overly complex models.

2. **Feature Engineering**:
   - **Feature Selection**: Choosing the right features to avoid overfitting by reducing the dimensionality of the input data.
   - **Feature Scaling**: Normalizing feature values to help gradient descent converge faster and improve model performance.

3. **Cross-Validation**:
   - **Definition**: A technique for assessing how the results of a statistical analysis will generalize to an independent dataset, helping to ensure that the model does not overfit.

## Addressing Overfitting

### **Strategies to Address Overfitting**

1. **Collect More Training Data**:
   - **Description**: Increasing the amount of training data can help the model learn a more generalized function, reducing the impact of overfitting.
   - **Benefits**: With more data, even complex models (like high-order polynomials) can perform well, as they have more examples to learn from and are less likely to overfit.

2. **Feature Selection**:
   - **Description**: Reducing the number of features can help avoid overfitting by simplifying the model and focusing on the most relevant features.
   - **Approach**: Instead of using all available features, select a subset that is most relevant to the task. This reduces complexity and helps prevent the model from fitting noise in the data.
   - **Limitations**: Feature selection might discard useful information. Advanced techniques can automatically select features based on their relevance to the prediction task.

3. **Regularization**:
   - **Description**: Regularization is a technique that penalizes large coefficients in the model, effectively reducing the impact of less important features without completely eliminating them.
   - **Purpose**: By shrinking the values of parameters (but not setting them to zero), regularization helps the model avoid fitting the noise in the training data and generalize better to new examples.
   - **Implementation**: Regularization techniques include L1 regularization (Lasso), which can zero out some features, and L2 regularization (Ridge), which shrinks the coefficients but doesn’t set them to zero.

### **Key Points on Regularization**

- **Effectiveness**: Regularization can be particularly effective in high-dimensional settings where there are many features relative to the number of training examples.
- **Parameter Regularization**: Typically, only the model parameters (weights) are regularized, not the bias term, although regularizing the bias term is also possible but less common.

### **Summary**

- **Three Main Methods to Combat Overfitting**:
  1. **Collect More Data**: If feasible, this is the most straightforward solution.
  2. **Feature Selection**: Choose the most relevant features and avoid using excessive features that may introduce noise.
  3. **Regularization**: Apply regularization techniques to prevent the model from fitting noise by controlling the size of the parameters.


## Cost function with regularization

### **Regularization Overview**

**1. Regularization Concept:**
   - **Goal:** To penalize large parameter values to avoid overfitting by encouraging simpler models.
   - **Implementation:** Modify the cost function by adding a term that penalizes large parameter values.

**2. Modified Cost Function:**
   - **Original Cost Function:** Measures how well the model fits the training data, typically using mean squared error (MSE).
   - **Regularization Term:** Adds a penalty to the cost function based on the magnitude of the parameters. The modified cost function is:

     \[
     \text{Cost} = \text{Original Cost} + \lambda \frac{1}{2m} \sum_{j=1}^{n} w_j^2
     \]

     Here, \(\lambda\) is the regularization parameter, and \(m\) is the number of training examples.

**3. Regularization Parameter (\(\lambda\)):**
   - **Purpose:** Controls the trade-off between fitting the training data well (minimizing the original cost) and keeping the parameters small (minimizing the regularization term).
   - **Effect of \(\lambda\):**
     - **\(\lambda = 0\):** No regularization. The model may overfit, especially if it’s complex.
     - **Large \(\lambda\) (e.g., \(10^{10}\)):** Heavy penalty on parameters, causing them to be very small. This leads to underfitting, as the model may become too simplistic.
     - **Optimal \(\lambda\):** Balances the two goals, fitting the data well while keeping parameters from becoming too large.

**4. Regularization Term Details:**
   - **Scaling Factor:** Regularization term is scaled by \(\frac{1}{2m}\) to match the scaling of the original cost function, making it easier to choose an appropriate \(\lambda\) regardless of the number of training examples.
   - **Bias Term:** By convention, the bias term \(b\) is not regularized, although it makes little practical difference whether it is or not.

### **Application Example**

- **Housing Price Prediction:**
  - **Overfitting Example:** With \(\lambda = 0\), the model fits a high-order polynomial that overfits the training data.
  - **Underfitting Example:** With a very large \(\lambda\), the model fits a horizontal line (all parameters close to zero), leading to underfitting.
  - **Balanced Example:** Choosing an appropriate \(\lambda\) allows the model to fit a polynomial that balances complexity and generalization, leading to a better fit for the data.


##  Regularized linear regression

### **Gradient Descent with Regularization**

**1. Cost Function with Regularization:**
   - **Original Cost Function:** \[ J = \frac{1}{2m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2 \]
   - **Regularized Cost Function:** \[ J_{reg} = \frac{1}{2m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2 \]
   - Here, \(\lambda\) is the regularization parameter, \(m\) is the number of training examples, and \(n\) is the number of features.

**2. Gradient Descent Update Rules:**
   - **Without Regularization:**
     \[ w_j := w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) x_j^{(i)} \]
     \[ b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) \]
   - **With Regularization:**
     - Update for \(w_j\) (for \(j = 1\) to \(n\)):
       \[ w_j := w_j - \alpha \left(\frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j \right) \]
     - Update for \(b\):
       \[ b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) \]

**3. Explanation of Updates:**
   - **Regularization Term:** The term \(\frac{\lambda}{m} w_j\) is added to the gradient update for \(w_j\). This term shrinks \(w_j\) by a factor proportional to \(\lambda\) on each iteration.
   - **Update for \(b\):** The bias term \(b\) is not regularized, so its update rule remains unchanged.

**4. Intuitive Understanding of Regularization:**
   - **Effect on Parameters:** Regularization causes \(w_j\) to be multiplied by a factor slightly less than 1 (e.g., \(1 - \frac{\alpha \lambda}{m}\)). This shrinks the weights \(w_j\) slightly on each iteration, thereby controlling overfitting.
   - **Effect on Model:** Regularization discourages large values for parameters \(w_j\), leading to simpler models that generalize better.

**5. Derivation of Gradient Expressions (Optional):**
   - **Derivative with Respect to \(w_j\):**
     \[ \frac{\partial J_{reg}}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j \]
   - **Derivative with Respect to \(b\):**
     \[ \frac{\partial J_{reg}}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) \]
   - **Simplification:** The additional term \(\frac{\lambda}{m} w_j\) arises from the regularization term in the cost function.


## Regularized logistic regression

### **Regularized Logistic Regression**

**1. Regularization in Logistic Regression:**
   - **Original Cost Function for Logistic Regression:**
     \[ J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h(x^{(i)})) + (1 - y^{(i)}) \log(1 - h(x^{(i)})) \right] \]
     where \( h(x) = \frac{1}{1 + e^{-z}} \) and \( z = \mathbf{w}^T \mathbf{x} + b \).

   - **Regularized Cost Function:**
     \[ J_{reg}(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h(x^{(i)})) + (1 - y^{(i)}) \log(1 - h(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2 \]

   - **Effect of Regularization:** Adding the term \(\frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2\) penalizes large weights, preventing overfitting and leading to a more generalized decision boundary.

**2. Gradient Descent for Regularized Logistic Regression:**
   - **Gradient Descent Update Rules:**
     - **For \( w_j \) (for \( j = 1 \) to \( n \)):**
       \[ w_j := w_j - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j \right) \]
     - **For \( b \):**
       \[ b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) \]

   - **Similarities to Regularized Linear Regression:**
     - The gradient descent update for logistic regression with regularization looks similar to the update for linear regression. The main difference is that logistic regression uses the sigmoid function for \( h(x) \), whereas linear regression uses a linear function.

   - **Regularization Impact:**
     - The term \(\frac{\lambda}{m} w_j\) shrinks the weights \( w_j \) during each iteration, helping to prevent overfitting.
