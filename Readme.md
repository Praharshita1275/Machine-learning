# Enhancing Multi-Class Classification Accuracy Using Support Vector Machines (SVM) with Hyperparameter Tuning

This project demonstrates the application and optimization of the Support Vector Machine (SVM) algorithm for multi-class classification on the classic Iris dataset. It is based on the foundational paper "Support-Vector Networks" by Cortes & Vapnik (1995). The primary goal is to show how hyperparameter tuning can significantly improve the performance of an SVM model, moving from a strong baseline to perfect classification accuracy on the test set.

---

## Project Overview

Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification and regression. Introduced by Cortes and Vapnik, SVM functions by identifying an optimal hyperplane that creates the largest possible margin between different classes in a feature space. While effective, its performance is highly sensitive to the choice of kernel and hyperparameters like `C` and `γ`.

This project addresses this sensitivity by implementing a baseline SVM and then using `GridSearchCV` to find the optimal hyperparameters, thereby resolving initial misclassifications and boosting performance metrics.

### Project Objectives

* **Implement a baseline SVM classifier** on the Iris dataset.  
* **Identify research gaps**, particularly hyperparameter sensitivity.  
* **Perform hyperparameter tuning** using `GridSearchCV` to enhance accuracy.  
* **Evaluate and compare** the baseline and tuned models using metrics like accuracy, precision, recall, and F1-score.  
* **Deliver a complete project** with source code, a detailed report, and visualizations.

---

## Dataset

The project utilizes the **Iris flower dataset**, a classic benchmark for multi-class classification.

* **Samples**: 150  
* **Features (4)**: Sepal Length, Sepal Width, Petal Length, Petal Width  
* **Target Classes (3)**: Iris Setosa, Iris Versicolor, Iris Virginica  

---

## Technologies & Tools

| Category | Technology/Tool | Purpose |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.10+ | Core project implementation |
| **Libraries** | scikit-learn | SVM, GridSearchCV, metrics |
| | numpy | Numerical data handling |
| | matplotlib, seaborn | Data visualization |
| **Algorithm** | SVM (SVC) | Classification model |
| **Environment** | Jupyter Notebook/VS Code | Development and analysis |
| **Version Control** | GitHub | Code repository and documentation |

---

## Methodology

The project follows a standard machine learning workflow:

1. **Data Loading & Preparation**: The Iris dataset is loaded and split into training and testing sets.  
2. **Baseline Model Training**: An SVM classifier is trained using scikit-learn's default hyperparameters (`kernel='rbf'`).  
3. **Hyperparameter Tuning**: `GridSearchCV` is used to systematically search for the best combination of `C`, `gamma`, `kernel`, and `degree` to optimize the model.  
4. **Tuned Model Training**: A new SVM model is trained using the best parameters discovered by the grid search.  
5. **Evaluation & Comparison**: Both the baseline and tuned models are evaluated on the test set. Performance is measured using accuracy, precision, recall, F1-score, and a confusion matrix.

---

## Results

The hyperparameter tuning process resulted in a significant improvement in classification performance.

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline SVM** | 0.96 | 0.96 | 0.96 | 0.96 |
| **Tuned SVM (Grid)** | **1.00** | **1.00** | **1.00** | **1.00** |

* **Baseline Model Accuracy**: ~96%, with minor misclassifications between *Versicolor* and *Virginica*.  
* **Tuned Model Accuracy**: **100%**, successfully resolving all misclassifications.

### Optimal Hyperparameters

The best parameters found by `GridSearchCV` were:
* `C`: 10  
* `gamma`: 0.1  
* `kernel`: 'rbf'  

### Tuned Model Confusion Matrix

| Actual \ Predicted | Setosa | Versicolor | Virginica |
| :---: | :---: | :---: | :---: |
| **Setosa** | 10 | 0 | 0 |
| **Versicolor** | 0 | 9 | 0 |
| **Virginica** | 0 | 0 | 11 |

---

## Learning Outcomes

* Gained a practical understanding of multi-class SVM implementation.  
* Learned the critical impact of kernel selection and hyperparameter tuning on model performance.  
* Strengthened Python skills for machine learning, including the use of scikit-learn, matplotlib, and seaborn.  
* Completed an end-to-end machine learning workflow from data handling to evaluation and visualization.

---

## Future Work

* **Scalability Testing**: Apply the tuned model to larger, more complex datasets.  
* **Interpretability**: Use feature selection techniques to better understand feature importance.  
* **Comparative Analysis**: Benchmark the SVM's performance against other classifiers like Random Forest and XGBoost.  

---

## Citation

Cortes, C., & Vapnik, V. (1995). *Support-Vector Networks.* *Machine Learning, 20*(3), 273–297.
