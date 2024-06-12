---
tags:
  - Machine_Learning
  - Scikit-Learn
  - todo
  - Classification
  - Regression
---
`Here are some vital rules I need to adhere to in ML workflows, in no particular order:`


### Always set `stratify = y` when doing `train_test_split` for classification problems

**Why?**
This is done to maintain a constant proportion of each of the classes in the target variable for both training and testing data, so that there isn't class imbalance in training and testing splits which could produce false accuracies during model training

### Use `permutation_importance` when assessing a model's dependence on certain features, instead of `feature_importance`

**why**? 

`permutation_importance` is a method for assessing the importance of features in a dataset by permuting the values of each feature and measuring the effect on the model’s performance. [It is a more robust method than `feature_importance` because it is less sensitive to the presence of correlated features and can detect non-linear relationships between features and the target variable](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html) [1](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html)[2](https://scikit-learn.org/stable/modules/permutation_importance.html).

[In contrast, `feature_importance` is based on the impurity reduction of each feature in a decision tree or random forest model and can be biased towards features with many categories or high cardinality](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html) [1](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html)[2](https://scikit-learn.org/stable/modules/permutation_importance.html).

Therefore, `permutation_importance` is generally a better choice when assessing a model’s reliance on a dataset’s features, especially when the features are highly correlated or have complex relationships with the target variable.

Here's a PDF illustrating how it's used:
[Permutation Importance - Feature Selection.pdf](file:///D:/Data/Stats%20&%20ML/Scikit-Learn/Permutation%20Importance%20-%20Feature%20Selection.pdf)


### Always use `StratifiedKFold()` when splitting classification data whether in `GridSearchCV` or otherwise manually
**why?**

StratifiedKFold is a cross-validation technique that preserves the proportion of each class in the data when splitting it into training and test sets. It is useful for imbalanced datasets or datasets with rare classes.

To see how it's done, check here: [[Using StratifiedKFold in an ML pipeline]]

### Optimize Linear Regression Models like `LinearRegression`, `Lasso` & `Ridge` for Polynomial relationships using the _`PolynomialFeatures()`_ class

In case your data has higher orders that cannot be accurately modelled by Linear Regression, you can use **Polynomial Linear Regression**
Here's a full read on how it's used: [[Polynomial Linear Regression]]
### For Linear Regression tasks, use `Ridge` or `Lasso` preferably, unless for very large datasets
This is because Lasso and Ridge introduce regularization by including a penalty term to minimize the residual sum of squares, which generally prevents overfitting.
For very large datasets however _(ex. datasets with 1M+ rows)_, the advantages of Lasso and Ridge are negligible
### Always use a baseline model before building a more complex Machine Learning Model

*Here's why:*

_In machine learning, a **baseline model** is a simple model that you create in a short amount of time. It’s your first attempt at modeling which provides a reference point for the performance of more complex models._

Baseline models can be simple stochastic models or they can be built on rule-based logic1. They are created using the same data and outcome variable that will be used to create your actual model.

The purpose of a baseline model is to help you understand your data faster1. By looking at the results of a baseline model, you can identify difficult to classify observations, different classes to classify, and low signal data.

In essence, a baseline model serves as a benchmark for other machine learning models. If your actual model performs better than the baseline, it indicates that your model is learning something useful from the data.

For example, in a regression problem, an ordinary least squares regression could serve as a baseline3. In classification problems, strategies like predicting the most frequent class or generating predictions uniformly at random can serve as baselines.

Visit this page to see how it's implemented: [[Baseline Models in Machine Learning]]


### Test Performances of Multiple models in one pipeline using a `Switcher()` class
To summarize,

Here is an easy way to optimize over any classifier and for each classifier any settings of parameters $\rightarrow$  [[Using a Switcher Class in SKLearn]]


## An automatic alternative to using `df.select_dtypes()` when doing selective feature engineering...

So I also never knew this before, but you can directly create a column_transformer and sieve out the numeric columns and categoricals, all at a go:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline


ct = make_column_transformer(
	(StandardScaler(),
	 make_column_selector(dtype_include = np.number)),
	 (OneHotEncoder(sparse = False), 
	 make_column_selector(dtype_include = object))
)
```

This is the alternative to a lengthier method:

```python
from sklearn.compose import make_pipeline

categoricals = df.select_dtypes(include = ['number']).columns
numerics = df.select_dtypes(include = ['category', 'object']).columns

ct = make_column_transformer(
	(StandardScaler(), numerics),
	(OneHotEncoder(sparse = False), categoricals)
)
```

And what about training a classifier?

Let's see how it's done:

```python
from Ipython.display import display
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector

from tensorflow import keras
from keras import layers, callbacks

ct = make_column_transformer(
	(StandardScaler(),
	 make_column_selector(dtype_include = 'numeric')),
	(OneHotEncoder(),
	 make_column_selector(dtype_include = ['category', 'object']))
)
pipe = Pipeline([
	('transformer', ct)
])
df_ = pipe.fit_transform(df)
```