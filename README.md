# Fintech_Module14

![algo_trading.jpg](https://github.com/nielsdehaan1977/Fintech_Module14/blob/main/Images/algo_trading.jpg)
---
# Algorithmic Trading 
---
## This jupyter notebook can be used as a template to create and test algorithmic trading strategies. 

## machine_learning_trading_bot.ipynb
---
### This notebook can be used as a template to build a algorithmic trading model. The notebook starts off with creating a baseline perfomance and shows examples how to tune the baseline trading algorithm. And lastly it also evaluates a different machine learning classifier. 
---
The tool can help to create and to test algorithmic trading strategies.
* The tool goes through on the following steps: 
1. Establish a Baseline Performance
2. Tune the Baseline Trading Algorithm
3. Evaluate a new machine learning classifier
4. Create an evaluation report
---
## Table of Content

- [Tech](#technologies)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [Contributor(s)](#contributor(s))
- [License(s)](#license(s))

---
## Tech

This project leverages python 3.9 and Jupyter Lab with the following packages:

* `Python 3.9`
* `Jupyter lab`

* [JupyterLab](https://jupyter.org/) - Jupyter Lab is the latest web-based interactive development environment for notebooks, code, and data.

* [Path](https://docs.python.org/3/library/pathlib.html) - This module offers classes representing filesystem paths with semantics appropriate for different operating systems.

* [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) - Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

* [numpy](https://numpy.org/doc/stable/index.html) - NumPy is the fundamental package for scientific computing in Python.

* [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) - Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).

* [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) - SVC is a class capable of performing binary and multi-class classification on a dataset.

* [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) - Ordinary least squares Linear Regression.

* [hvplot](https://hvplot.holoviz.org/) - A familiar and high-level API for data exploration and visualization

* [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) - Build a text report showing the main classification metrics.
---

## Installation Guide

### Before running the application first install the following dependencies in either Gitbash or Terminal. (If not already installed)

#### Step1: Activate dev environment in Gitbash or Terminal to do so type:
```python
    conda activate dev
```
#### Step2: install the following libraries (if not installed yet) by typing:
```python
    pip install pandas
    pip install numpy
    pip install -U scikit-learn

 ```
#### Step3: Start Jupyter Lab
Jupyter Lab can be started by:
1. Activate your developer environment in Terminal or Git Bash (already done in step 1)
2. Type "jupyter lab --ContentsManager.allow_hidden=True" press enter (This will open Jupyter Lab in a mode where you can also see hidden files)

![JupyterLab](https://github.com/nielsdehaan1977/Fintech_Module13/blob/main/Images/JupyterLab.PNG)


## Usage

To use the venture funding with deep learning jupyter lab notebook, simply clone the full repository and open the **venture_funding_with_deep_learning.ipynb** file in Jupyter Lab. 

The tool will go through the following steps:

### Establish a Baseline Perfomance
* import of data to analyze
* generate trading signals using short- and long-window SMA values
* split the data into training and testing datasets
* review classification report associated with the SVC model predictions
* create a predictions dataframe
* create a cumulative return plot that shows the actual returns vs the strategy returns

### Tune the baseline Trading Algorithm
* tune the training algorithm by adjusting the size of the training dataset
* tune the trading algorithm by adjusting the SMA input features.

### Evaluate a new machine learning classifier
* import a new classifier
* using the original training data as the baseline model, fit another model with the new classifier.
* backtest the new model to evaluate its performance.

## Contributor(s)

This project was created by Niels de Haan (nlsdhn@gmail.com)

---

## License(s)

MIT

---
# Evalution Report
---

## Baseline Performance

The baseline strategy performance was not impressive with a loss of about 30% based on historical data used. Strategy was based on generatinge trading signals using short- and long-window SMA values. Baseline SMA values were 4 periods for the short sma and 100 periods for the long sma. Strategy signals were set to 1 if return was equal or larger than 0 and -1 if return was smaller than 0. 

![Baseline_Strategy_returns.jpg](https://github.com/nielsdehaan1977/Fintech_Module14/blob/main/Images/Baseline_Strategy_returns.jpg)

---

## Tune Baseline Trading Algorithm

While trying to improve the results of the baseline strategy, we adjusted the model's input features to find the parameters that result in better trading outcomes.
The following parameters were changed and the results are below:

The SVM model with 3M of training data and SMA_short at 4 periods and SMA_long at 100 periods for the most part of 2019 and 2020 performed better than the actual returns. Also when changing the amount of training data to 6M the strategy ultimately outperforms the actual returns. For the model with 6M training data the actual returns outperform the strategy returns for most of 2019, but outperforms the actual returns from March 2020 onwards. With a total return of about 80% whereas the model with 3M training data "only" produces about 50% return. 

1. adjusting the training window from 3 months to 6 months

While changing the training window from 3m to 6m:
* accuracy of the model increased by 1 point from  0.55 to 0.56. 
* Precision for -1 increased by 1 point from 0.43 to 0.44
* Recall for -1 decreased from 0.04 to 0.02

---
![Image_of_3m_class](https://github.com/nielsdehaan1977/Fintech_Module14/blob/main/Images/svm_class_report_3m_training_data_SMAFast_4_SMASlow_100.jpg)  ![Image_of_6m_class](https://github.com/nielsdehaan1977/Fintech_Module14/blob/main/Images/svm_class_report_6m_training_data_SMAFast_4_SMASlow_100.jpg)

![Image_of_3m_training](https://github.com/nielsdehaan1977/Fintech_Module14/blob/main/Images/svm_plot_3M_training_and_SMAFast_4_SMASlow_100.png)

![Image_of_6m_training](https://github.com/nielsdehaan1977/Fintech_Module14/blob/main/Images/svm_plot_6M_training_and_SMAFast_4_SMASlow_100.png)

---
2. adjusting the SMA input features 
---
2a. adjusting SMA short to 2 and SMA long to 50

While changing the SMA window to 2 and 50:
* accuracy of the model decreased by 1 point from  0.55 to 0.54. 
* Precision for -1 decreased by 4 points from 0.43 to 0.39
* Recall for -1 increased from 0.04 to 0.07

![Image_of_3m_class_2_50](https://github.com/nielsdehaan1977/Fintech_Module14/blob/main/Images/svm_class_report_3m_training_data_SMAFast_2_SMASlow_50.jpg)
![Image_of_3m_SMAFast_2_SMASlow_50](https://github.com/nielsdehaan1977/Fintech_Module14/blob/main/Images/svm_plot_3M_training_and_SMAFast_2_SMASlow_50.png)

---
2b. adjusting SMA short to 1 and SMA long to 25

While changing the SMA window to 1 and 25:
* accuracy of the model decreased by 1 point from  0.55 to 0.54. 
* Precision for -1 decreased by 1 points from 0.43 to 0.42
* Recall for -1 increased from 0.04 to 0.13

![Image_of_3m_class_1_25](https://github.com/nielsdehaan1977/Fintech_Module14/blob/main/Images/svm_class_report_3m_training_data_SMAFast_1_SMASlow_25.jpg)
![Image_of_3m_SMAFast_1_SMASlow_25](https://github.com/nielsdehaan1977/Fintech_Module14/blob/main/Images/svm_plot_3M_training_and_SMAFast_1_SMASlow_25.png)

---

## Evaluate a New Machine Learning Classifier

For the evaluation of how a different machine learning classifier would perform compared to the provided baseline model, the logistic regression model was used for the comparison. and the model was ran with 3M of training data and SMA_short of 1 period and SMA_long of 25

While changing the SVM model to Logistic Regression Model (SMA 1 and 25):
* accuracy of the model was the same at 0.54. 
* Precision for -1 increased by 1 points from 0.42 to 0.43
* Recall for -1 increased from 0.13 to 0.16

![lr_class_report_3m_training_data_SMAFast_1_SMASlow_25.jpg](https://github.com/nielsdehaan1977/Fintech_Module14/blob/main/Images/lr_class_report_3m_training_data_SMAFast_1_SMASlow_25.jpg)
![lr_plot_3M_training_and_SMAFast_1_SMASlow_25.png](https://github.com/nielsdehaan1977/Fintech_Module14/blob/main/Images/lr_plot_3M_training_and_SMAFast_1_SMASlow_25.png)

