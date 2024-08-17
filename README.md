# Introduction to Machine Learning Course Repository

This repository contains materials for an Introduction to Machine Learning course. It includes Jupyter notebooks, exercises, and datasets designed to provide a comprehensive understanding of various machine learning models and techniques.

## Repository Structure

The repository is organized into several directories, each focusing on different types of machine learning problems and topics:

### 1. `classification-problems/`

This directory contains projects related to various classification problems, including:

- **breast-cancer-classification/**: A project on classifying breast cancer cases.
- **income-classification/**: Contains the dataset (`income_dataset.csv`) and notebook (`ex1.ipynb`) for classifying income levels.
- **iris-classification/**: Focuses on the famous Iris dataset classification problem.

### 2. `regression-problems/`

This directory focuses on regression problems:

- **diabetes-regression/**: A project on predicting diabetes progression.
- **flue-consumption-regression/**: Contains notebooks and datasets for predicting flu consumption.

### 3. `theory-and-implementation/`

This directory includes theoretical notebooks and implementations of various machine learning algorithms:

- `1-perceptron.ipynb`: Implementation of the Perceptron algorithm.
- `2-adaline.ipynb`: Implementation of the Adaline algorithm.
- `3-linear-regression.ipynb`: Introduction to linear regression.
- `4-polynomial-regression.ipynb`: Polynomial regression.
- `5-logistic-regression.ipynb`: Logistic regression.
- `introduction_to_machine_learning.pdf`: A PDF document introducing the basics of machine learning.

## Other Files

- **README.md**: This file.
- **requirements.txt**: Lists the Python dependencies required to run the notebooks.
- **shell.nix**: Nix shell configuration for setting up the environment.

## Getting Started

To get started with the notebooks, you will need to install the required dependencies. This can be done using the following command:

```bash
pip install -r requirements.txt
```

If you're using Nix, you can set up your environment with:

```bash
nix-shell
```

## How to Use

1. Navigate to the directory of interest (e.g., `classification-problems/iris-classification/`).
2. Open the Jupyter notebook (`.ipynb` file) using JupyterLab or Jupyter Notebook.
3. Follow the instructions and run the cells to explore the different machine learning models and techniques.

## Contributing

This project is only for educational purposes, and I'm not accepting contributions.
