# dataiku

## Getting Started

### Install Dependencies

To install the required Python packages, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

It's recommended to use a virtual environment to avoid package conflicts:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Model

Once dependencies are installed and the virtual environment is activated, you can run the full pipeline by executing:

```bash
python main.py
```

This will:
- Load and preprocess the data
- Train and evaluate multiple models
- Apply fairness-aware modeling
- Save the best model and performance metrics in the `model_output` folder

## Assignment Overview

This repository contains my submission for the Dataiku Data Scientist Technical Assessment and Presentation.

### Background

The assignment focuses on analyzing anonymized data from the US Census archive (~300,000 individuals). The goal is to identify characteristics that influence whether a person earns more or less than $50,000 annually.

### Provided Data

- `census_income_learn.csv`: Training dataset
- `census_income_test.csv`: Testing dataset
- `census_income_metadata.txt`: Metadata describing the datasets
- `census_income_additional_info.pdf`: Supplemental data context

### Objective

Construct a robust data analysis and modeling pipeline using Python. The key tasks include:
- Exploratory Data Analysis (EDA)
- Data Preparation (cleaning, preprocessing, feature engineering)
- Building multiple classification models
- Evaluating and comparing model performance
- Summarizing findings, insights, and future recommendations

### Presentation

The project also requires preparing a 20-minute presentation to a mock customer (played by Dataiku personnel), with a focus on clear communication to both technical and non-technical audiences.

### Submission Includes

- Well-documented and reproducible code
- Results and visualizations
- Slide deck summarizing methodology and insights

📁 This repository was created as part of the Dataiku Data Scientist Technical Assessment.
