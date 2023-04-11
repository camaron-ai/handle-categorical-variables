# Handeling Categorical Variables

Welcome! Our goal is to provide a comprehensive analysis of various approaches to handling categorical variables in machine learning.

So far, we have analyzed the following techniques:
- Label Encoding
- One Hot Encoding
- Frequency Encoding

In addition to these methods, we plan to implement and compare several other approaches in the future.

Our hope is that this repository will serve as a valuable resource for anyone looking to improve their machine learning models by handling categorical variables more effectively.


## Dataset
our experiment use the [Porto Seguro Sage Driver Dataset](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction).

download the dataset from Kaggle and unzip it into the data/raw/ folder.
Instructions

### Step 1: Set up Environment
To get started, you'll need to install the necessary dependencies via Anaconda. We recommend using mamba, which is faster than using conda. Follow these steps to set up the environment:

Install mamba: 
```bash
conda install mamba
```

Use the environment.yml file to build the environment: 
```bash
mamba env create --file=environment.yml
```

Activate the environment: 
```bash
conda activate hndl-cat
```


### Step 2: Add PYTHONPATH
To allow us to use the src/ directory as a package, we need to add the root of the repository to the Python path. Here's how to do it:

Export the PYTHONPATH variable: 
```bash
export PYTHONPATH=${PYTHONPATH}:${PWD}
```

### Step 3: Set up Data
Before we can process categorical variables, we need to scale continuous variables from 0 to 1 and write the training/validation indices to disk. Here's how to do it:

Run the following command: 
```bash
make setup-data
```

### Step 4: Process Categorical Variables
To process categorical variables, run the following command:
```bash
make process-cat
```

This command will perform all of the transformations discussed in the blog post. Once it finishes running, the categorical variables in the data will be ready to use in your machine learning models.

### Step 5: Run Experiments
To run all experiments, run the following command:
```bash
python run_all_experiment.py
```

That's it! Once you have run the experiments, you can see the results on result-analysis.ipynb and further analyze each approach. Have fun!

If you have any questions or issues, please contact us for support.