# demo-mcds

This repository contains a demo of the Multivariate Correlated Data Synthesizer package in https://github.com/JdHondt/gcm.
The purpose of the demo is to illustrate how to use the MCDS package for the data generation based on a real-world dataset.
The demo is implemented in a Jupyter Notebook file named `demo_mcds.ipynb`.

The scenario is based on the Lending Club loan data, which is a popular dataset for credit risk modeling.
Sharing this kind of data can be prohibited due to privacy concerns.
However, using the GCM package, we can generate synthetic data that preserves the statistical properties of the original data while ensuring privacy.
We show that not only pairwise, but also higher-order correlations are preserved in the synthetic data.

For more information, we refer to the MCDS package documentation and the [paper](https://github.com/JdHondt/gcm/blob/master/whitepaper.pdf) that introduces the MCDS method.


## Requirements
The demo requires the following Python packages:
- gcm-syn (Multivariate Correlated Data Synthesizer)
- pandas
- numpy
- matplotlib
- seaborn

You can install these packages using pip. For example:

```bash
pip install gcm-syn pandas numpy matplotlib seaborn
```

## Download the demo data
You can download the demo data from the following link: [All lending club loan data: Accepted 2007 to 2018 Q4](https://www.kaggle.com/datasets/wordsforthewise/lending-club). It requires a Kaggle account to download the data.
In the demo, the 'accepted_2007_to_2018Q4.csv' file is used.
After downloading, unzip the file to a directory of your choice.

## Run the demo
To run the demo, navigate to the directory of the jupyter notebook file and execute the following command in your terminal:

```bash
jupyter notebook demo_gcm.ipynb
```
This will open the Jupyter Notebook interface in your web browser. Open the `demo_mcds.ipynb` file and follow the instructions in the notebook to run the demo.
Make sure to adjust the file path in the notebook to point to the location where you unzipped the data file.

