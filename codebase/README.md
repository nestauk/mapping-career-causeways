# Mapping Career Causeways codebase

***Public codebase featuring the career transitions algorithm and data analysis of the [Mapping Career Causeways](https://www.nesta.org.uk/project/mapping-career-causeways/) project.***

## Welcome to the codebase!

The code in this repository will allow you to replicate the results of our [research report](https://www.nesta.org.uk/report/mapping-career-causeways-supporting-workers-risk/), generate career transition recommendations, and build upon the project outputs.

## Getting started

If you only wish to generate job transition recommendations and inspect skills gaps, you can use the interactive [Google Colab notebook](#). For that, you don't have to install or download anything - simply follow the steps in the notebook.

If you wish to run the analysis on your local machine or build upon the codebase, make sure you have followed the [installation instructions](https://github.com/nestauk/mapping-career-causeways/#installation). Then, you'll have to download the input data necessary to run the analysis and inspect the career transitions from our Amazon S3. This can be easily done by running the following commands in the terminal:

```
$ cd data
$ python download_data.py
```

This will set up all required input files into the `data/raw`, `data/interim` and `data/processed` folders. See the readme files in these folders to find more information about specific data.

Note that the code and data for generating the crosswalk between the O*NET and ESCO occupations has been factored out in the Supplementary online data section of the repository.

In addition, the `mapping_career_causeways` folder contains additional custom packages and utilities that were developed by us for performing NLP-adjusted overlap, consensus clustering and other functions.

## Using the codebase

### Replicating results
Data analyses underpinning the project [report](https://www.nesta.org.uk/report/mapping-career-causeways-supporting-workers-risk/) are presently available as a set of Jupyter notebooks in the `notebooks/` folder. These are organised into several sections pertaining to different parts of the project. The notebooks clearly explain each step of the analysis, and produce output files that are stored in the `data/` and used in further downstream analyses.

### Exploring transition recommendations
We have created a validated set of job transitions by taking into account people's perceptions of their feasibility. More information about the validation study can be found [in the technical appendix](#), and the set of validated transitions is available [here](#).

### Generating career transitions
For generating your own career transition recommendations, inspecting skills gaps, and identifying transferable skills - subject to your own transition viability and desirability parameters - you can use the [tutorial notebook](#), as well as the collection of utility functions defined in `../mapping_career_causeways/transitions_utils.py`.
