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

This will set up all required input files into the `data/raw`, `data/interim` and `data/processed` folders. See the readme files in these folders to find more information about specific files.

## Directory structure

The codebase has the following main directories:

```
  ├── codebase
  │   ├── data
  │   │   ├── raw               <- The original, immutable data
  │   │   ├── interim           <- Intermediate output data that has been transformed
  │   │   ├── processed         <- Final, processed data sets for analysis
  │   │   │   ├── linked_data   <- Various labour market data linked to ESCO occupations
  │   │   │   ├── sim_matrices  <- Various similarity matrices between all ESCO occupations,
  │   │   │   │                    for analysing career transition viability
  │   │   │   ...    
  │   ├── notebooks             <- Data analysis and generation of career transition recommendations
  │   └── reports               <- Reports, briefs and figures associated with the project
  ├── supplementary_online_data
  │   ├── ONET_ESCO_crosswalk   <- Crosswalk between O*NET and ESCO occupations
  │   ...    
  └── mapping_career_causeways  <- Custom packages      
```

Note that the code and data for generating the crosswalk between the O*NET and ESCO occupations has been factored out in the Supplementary online data section of the repository.

In addition, the `mapping_career_causeways` folder contains additional custom packages and utilities that were developed by us for performing NLP-adjusted overlap, consensus clustering and other functions.

## Using the codebase

### Replicating results of the project report
Data analyses underpinning the project [report](https://www.nesta.org.uk/report/mapping-career-causeways-supporting-workers-risk/) are presently available as a set of Jupyter notebooks in the `notebooks/` folder. These are organised into several sections pertaining to different parts of the project. The notebooks clearly explain each step of the analysis, and produce output files that are stored in the `data/` and used in further downstream analyses.

### Exploring validated career transition recommendations
We have validated a set of job transitions by assessing people's perceptions of their feasibility. More information about the validation study, and the set of validated transitions is available [here](#).

### Generating your own career transition recommendations
For generating your own career transition recommendations, subject to your own transition viability and desirability parameters, you can use the [tutorial notebook](#), as well as the collection of utility functions defined in `../mapping_career_causeways/transitions_utils.py`.

## Future steps

We hope to further develop the codebase to add support for the following functions (subject to our capacity):
- Further improve transition viability assessment by using data from the validation study
- Check for new updates of external datasets and easily rerun the whole analysis pipeline
- Explore the possibility for creating a local API for generating transition recommendations
- Further enrich the occupational profiles with useful features: for example, timely data on local occupational vacancies or employment growth prospects; check here for the full ["wishlist"](#).

## Feedback & Contributions

Anyone is welcome to use and build upon the published data and code. If you would like to leave feedback or report a bug, please create a new issue. If you would like to contribute to this framework (e.g. by working on of the features listed above) you can collaborate by using pull requests:
1. Create your new branch from dev
2. Open a new issue and describe the feature that you will contribute
3. Add your feature
4. Issue a pull request
