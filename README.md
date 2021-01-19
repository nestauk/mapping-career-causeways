# Mapping Career Causeways

***Public repository for hosting the research outputs of the [Mapping Career Causeways](https://www.nesta.org.uk/project/mapping-career-causeways/) project.***

## Welcome!

Nesta is mapping viable pathways to fulfilling new occupations for people whose jobs are likely to change or be lost as a result of automation. We are calling these ‘career causeways’.

In this repository, you will find the research outputs of the report [Mapping Career Causeways: Supporting workers at risk](https://www.nesta.org.uk/report/mapping-career-causeways-supporting-workers-risk/). Note that we will release the outputs gradually, with the first batch in November, followed by a more complete release in early 2021. Please see the contents sections below for more information.

*The project is part of Nesta's [Open Jobs](https://www.nesta.org.uk/project/open-jobs/) programme, and supported by the J.P.Morgan as part of the [New Skills At Work initiative](https://www.jpmorganchase.com/impact/our-approach/jobs-and-skills).*

## Contents

### Supplementary online data of the project report

`STATUS: Published on November 26th, 2020`

The supplementary data consists of several folders corresponding to the different parts of the Mapping Career Causeways project report:

- [**Crosswalk between O\*NET and ESCO occupations**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/ONET_ESCO_crosswalk/):
A mapping from the US O*NET framework to EU ESCO, together with the code used in developing the crosswalk.
- [**Automation risk estimates**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/automation_risk/):
Results related to task automation of 1627 ESCO occupations.
- [**Demographic analysis**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/demographic_analysis/):
Data on at-risk workers in the UK, Italy and France.
- [**Results on job transitions and skills analysis**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/transitions/): The number of safe and desirable transitions for 1627 ESCO occupations, and the effectiveness of core skills in uncovering new transitions.
- [**Skills-based sectors**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/skills_based_sectors): Breakdown of data-driven groups of related ESCO occupations that share similar job requirements and work characteristics

### Codebase and full job transitions data

`STATUS:  Published on January 20th, 2020`

- [**Project codebase together with a complete list of the recommended career transitions for all considered ESCO occupations. This open-source code will allow anyone to generate transition recommendations, replicate the results and build upon the project outputs.

In the meantime, we will be conducting research to further validate the recommendations from the career transitions algorithm.

## Installation

It is possible to browse and download the [supplementary online data](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data) without installing anything on your local machine. However, if you'd like to run any of the analysis code, you should follow the steps below. *Note that the complete analysis code will be published in early 2021.*

### Clone and set up the repo

To clone the repo and install the dependencies, navigate to a convenient directory and run the following commands from the terminal:

```shell
$ git clone https://github.com/nestauk/mapping-career-causeways
$ cd mapping-career-causeways
```

It should be possible to set up the environment using the specified configuration file

```shell
$ conda env create -f conda_environment.yaml
```
You should then activate the newly created conda environment and install the repository package:

```shell
$ conda activate mapping_career_causeways
$ pip install -e .
```

## Directory structure

The repository has the following main directories:

```
  ├── mapping_career_causeways  <- Packaged code (various modules and utilities)
  ├── codebase                  <- Analysis code (notebooks) and data
  │   ├── data
  │   │   ├── raw               <- The original, immutable data
  │   │   ├── interim           <- Intermediate output data that has been transformed
  │   │   ├── processed         <- Final, processed data sets for analysis
  │   │   │   ├── linked_data   <- Various labour market data linked to ESCO occupations
  │   │   │   ├── sim_matrices  <- Various similarity matrices between all ESCO occupations,
  │   │   │   │                    for analysing career transition viability
  │   │   │   ├── validation    <- Crowdsourced transition feasibility data
  │   │   │   ...    
  │   ├── notebooks             <- Data analysis and generation of career transition recommendations
  │   └── reports               <- Reports, briefs and figures associated with the project
  └── supplementary_online_data
      ├── ONET_ESCO_crosswalk   <- Crosswalk between O*NET and ESCO occupations
      ├── transitions           <- Curated data on transitions between ESCO occupations
      ...         
```

## Feedback & contributions

Anyone is welcome to use and build upon the published data and code. As more people try out the career transitions algorithm, we expect to uncover peculiarities and areas for improvement - if you would like to leave feedback or report a bug, please create a new github issue. For more general enquiries, [write to us](mailto:open.jobs@nesta.org.uk).

If you would like to contribute to this framework (e.g. by working on of the features listed above) you can collaborate by using pull requests:
1. Create your new branch from dev
2. Open a new issue and describe the feature that you will contribute
3. Add your feature
4. Issue a pull request

## Citing
To refer to the work, please cite the Mapping Career Causeways report:

*Kanders K., Djumalieva, J., Sleeman, C. and Orlik, J. (2020). Mapping Career Causeways: Supporting workers at risk. London: Nesta*

To specifically cite this github repository, you can use the following digital object identifier (DOI), which
is linked to [this Zenodo repository](#).
