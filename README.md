# Mapping Career Causeways

***Public repository for hosting the research outputs of the [Mapping Career Causeways](https://www.nesta.org.uk/project/mapping-career-causeways/) project.***

## Welcome!

Nesta is mapping viable pathways to fulfilling new occupations for people whose jobs are likely to change or be lost as a result of automation. We are calling these ‘career causeways’.

In this repository, you will find the research outputs of the report [Mapping Career Causeways: Supporting workers at risk](https://www.nesta.org.uk/report/mapping-career-causeways-supporting-workers-risk/) and open-source code for generating career transition recommendations.

*The project is part of Nesta's [Open Jobs](https://www.nesta.org.uk/project/open-jobs/) programme, and supported by the J.P.Morgan as part of the [New Skills At Work initiative](https://www.jpmorganchase.com/impact/our-approach/jobs-and-skills).*

*We would be grateful if anyone who downloads or uses our resources would complete this [2 minute survey](https://docs.google.com/forms/d/1IepcbAmIKAS2fDaDO4NfcI7uqNRkof02s52VmEYrZCY/edit?ts=6005b209&gxids=7628). Thank you!*

## Contents

### Codebase and job transitions data

`Published on January 20th, 2021`

- [**Project codebase**](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/): Python code for generating career transition recommendations and replicating analyses from the report
- [**Crowdsourced transition feasibility ratings**](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/reports/crowd_feasibility_ratings/): Results from our study to validate the transition recommendations
- [**Curated job transitions**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/transitions/transitions_tables/): A set of transitions recommended by our algorithm and validated using the crowdsourced feasibility ratings

### Supplementary online data of the project report

`Published on November 26th, 2020`

The supplementary data consists of several folders corresponding to the different parts of the Mapping Career Causeways project report:

- [**Crosswalk between O\*NET and ESCO occupations**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/ONET_ESCO_crosswalk/):
A mapping from the US O*NET framework to EU ESCO, together with the code used in developing the crosswalk.
- [**Automation risk estimates**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/automation_risk/):
Results related to task automation of 1627 ESCO occupations.
- [**Demographic analysis**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/demographic_analysis/):
Data on at-risk workers in the UK, Italy and France.
- [**Results on job transitions and skills analysis**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/transitions/): The number of safe and desirable transitions for 1627 ESCO occupations, and the effectiveness of core skills in uncovering new transitions.
- [**Skills-based sectors**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/skills_based_sectors): Breakdown of data-driven groups of related ESCO occupations that share similar job requirements and work characteristics

## Installation

It is possible to browse and download the [supplementary online data](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data), or try out [cloud-based tutorials](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase#tutorials-on-google-colab) for exploring career transition recommendations, without installing anything on your local machine. If you'd like to run any of the analysis code on your local machine, please follow the steps below.

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
### Download input data

As the final step, you'll have to download the input data necessary to run the analysis and inspect the career transitions. This can be easily done by running the following command in the terminal, which will synchronise your local data directory with the copy on our S3 cloud storage:

```
make sync_data_from_s3
```

This will set up the required input files into the `data/raw`, `data/interim` and `data/processed` folders (except for very large intermediate analysis outputs that can be downloaded manually if needed).

In case you run into some issues with the AWS command line interface, you might alternatively try the commands below. This will
download and unzip an archived version of the input data. However, this archive might not always be up to date and, therefore, we recommend using the first approach.

```
$ cd codebase/data
$ python download_data.py
```

## Directory structure

The repository has the following main directories:

```
  ├── mapping_career_causeways   <- Packaged code (various modules and utilities)
  ├── codebase                   <- Analysis codebase (notebooks) and data
  │   ├── data
  │   │   ├── raw                <- The original, immutable data
  │   │   ├── interim            <- Intermediate output data that has been transformed
  │   │   ├── processed          <- Final, processed data sets for analysis
  │   │   │   ├── linked_data    <- Various labour market data linked to ESCO occupations
  │   │   │   ├── sim_matrices   <- Various similarity matrices between all ESCO occupations,
  │   │   │   │                     for analysing career transition viability
  │   │   │   ├── validation     <- Crowdsourced transition feasibility data
  │   │   │   ...    
  │   ├── notebooks              <- Data analysis and generation of career transition recommendations
  │   └── reports                <- Reports, briefs and figures associated with the project
  └── supplementary_online_data  
      ├── ONET_ESCO_crosswalk    <- Crosswalk between O*NET and ESCO occupations     
      ├── automation_risk        <- Automation risk estimates for ESCO occupations
      ├── transitions            <- Results on career transitions
      │   └── transitions_tables <- Curated set of transitions recommended by our algorithm and
      ...                           validated using the crowdsourced feasibility ratings                                    
```

<a name="feedback"></a>
## Feedback & contributions

### Get in touch!
If you find this work interesting and useful, we would be very grateful if you [fill out this short survey](https://docs.google.com/forms/d/1IepcbAmIKAS2fDaDO4NfcI7uqNRkof02s52VmEYrZCY/edit?ts=6005b209&gxids=7628) to tell us about yourself and your interest in this project. As a bonus, we will keep you up to date with future project updates as we undertake further research on validating the career transition recommendations, and release practical user guides for employers, policy makers and employment services.

Anyone is welcome to use and build upon the published data and code. As more people try out the career transitions algorithm, we expect to uncover peculiarities and areas for improvement - if you would like to leave a technical comment or report a bug, please create a new github issue. For more general enquiries, [write to us](mailto:open.jobs@nesta.org.uk).

### Contribute
If you would like to contribute to this framework, you can collaborate by using pull requests:
1. Create your new branch from dev
2. Open a new issue and describe the feature that you will contribute
3. Add your feature
4. Issue a pull request

## Citing
To refer to the work on automation risk, career transitions algorithm or the crosswalk between O\*NET and ESCO, please cite the report:

*Kanders K., Djumalieva, J., Sleeman, C. and Orlik, J. (2020). Mapping Career Causeways: Supporting workers at risk. London: Nesta*

To specifically cite the repository, or the crowdsourced feasibility data, please use the repo's digital object identifier (DOI) that is linked to [Zenodo](https://zenodo.org/badge/latestdoi/307661339):

*Karlis Kanders, & George Richardson. (2021, January 20). nestauk/mapping-career-causeways: v2.0.0 (January 2021) (Version v2.0.0). Zenodo. http://doi.org/10.5281/zenodo.4451887*

[![DOI](https://zenodo.org/badge/307661339.svg)](https://zenodo.org/badge/latestdoi/307661339)
