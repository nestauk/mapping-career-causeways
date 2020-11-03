# Mapping Career Causeways

`UNDER CONSTRUCTION - COME BACK ON NOVEMBER 23RD, 2020`

*Public repository for hosting the research outputs of the [Mapping Career Causeways](https://www.nesta.org.uk/project/mapping-career-causeways/) project.*

*Nesta is mapping viable pathways to fulfilling new occupations for people whose jobs are likely to change or be lost as a result of automation. We are calling these ‘career causeways’.*

*Please find the associated project report, and explore a data visualisation of job transition recommendations and a map of occupations [here](https://www.nesta.org.uk/project/mapping-career-causeways/).*

*The project is carried out within Nesta's [Open Jobs](https://www.nesta.org.uk/project/open-jobs/) programme, and it is supported by the J.P.Morgan as part of the [New Skills At Work initiative](https://www.jpmorganchase.com/impact/our-approach/jobs-and-skills).*

## Introduction

The research outputs will be published in two steps. First, we are releasing [Supplementary online data](https://github.com/nestauk/mapping-career-causeways/tree/main/Supplementary_online_data) together with the report on November 23rd, 2020. This mainly contains additional tables supporting the results published in the report. 

Project codebase will be published in early 2021, together with a complete list of the recommended career transitions for all considered ESCO occupations. This will allow anyone to replicate and build upon the project outputs.

In the meantime, we will be conducting research to further validate the recommendations from the career transitions algorithm.

## Contents

### Supplementary online data

`STATUS: Not yet released; to be published on November 23rd, 2020`

- [Crosswalk between O\*NET and ESCO occupations](https://github.com/nestauk/mapping-career-causeways/tree/main/Supplementary_online_data/ONET_ESCO_crosswalk/)
- [Automation risk estimates of ESCO occupations](https://github.com/nestauk/mapping-career-causeways/tree/main/Supplementary_online_data/Automation_risk/)
- [Demographic analysis of at-risk workers](https://github.com/nestauk/mapping-career-causeways/tree/main/Supplementary_online_data/Demographic_analysis/)
- [Results on transition options for at-risk workers](https://github.com/nestauk/mapping-career-causeways/tree/main/Supplementary_online_data/Transitions/)

### Codebase and job transitions

`STATUS: Not yet released; to be published in early 2021`

- Code for automation risk analysis
- Career transition recommendation algorithm
- Complete list of job transitions for all considered ESCO occupations

## Installation

It is possible to browse and download the [Supplementary online data](https://github.com/nestauk/mapping-career-causeways/tree/main/Supplementary_online_data) without installing anything on your local machine. However, if you'd like to run any analysis code, you should follow the steps below. *Note that the analysis code will be added to this repo in early 2021; presently, you only need to do this if you'd like to run the Jupyter notebook used in developing the O\*NET-ESCO crosswalk.*

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

You should also install the repository package:

```shell
$ pip install -e .
```
