# Mapping Career Causeways

***Public repository for hosting the research outputs of the [Mapping Career Causeways](https://www.nesta.org.uk/project/mapping-career-causeways/) project.***

## Welcome!

Nesta is mapping viable pathways to fulfilling new occupations for people whose jobs are likely to change or be lost as a result of automation. We are calling these ‘career causeways’.

In this repository, you will find the research outputs (including data and eventually also the full analysis code) of the report [Mapping Career Causeways: Supporting Workers at Risk](https://www.nesta.org.uk/project/mapping-career-causeways/). Please see the contents sections below for more information.

*The project is carried out within Nesta's [Open Jobs](https://www.nesta.org.uk/project/open-jobs/) programme, and supported by the J.P.Morgan as part of the [New Skills At Work initiative](https://www.jpmorganchase.com/impact/our-approach/jobs-and-skills).*

## Contents

### Supplementary online data of the project report

`STATUS: To be published on November 23rd, 2020`

The supplementary data consists of several folders corresponding to the different parts of the Mapping Career Causeways project report:

- [**Crosswalk between O\*NET and ESCO occupations**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/ONET_ESCO_crosswalk/): 
A mapping from the US O*NET framework to EU ESCO, together with the code used in developing the crosswalk.
- [**Automation risk estimates**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/automation_risk/): 
Results related to task automation of 1627 ESCO occupations.
- [**Demographic analysis**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/demographic_analysis/): 
Data on at-risk workers in the UK, Italy and France.
- [**Results on job transitions and skills analysis**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/transitions/): The number of safe and desirable transitions for 1627 ESCO occupations, and the effectiveness of core skills in uncovering new transitions.
- [**Skills-based sectors**](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/skills_based_sectors): Breakdown of data-driven groups of related ESCO occupations that share similar job requirements and work characteristics

### Codebase and job transitions

`STATUS: Not yet released; to be published in early 2021`

Project codebase will be published here in early 2021, together with a complete list of the recommended career transitions for all considered ESCO occupations. This will allow anyone to generate transition recommendations, replicate the results and build upon the project outputs.

In the meantime, we will be conducting research to further validate the recommendations from the career transitions algorithm.

## Installation

It is possible to browse and download the [Supplementary online data](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data) without installing anything on your local machine. However, if you'd like to run any of the analysis code, you should follow the steps below. *Note that the complete analysis code will be published in early 2021.*

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

## Feedback

Anyone is welcome to use and build upon the published data and code. If you would like to leave feedback, you can either create a new github issue (for raising technical questions) or [write to us](mailto:open.jobs@nesta.org.uk) for more general enquiries. To refer to this work, please cite the Mapping Career Causeways report:

```
Kanders K., Djumalieva, J., Sleeman, C. and Orlik, J. (2020). Mapping Career Causeways: Supporting Workers at Risk. London: Nesta
```
