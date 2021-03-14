# Mapping Career Causeways codebase

***Public codebase featuring the career transitions algorithm and data analysis of the [Mapping Career Causeways](https://www.nesta.org.uk/project/mapping-career-causeways/) project.***

## Welcome to the codebase!

Mapping Career Causeways takes data from [ESCO](https://ec.europa.eu/esco), [O\*NET](onetonline.org) and academic research to create a career transition recommendation algorithm. Given a worker's current occupation, the algorithm evaluates the fit to other ESCO occupations, in terms of skills requirements, typical work activities, and the interpersonal, physical and structural work context, to recommend *viable* transitions. Subsequently, the algorithm can also account for the expected earnings of the potential destination occupations to suggest *desirable* transitions, and their risk of automation to suggest transitions to *safer* employment.

We have also carried out a [crowdsourcing study](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/reports/crowd_feasibility_ratings/) to better understand public perceptions of transition feasibility. This data can be used to further refine the algorithm's recommendations.

The code in this repository will allow you to replicate the results of our [research report](https://www.nesta.org.uk/report/mapping-career-causeways-supporting-workers-risk/), generate your own career transition recommendations, and build upon the project outputs.

*We would be grateful if anyone who downloads or uses our resources would complete this [2 minute survey](https://docs.google.com/forms/d/1IepcbAmIKAS2fDaDO4NfcI7uqNRkof02s52VmEYrZCY/edit?ts=6005b209&gxids=7628). Thank you!*

## Getting started

After following the [installation instructions](https://github.com/nestauk/mapping-career-causeways/#installation), you can begin with the [tutorial notebooks](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/notebooks/). `Tutorial_01_transitions.ipynb` demonstrates how to use the codebase to generate career transition recommendations for any ESCO occupation, inspect the differences and similarities between occupations, analyse skills gaps and simulate upskilling. Moreover, it also demonstrates how to incorporate insights from the [crowdsourcing study on transition feasibility](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/reports/crowd_feasibility_ratings/) into the recommendations.

The second tutorial `Tutorial_02_exploring_data.ipynb` gives a brief tour on accessing and exploring the different occupational and skills data.

For a high-level overview of the methodology, you can also check this [webinar](https://www.youtube.com/watch?v=TrwEhOGxkjU), presented as part of the [ESCoE Covid-19 Economic Measurement series](https://www.escoe.ac.uk/events/mapping-career-causeways-for-workers-displaced-by-automation-and-covid-19/).

### Tutorials on Google Colab

To quickly try out the transition recommendation algorithm, you can also use the cloud-based Google Colab tutorials. These don't require you to download or install anything on your local machine:
- [Tutorial #1](https://colab.research.google.com/drive/1odo3NAHQdYEKHGQmrsQ_7kCK0TAzqgg-?usp=sharing): Introduction on generating and analysing transition recommendations
- [Tutorial #2](https://colab.research.google.com/drive/16p86KOUfiAaPUOYjNxb2pAHSgPAHAtLI?usp=sharing): Exploring the underlying occupation and skills data.
- [Tutorial #3](https://colab.research.google.com/drive/1VctohO6z9sigwXAd8HtNcGiLeyISxe99?usp=sharing): Exploring prevalent skills gaps for cross-sectoral transitions

## Using the codebase

### Replicating results
Data analyses underpinning the project [report](https://www.nesta.org.uk/report/mapping-career-causeways-supporting-workers-risk/) are available as a set of Jupyter notebooks in the [notebooks](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/notebooks) folder. These are organised into several sections pertaining to different parts of the project. The notebooks clearly explain the analysis steps, and produce output files that are stored in `data/interim` and `data/processed` and used in further downstream analyses.

### Exploring transition recommendations
We have generated a validated set of safe and desirable transition recommendations that also takes into account people's perceptions of their feasibility. More information about the validation can be found [in the technical appendix](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/reports/crowd_feasibility_ratings/). The set of validated transitions that are also perceived as feasible can be found [here](), and the data can be explored via [this tutorial](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/notebooks/Tutorial_02_exploring_data.ipynb).

### Generating career transitions
For generating your own career transition recommendations, inspecting skills gaps, and identifying transferable skills - subject to your own transition viability, desirability and feasibility parameters - you can use the functions and classes defined in the module `../mapping_career_causeways/transitions_utils.py` (see [this tutorial](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/notebooks/Tutorial_01_transitions.ipynb) for examples).

For a quick start, try the code below; this will output transitions for the shop assistant, which were examined in [this data visualisation](https://data-viz.nesta.org.uk/career-causeways/index.html):
```python
import mapping_career_causeways.transitions_utils as trans_utils
transitions = trans_utils.get_transitions(origin_ids=[139])
transitions[transitions.is_viable]
```

## Modules
Besides the module `transitions_utils` for generating and analysing career transitions, the folder [mapping_career_causeways](https://github.com/nestauk/mapping-career-causeways/tree/main/mapping-career-causeways) hosts other custom packages and utilities that were developed by us for performing various functions, such as calculating the NLP-adjusted overlap (`compare_nodes_utils.py`) and performing consensus clustering (`cluster_utils.py`).

Note that the code and data for generating the crosswalk between the O*NET and ESCO occupations has been factored out in the [Supplementary online data section](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/ONET_ESCO_crosswalk) of the repository.
