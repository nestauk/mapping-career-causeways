# Crowdsourcing feasibility ratings of career transitions

After publishing the Mapping Career Causeways project report, we have carried out a subsequent crowdsourcing study to better understand public perceptions of transition feasibility and validate the outputs from our career transitions algorithm.

This part of the project work is being led by George Richardson, with additional thanks to Karlis Kanders, Genna Barnett, and Cath Sleeman. If you would like to build upon the crowdsourced data of transition feasibility ratings, please cite this github repository.

## Technical appendix

[The technical appendix](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/reports/crowd_feasibility_ratings/Mapping_Career_Causeways_Crowdsourcing_study.pdf) describes the study design and the initial results. We tested about 10,000 transitions in terms of whether an imagined worker in the origin occupation could reasonably make the transition to the suggested destination, based on a “common sense” judgement by members from the public. We then used these results to create a predictive model for generating crowd feasibility ratings for transitions between any two ESCO occupations.

## Dataset

The crowdsourced dataset of the transition feasibility ratings can be found, as part of the data download from S3, in `data/processed/validation/crowd_feasibility_ratings.csv` (see the technical appendix for a schema). Alternatively, you can also access it directly from [S3](https://ojd-mapping-career-causeways.s3.eu-west-2.amazonaws.com/data/processed/validation/crowd_feasibility_ratings.csv).

## Analysis code

The [notebooks for analysing the crowdsourcing results](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/notebooks/07_validation/) are located in `notebooks/07_validation/`. See also [this tutorial](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/notebooks/Tutorial_01_transitions.ipynb) which touches upon how to generate a feasibility prediction for any transition.
