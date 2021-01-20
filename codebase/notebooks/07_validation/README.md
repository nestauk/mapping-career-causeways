# Analysis of crowdsourced feasibility ratings

After publishing the project report, we carried out a subsequent crowdsourcing study to better understand public perceptions of transition feasibility and validate the outputs from our career transitions algorithm. [A technical appendix](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/reports/crowd_feasibility_ratings/) describes the study design and the initial results.

The crowdsourced dataset can be found in `../../data/processed/validation/crowd_feasibility_ratings.csv` or downloaded separately from [S3](https://ojd-mapping-career-causeways.s3.eu-west-2.amazonaws.com/data/processed/validation/crowd_feasibility_ratings.csv) (see the technical appendix for a schema).

## Sampling of the test data

**`Validation_00_Generate_validation_data.ipynb`**  
Generates a sample of approximately 10,000 transitions for the crowdsourcing study.

## Analysis

**`Validation_01_Preprocessing_and_sensitive_data.ipynb`**  
Preprocesses the raw crowdsourcing results and, for reasons of extra caution,
removes the year of birth and location information about contributors (included for the sake of completeness).

**`Validation_02_EDA_and_figures.ipynb`**  
Exploratory data analysis of the crowdsourcing results.

**`Validation_03_Predictive_model.ipynb`**  
Creates a model to predict the crowd feasibility ratings for transitions between
any two occupations.

## Curated transitions

**`Validation_04_Output_validated_transitions.ipynb`**  
Generates a curated set of transitions for all top level ESCO occupations, that are
both safe and desirable according to the career transitions algorithm, and feasible
according to the crowd feasibility judgments.
