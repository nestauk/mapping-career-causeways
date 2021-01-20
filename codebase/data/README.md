# Data

This folder hosts raw and transformed input data for the project analyses. These files
are too large to be shared via github, and instead are hosted separately on S3.

## Setting up

To download the data,
follow the [installation instructions](https://github.com/nestauk/mapping-career-causeways/#installation).

The data is then processed and analysed within the [notebooks](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/notebooks) in the `../notebooks/` folder. The data download already contains the transformed and processed data, so you don't have
to necessarily rerun all of the notebooks.

## Datasets  

The analyses are mainly based on the following datasets:

- **[ESCO](https://ec.europa.eu/esco)** (v1.0.5) as the foundation of our career transitions framework
- **[O\*NET](https://www.onetonline.org/)** (v24.2) for linking rich occupational features and automation risk estimates to ESCO
- **[ASHE](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/occupation4digitsoc2010ashetable14)** (Table 14, provisional 2019) for data on annual earnings and paid hours
- **[Brynjolfsson et al. (2018)](http://openicpsr.org/openicpsr/project/114436/version/V1/view)** study on task suitability for machine learning
- **[del Rio-Chanona et al. (2020)](https://zenodo.org/record/3751068#.YAernpOTJTY)** study on remote labor index and COVID-19 shocks

See an extended discussion on these data in the [project report](https://www.nesta.org.uk/report/mapping-career-causeways-supporting-workers-risk).

Note that presently we have not implemented a functionality to automatically update the career transitions framework with the newest versions of ESCO, O\*NET and other datasets.
