# Linking ESCO occupations to labour market data

Here, we build richer occupational profiles by linking ESCO occupations to data such as estimates of annual earnings, capacity to work remotely etc. The output tables of these notebooks are stored in `codebase/data/processed/linked_data` folder.

For more background on the different data sources and how we used them, see pp. 18-20 and pp. 28-34 of the project [research report](https://www.nesta.org.uk/report/mapping-career-causeways-supporting-workers-risk/).

## Education and experience requirements
**`Link_occupations_to_Job_Zones.ipynb`**  
Links ESCO occupations to O\*NET's [Job Zones](https://www.onetonline.org/help/online/zones) and the variables that underpin the Job Zone estimates: education, related work experience, and on-the-job training.

## Annual earnings and paid hours
**`Link_UK_SOC_to_ISCO.ipynb`**  
Explores two options for crosswalking labour market data associated with UK SOC codes to the ESCO occupations: (1) based on an official crosswalk between ISCO-08 and UK SOC codes, or (2) based on the UK SOC coding index; exports preprocessed crosswalk tables.

**`Link_occupations_to_UK_earnings_hours.ipynb`**  
Links earnings and hours data from ASHE table 14 to ESCO occupations, by using a UK SOC to ISCO (and hence to ESCO) crosswalk inferred from UK SOC coding index.

In addition, calibrates a lower bound of acceptable earnings difference when making a job transition (necessary for determining desirable job transitions).

## Employment
**`Link_occupations_to_UK_employment.ipynb`**  
Uses employment estimates for three-digit ISCO occupational groups from the EU LFS to derive rough estimates of workers employed in each 'top level' ESCO occupation. In this particular example, we use estimates for the UK in 2018.

## Potential impact from COVID-19
**`Link_occupations_to_Remote_Labor_Index.ipynb`**  
Links Remote Labor Index (developed by [del Rio-Chanona et al. 2020](https://www.oxfordmartin.ox.ac.uk/publications/supply-and-demand-shocks-in-the-covid-19-pandemic-an-industry-and-occupation-perspective/)) to ESCO occupations. The Remote Labor Index indicates what proportion of occupation's work activities can be performed remotely.

**`Estimate_COVID_Exposure.ipynb`**  
Derives a simple estimate for occupational-level exposure to the impact from COVID-19 based on the workers’ physical proximity to other people (determined from O*NET) and on the extent to which the work has to be performed on-site (derived from the Remote Labor Index).
