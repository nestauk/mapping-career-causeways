# Analyses of occupational automation risk

Generate estimates of automation risk for European occupations specified in the European Skills, Competences, Qualifications and Occupations framework ([ESCO](https://ec.europa.eu/esco/)), by translating the results from [Brynjolfsson, Mitchell and Rock (2018)](https://www.aeaweb.org/articles?id=10.1257/pandp.20181019) from the US [O\*NET](https://www.onetonline.org/) occupations to ESCO using our [crosswalk](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/ONET_ESCO_crosswalk/).

## Automation risk for ESCO occupations
**`Automation_risk_for_ESCO_occupations.ipynb`**  
Calculates the overall automation risk and the prevalence of bottleneck tasks for 1627 ESCO occupations (and extends the mapping to the rest of the 2942 ESCO occupations as well). For this purpose, we used the estimates of task suitability for machine learning (SML), originally developed by Brynjolfsson, Mitchell and Rock (2018) for the O\*NET occupations

## Task impact on automation risk
**`Impact_tasks_on_automation.ipynb`**  
Calculates the impact of tasks and work activities on the risk of automation. This was used in the Mapping Career Causeways report to find the types of tasks and work activities that raise or lower automation risk the most.
