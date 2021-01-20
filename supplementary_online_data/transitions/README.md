# Results on job transitions and skills analysis

For each ESCO occupation, we identified alternative occupations that are similar and into which a worker could potentially transition.

We combined the information in [ESCO](https://ec.europa.eu/esco) and [O\*NET](onetonline.org) frameworks via our [crosswalk](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/ONET_ESCO_crosswalk/), and measured ‘similarity’ between pairs of occupations by comparing the essential and optional skills required to perform each job, the particular work activities, the interpersonal, physical and structural work characteristics, and the typical required levels of education and experience. Those jobs which are sufficiently similar are deemed ‘viable transitions’ for the at-risk worker. The subset of these occupations which offer comparable or higher levels of pay is called ‘desirable transitions’. Finally, an even smaller subset that we call ‘safe and desirable transitions’ will reduce workers’ exposure to automation.

Below, we describe tables with the main results related to the career transition algorithm outputs. These results and the full set of recommended transitions can be replicated using the analysis notebooks in `codebase/notebooks/05_transition_analyses` [folder](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/notebooks/05_transition_analyses).

Moreover, we also provide a [curated set of transitions](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/transitions/transitions_tables/) that were recommended by our algorithm and that have been additionally validated by a subsequent crowdsourcing study that we carried out after publishing the report. This is a more restricted set of transitions than the one analysed in the report; more details on the crowdsourcing study can be found [here](https://github.com/nestauk/mapping-career-causeways/tree/main/codebase/reports/crowd_feasibility_ratings/).

## Contents

- [Number of transition options for ESCO occupations](#esco)
- [Transition matrices](#matrices)
- [Core skills](#skills)

<a name="esco"></a>
## Number of transition options for ESCO occupations

**`mcc_transitions_Occupation_data.csv`**

Estimates of the number of safe and desirable transition options for the 1627 ESCO occupations that were analysed in the Mapping Career Causeways project. The table also contains linked occupation data that was used in determining the viability and desirability of transitions.

| Column name   | Description   |  Source |
|:---------------|:---------------|:------|
|id   | Unique integer identifier of the ESCO occupation; used only internally, within the scope of this project. | - |
|occupation  | Preferred label of the ESCO occupation.   | [ESCO](https://ec.europa.eu/esco/portal/occupation) |
|isco_code   | Four-digit ISCO-08 code, indicating the broader ISCO unit group to which the occupation belongs; the code is provided by the ESCO API. Find more information about ISCO on [ilo.org](https://www.ilo.org/public/english/bureau/stat/isco/isco08/). | [ESCO](https://ec.europa.eu/esco/portal/occupation) |
| skills_based_sector_code | Code indicating skills-based sectors - groups of related occupations sharing similar worker requirements and work characteristics. These groups were determined by applying graph-based clustering (an unsupervised machine learning method; see the Mapping Career Causeways report for more details). | Analysis |
| skills_based_sector | Label of the skills-based sector. | Analysis |
| sub_sector_code | Code indicating skills-based sub-sectors, which are nested within the skills-based sectors. | Analysis |
| sub_sector | Label of the skills-based sub-sector. | Analysis |
|risk_category | Category of automation risk, which is determined by both the overall automation risk and the prevalence of bottleneck tasks (see [Automation risk estimates](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/automation_risk/)). Occupations in the fourth quartile of risk and first quartile of prevalence are 'High risk'; all other occupations are 'Lower risk'. | Analysis |
| job_zone | Indicator of the preparation requirements in terms of education, related work experience and on-the-job training, on a scale of 1 to 5 (1=occupation needs little or no preparation, and 5=occupation needs extensive preparation). See [O\*NET Help](https://www.onetonline.org/help/online/zones) for more information. | [O\*NET](https://www.onetcenter.org/dictionary/22.0/excel/job_zones.html) |
| education_level | Required level of education, on a scale of 1 to 12 (1=less than a high school diploma, and 12=post-doctoral training). See [O\*NET Resource Center](https://www.onetcenter.org/dictionary/20.1/excel/ete_categories.html) for more information on the scale. | [O\*NET](https://www.onetcenter.org/dictionary/20.1/excel/education_training_experience.html) |
| related_work_experience | Required related work experience, on a scale of 1 to 11 (1=none, and 11=over 10 years). See [O\*NET Resource Center](https://www.onetcenter.org/dictionary/20.1/excel/ete_categories.html) for more information on the scale. | [O\*NET](https://www.onetcenter.org/dictionary/20.1/excel/education_training_experience.html) |
| on_the_job_training | Required on-the-job training, on a scale of 1 to 9 (1=none or a short demonstration, and 9=over 10 years). See [O\*NET Resource Center](https://www.onetcenter.org/dictionary/20.1/excel/ete_categories.html) for more information on the scale. | [O\*NET](https://www.onetcenter.org/dictionary/20.1/excel/education_training_experience.html) |
| annual_earnings | Indicative estimates of occupation's annual earnings in GBP. | Analysis/[ASHE](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/occupation4digitsoc2010ashetable14) |
| remote_labor_index | The proportion of job activities that can be preformed remotely. Note that not all occupations have these estimates. | O\*NET/[del Rio-Chanona et al. (2020)](https://zenodo.org/record/3751068#.X6iE8JOTJTY) |
| physical_proximity | Indicator of whether this job requires the worker to perform tasks in close physical proximity to other people (0=worker doesn't work near other people (beyond 100 ft.), and 1=very close, near touching). | [O\*NET](https://www.onetonline.org/find/descriptor/result/4.C.2.a.3) |
| exposure_score | Indicator of potential exposure to the impacts from COVID-19, calculated as the geometric mean of `physical_proximity` and (1 – `remote_labor_index`). | Analysis|
| n_desirable | Number of desirable job transitions (transitions that are above the viability threshold and preserve at least 75\% of the annual earnings; see the Mapping Career Causeways report). | Analysis|
| n_desirable_and_highly_viable | Number of desirable job transitions that are also highly viable (particularly good job fits). | Analysis|
| n_safe_desirable | Number of desirable job transitions to occupations that are not in the 'High risk' category. | Analysis|
| n_safe_desirable_and_highly_viable |  Number of highly viable, desirable job transitions to occupations that are not in the 'High risk' category. | Analysis|
| n_safe_desirable_strict | Number of desirable job transitions to occupations that are not in the 'High risk' category, and that satisfy the stricter condition of simultaneously reducing worker's overall automation risk AND increasing the prevalence of bottleneck tasks. | Analysis|
| n_safe_desirable_strict_and_highly_viable | Number of highly viable, desirable job transitions to occupations that are not in the 'High risk' category, and that satisfy the stricter condition of simultaneously reducing worker's overall automation risk AND increasing the prevalence of bottleneck tasks. | Analysis|
| concept_uri | Universal identifier of the ESCO occupation used by the ESCO API. Find more information in the [ESCO documentation](https://ec.europa.eu/esco/api/doc/esco_api_doc.html#rest-calls-get-conceptschemes-by-uris). | ESCO |  

<a name="matrices"></a>
## Transition matrices

**`mmcc_transitions_Matrix_sectors.csv`**  
**`mcc_transitions_Matrix_subsectors.csv`**

Transition matrices between skills-based sectors and sub-sectors. Each row of the matrix shows the proportion of safe and desirable transitions that go from high-risk occupations in one sector to lower-risk occupations in other sectors. The transitions satisfy the stricter condition for safe transitions by simultaneously reducing worker's overall automation risk and increasing the prevalence of bottleneck tasks.

<a name="skills"></a>
## Core skills

Effect of different types of upskilling on workers’ range of transitions. Using methods from network science, we identified 100 core ESCO skills which reflect the central competencies for a wide range of jobs. To further evaluate the impact of adopting a new core skill, these were added, one at a time, to the skills set of each high-risk occupation and the similarities to other occupations were recalculated. We then evaluated the number of new safe and desirable transitions that emerged as the result of learning each skill.

**`mcc_transitions_Core_skills.csv`**  
Core skills and their effectiveness across all high-risk occupations.

**`mcc_transitions_Core_skills_Business_admin.csv`**  
Effectiveness across all high-risk occupations from the Business and administration skills-based sector (code 2).

**`mcc_transitions_Core_skills_Services.csv`**  
Effectiveness across all high-risk occupations from the Sales and services skills-based sector (code 3).

**`mcc_transitions_Core_skills_ICT.csv`**  
Effectiveness across all high-risk occupations from the ICT skills-based sector (code 12).

**`mcc_transitions_Core_skills_Arts.csv`**  
Effectiveness across all high-risk occupations from the Arts and media skills-based sector (code 6).

| Column name   | Description   |  
|:---------------|:---------------|
|skill_id   | Unique integer identifier of the O\*NET task. |
|id   | Unique integer identifier of the ESCO skill; used only internally, within the scope of this project. |
|esco_skill  | Preferred label of the ESCO skill.   |
|avg_new_safe_desirable | Average number of new safe and desirable transitions, that simultaneously reduce worker's overall automation risk AND increase the prevalence of bottleneck tasks.|
|med_new_safe_desirable | Median number of new safe and desirable transitions, that simultaneously reduce worker's overall automation risk AND increase the prevalence of bottleneck tasks.|
| significant_effect| True if the effect across the examined group of occupations is significantly different from 0 (Wilcoxon signed-rank test; p-value<1e-5 in `mcc_transitions_Core_skills.csv`, p-value<0.01 in the other tables)|
|coreness | A measure of coreness that combines betweenness centrality, eigenvector centrality and clustering coefficient metrics. The coreness measure of a skill will be high if the skill is connected to diverse sets of skills that are weakly connected to each other. The shown values are normalised with respect to the skill with the highest coreness measure. |
|esco_skill_category | First-level category of the [ESCO skills hierarchy](https://ec.europa.eu/esco/portal/escopedia/Skills_pillar) |
|esco_skill_subcategory | Second-level category of the [ESCO skills hierarchy](https://ec.europa.eu/esco/portal/escopedia/Skills_pillar) |
| concept_uri | Universal identifier of the ESCO skill used by the ESCO API. Find more information in the [ESCO documentation](https://ec.europa.eu/esco/api/doc/esco_api_doc.html#rest-calls-get-conceptschemes-by-uris). |
