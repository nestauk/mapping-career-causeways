# Automation risk estimates

Estimates of automation risk for European occupations specified in the European Skills, Competences, Qualifications and Occupations framework ([ESCO](https://ec.europa.eu/esco/)). We translated the results by [Brynjolfsson, Mitchell and Rock (2018)](https://www.aeaweb.org/articles?id=10.1257/pandp.20181019), which were originally obtained for the US [O\*NET](https://www.onetonline.org/) occupations, to ESCO by developing a [crosswalk](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/ONET_ESCO_crosswalk/) between both frameworks. Please find below the descriptions of the provided tables.

## Contents

- [Automation risk estimates for ESCO occupations](#esco)
- [Automation risk estimates for ISCO occupational groups](#isco)
- [Mapped tasks for each ESCO occupation](#esco_tasks)
- [Impact of tasks and activities on automation risk](#tasks)

<a name="esco"></a>
## Automation risk estimates for ESCO occupations

**`mcc_risk_All_occupations.csv`**

Estimates of the overall automation risk and the
prevalence of bottleneck tasks for 1627 ESCO occupations that were analysed in the Mapping Career Causeways project. These occupations build up the
top level of the ESCO hierarchy of occupations (the next intermediate level after ISCO four-digit unit groups). Other, lower level
occupations (level 6 to level 8) may inherit the automation risk of their corresponding broader occupations.

**Warning:** Note that there are cases where our crosswalk has mapped several individual (related) ESCO occupations to one O\*NET occupation, and hence they share the same estimates of overall automation risk and prevalence of bottleneck tasks. In these cases, the automation estimates should be interpreted with caution, as the more nuanced differences between the individual ESCO occupations could not be taken into account (this warning is especially pertinent for several creative ESCO occupations that have been mapped to 'multimedia artists and animators' in O\*NET). Note, however, that the number of safe and desirable transitions may vary substantially among the different ESCO occupations mapped to the same O\*NET occupation, which highlights the value of using the more granular ESCO framework as the foundation for the career transitions algorithm.

| Column name   | Description   |  
|:---------------|:---------------|
|id   | Unique integer identifier of the ESCO occupation; used only internally, within the scope of this project. |   
|occupation  | Preferred label of the ESCO occupation.   |   
|isco_code   | Four-digit ISCO-08 code, indicating the broader ISCO unit group to which the occupation belongs; the code is provided by the ESCO API. Find more information about ISCO on [ilo.org](https://www.ilo.org/public/english/bureau/stat/isco/isco08/).    |
|risk | Overall automation risk, on a scale of 1 to 5 (1=minimal risk and 5=maximal risk).  Note that the contribution of each task to the occupation-level risk is weighted by the task's importance. |
|prevalence | Prevalence of bottleneck tasks, on a scale from 0 to 1 (0=occupation has no bottleneck tasks, and 1=all of the occupation's tasks are bottlenecks). Task is defined to be an automation bottleneck if at least one of its 'automation dimensions' has a rating smaller than or equal to 2. Note that the contribution of each bottleneck task is weighted by its importance. |
|risk_category | Category of automation risk, which is determined by both the overall automation risk and the prevalence of bottleneck tasks. Occupations in the fourth quartile of risk and first quartile of prevalence are 'High risk'; conversely, occupations in the first quartile of risk and last quartile of prevalence are 'Low risk'. Occupations outside these two bands were labelled as 'Other'. |
|risk_nonphysical | Overall automation risk of non-physical tasks (tasks that don't require dexterity or physical labour). |
|prevalence_nonphysical | Prevalence of bottleneck tasks when requirements of dexterity and physical labour are no longer bottlenecks to automation. |
|onet_code | O\*NET code of the ESCO occupation, inferred by using a [crosswalk](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/ONET_ESCO_crosswalk/) developed by the authors.  |
|onet_occupation | Title of the O\*NET occupation. |
|concept_uri | Universal identifier of the ESCO occupation used by the ESCO API. Find more information in the [ESCO documentation](https://ec.europa.eu/esco/api/doc/esco_api_doc.html#rest-calls-get-conceptschemes-by-uris). |  
| skills_based_sector_code | Code indicating skills-based sectors - groups of related occupations sharing similar worker requirements and work characteristics. These groups were determined by applying graph-based clustering (an unsupervised machine learning method; see the Mapping Career Causeways report for more details). | Analysis |
| skills_based_sector | Label of the skills-based sector. | Analysis |
| sub_sector_code | Code indicating skills-based sub-sectors, which are nested within the skills-based sectors. | Analysis |
| sub_sector | Label of the skills-based sub-sector. | Analysis |

<a name="isco"></a>
## Automation risk estimates for ISCO occupational groups

**`mcc_risk_ISCO_3digit.csv`**

Estimates of automation risk for three-digit ISCO minor groups. These estimates were obtained by taking the average across ESCO occupations in the same ISCO occupational group. Find more information about ISCO on [ilo.org](https://www.ilo.org/public/english/bureau/stat/isco/isco08/).

| Column name   | Description   |  
|:---------------|:---------------|
|isco_code   | Three-digit ISCO-08 code of the minor group.|
|isco_minor_group | Title of the occupational group |
|risk|Overall automation risk, on a scale of 1 to 5 (1=minimal risk and 5=maximal risk). |
|prevalence| Prevalence of bottleneck tasks, on a scale from 0 to 1 (0=occupations have no bottleneck tasks, and 1=all of the occupations' tasks are bottlenecks).
|risk_category | Category of automation risk, which is determined by both the overall automation risk and the prevalence of bottleneck tasks. Occupational groups in the fourth quartile of risk and first quartile of prevalence are 'High risk'; conversely, occupational groups in the first quartile of risk and last quartile of prevalence are 'Low risk'. Occupational groups outside these two bands were labelled as 'Other'.  |

&nbsp;    
**`mcc_risk_ISCO_4digit.csv`**

Estimates of automation risk for four-digit ISCO unit groups. These estimates were obtained by taking the average across ESCO occupations in the same ISCO occupational group. Find more information about ISCO on [ilo.org](https://www.ilo.org/public/english/bureau/stat/isco/isco08/).

| Column name   | Description   |  
|:---------------|:---------------|
|isco_code   | Four-digit ISCO-08 code of the unit group.  |
|isco_unit_group | Title of the occupational group. |
|risk|Overall automation risk, on a scale of 1 to 5 (1=minimal risk and 5=maximal risk). |
|prevalence| Prevalence of bottleneck tasks, on a scale from 0 to 1 (0=occupations have no bottleneck tasks, and 1=all of the occupations' tasks are bottlenecks).
|risk_category | Category of automation risk, which is determined by both the overall automation risk and the prevalence of bottleneck tasks. Occupational groups in the fourth quartile of risk and first quartile of prevalence are 'High risk'; conversely, occupational groups in the first quartile of risk and last quartile of prevalence are 'Low risk'. Occupational groups outside these two bands were labelled as 'Other'. |

<a name="esco_tasks"></a>
## Mapped tasks for each ESCO occupation

**`mcc_risk_ESCO_to_tasks.csv`**

ESCO occupations and their corresponding O\*NET tasks. You can use this table to explore how specific tasks are contributing to the occupation's overall automation score.

The tasks have been mapped via our [crosswalk](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/ONET_ESCO_crosswalk/). Each task has a suitability for machine learning (SML) score inferred from the Brynjolfsson et al. ratings. To arrive at the occupation-level automation risk, the task SML scores for each occupation have been weighted by their importance scores and then summed up.

| Column name   | Description   |  
|:---------------|:---------------|
|id   | Unique integer identifier of the ESCO occupation; used only internally, within the scope of this project. For the corresponding official ESCO identifier, see the table `mcc_risk_All_occupations.csv`. |   
|esco_occupation  | Preferred label of the ESCO occupation. |   
|task_id   | Unique integer identifier of the O\*NET task. |
|onet_task | Title of the O\*NET task. |
|weight| Relative importance weight of the task in the particular occupation. These have been inferred from the O\*NET database, and normalised such that all weights for each occupation sum up to 1. |
| mean_task_SML | Task-level suitability for machine learning (SML) that is calculated by averaging the ratings of columns `q1`-`q14`, `qD` and `q19`-`q23`.|
|weighted_task_SML | `mean_task_SML` multiplied by the importance weight `weight`. The occupation-level overall automation risk is obtained by summing these scores up across all occupation's tasks | 
|has_bottlenecks | True if task is a bottleneck task (task is defined to be a bottleneck if at least one of the `q{x}` columns has a score less than or equal to 2). |
|onet_occupation | Title of the O\*NET occupation corresponding to the ESCO occupation. |
|onet_code | Corresponding O\*NET code, inferred by using our [crosswalk](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/ONET_ESCO_crosswalk/).  |
|q\{x\}| Ratings across 23 questions (so-called 'automation dimensions') evaluated in the study by Brynjolfsson et al. (2018). To find more about these questions, consult their [rubric](https://www.openicpsr.org/openicpsr/project/114436/version/V1/view?path=/openicpsr/114436/fcr:versions/V1/HITRubric.docx&type=file) and [supplementary data](https://www.openicpsr.org/openicpsr/project/114436/version/V1/view;jsessionid=163D056C673E82F00822034978FC9074?path=/openicpsr/114436/fcr:versions/V1/What-Can-Machines-Learn_DataAppendixReadme.pdf&type=file). Note that the ratings of `q15`-`q18` (questions related to the use of different types of data) have been aggregated in `qD` (a measure of 'data intensity').|

<a name="tasks"></a>
## Impact of tasks and activities on automation risk

The tables below can be used to identify the tasks and activities that are most likely
to rise or lower occupations' overall risk of automation.

&nbsp;   
**`mcc_risk_Tasks.csv`**

The overall impact of tasks on occupations-level automation risk was obtained
by removing tasks (one at a time) from the occupations in which they are required, recalculating the overall automation risk for
these occupations, and comparing the recalculated risk with the original risk.
For each task, its impact score was calculated by summing up the differences between the recalculated and original risks across
all occupations where the task is required.

| Column name   | Description   |  
|:---------------|:---------------|
|task_id   | Unique integer identifier of the O\*NET task. |
|task | Title of the O\*NET task. |
|detailed_work_activity | Detailed work activity to which the task belongs (find more information on [O\*NET](https://www.onetcenter.org/dictionary/21.0/text/tasks_to_dwas.html)) . |
|intermediate_work_activity | Intermediate work activity to which the detailed work activity belongs. |
| element| [Element](https://www.onetonline.org/find/descriptor/browse/Work_Activities/) (a broad group of work activities) to which the intermediate work activity belongs. |
| mean_task_SML | Task-level suitability for machine learning (SML) that is calculated by averaging the ratings of columns `q1`-`q14`, `qD` and `q19`-`q23`.|
|impact | Task impact on occupations' automation risk. Negative impact score values indicate that the task is lowering the automation risk of occupations whereas positive values indicate that the task is putting occupations more at risk. |
|q\{x\}| Ratings across 23 questions (so-called 'automation dimensions') evaluated in the study by Brynjolfsson et al. (2018). To find more about these questions, consult their [rubric](https://www.openicpsr.org/openicpsr/project/114436/version/V1/view?path=/openicpsr/114436/fcr:versions/V1/HITRubric.docx&type=file) and [supplementary data](https://www.openicpsr.org/openicpsr/project/114436/version/V1/view;jsessionid=163D056C673E82F00822034978FC9074?path=/openicpsr/114436/fcr:versions/V1/What-Can-Machines-Learn_DataAppendixReadme.pdf&type=file). Note that the ratings of `q15`-`q18` (questions related to the use of different types of data) have been aggregated in `qD` (a measure of 'data intensity').|

&nbsp;   
**`mcc_risk_DWAs.csv`**

The task impact scores were aggregated to the broader level of detailed work activities (DWAs) by summing up the impacts of tasks belonging
to the same DWA. To further identify the most risky and safest elements (even broader aggregations of activities) we used the median impact value across their corresponding DWAs.

| Column name   | Description   |  
|:---------------|:---------------|
|detailed_work_activity | Detailed work activities (DWAs; broader aggregations of tasks). |
|intermediate_work_activity | Intermediate work activity to which the detailed work activity belongs. |
|impact | DWA impact on occupations' automation risk. Negative impact score values indicate that the activity is lowering the automation risk of occupations whereas positive values indicate that the activity is putting occupations more at risk. |
|intermediate_work_activity | Intermediate work activity to which the detailed work activity belongs. |
| element| [Element](https://www.onetonline.org/find/descriptor/browse/Work_Activities/) (a broad group of work activities) to which the intermediate work activity belongs. |

&nbsp;   
**`mcc_risk_Bottlenecks.csv`**

Most common task automation bottlenecks (data underpinning the chart shown in Figure 21 of the Mapping Career Causeways report). The bottlenecks correspond to the 23 different 'automation dimensions' specified in the [rubric](https://www.openicpsr.org/openicpsr/project/114436/version/V1/view?path=/openicpsr/114436/fcr:versions/V1/HITRubric.docx&type=file) by Brynolfsson et al.

| Column name   | Description   |  
|:---------------|:---------------|
|sml_question | Question from the Brynolfsson et al.'s rubric. Note that the ratings of `q15`-`q18` (questions related to the use of different types of data) have been aggregated in `qD` (a measure of 'data intensity'). |
|task_bottleneck | Bottleneck corresponding to the rubric's question. |
|is_bottleneck | Number of instances when this question/automation dimension has contributed to a task being an automation bottleneck (i.e. has had a rating smaller than or equal to 2). |
|percentage_of_all_bottleneck_tasks | Indicates the percentage of all bottleneck tasks that has this bottleneck; note that a task can have several bottlenecks. |
