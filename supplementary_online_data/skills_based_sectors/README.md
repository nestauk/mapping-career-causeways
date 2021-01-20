# Skills-based sectors

To find and analyse groups of related occupations that share similar job requirements and work characteristics, we used our occupation similarity measures and a graph-based clustering approach. The resulting grouping of occupations organises the 2942 ESCO occupations at two hierarchical levels - that we call skills-based sectors and sub-sectors - with 14 groups at the first level and 54 groups at the second level (you can replicate this analysis using the notebooks `Clustering_{..}` in `../../codebase/notebooks/05_transition_analyses/`).

## ESCO occupation membership to skills-based sectors

**`mcc_skills_based_sectors.csv`**

Listing of all skills-based sectors and sub-sectors.

&nbsp;  
**`mcc_skills_based_sectors_All_ESCO.csv`**  

Skills-based sectors and sub-sectors for 2942 ESCO occupations (including narrower, lower level occupations). Note that the clustering was performed on all occupations, but in the Mapping Career Causeways report we only on a subset of the 1627 higher level ESCO occupations.

| Column name   | Description   |
|:---------------|:---------------|
|id   | Unique integer identifier of the ESCO occupation; used only internally, within the scope of this project. |
|occupation  | Preferred label of the ESCO occupation.   |
|isco_code   | Four-digit ISCO-08 code, indicating the broader ISCO unit group to which the occupation belongs; the code is provided by the ESCO API. Find more information about ISCO on [ilo.org](https://www.ilo.org/public/english/bureau/stat/isco/isco08/). |
| skills_based_sector_code | Code indicating skills-based sectors. |
| sub_sector_code | Code indicating skills-based sub-sectors, which are nested within the skills-based sectors. |
| skills_based_sector | Label of the skills-based sector. |
| sub_sector | Label of the skills-based sub-sector. |
| concept_uri | Universal identifier of the ESCO occupation used by the ESCO API. Find more information in the [ESCO documentation](https://ec.europa.eu/esco/api/doc/esco_api_doc.html#rest-calls-get-conceptschemes-by-uris). |

## Skills-based sectors and automation risk

**`mcc_skills_based_sectors_Risk.csv`**  
**`mcc_skills_based_sub_sectors_Risk.csv`**

Number of occupations in each [automation risk](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/automation_risk/) category, for each skills-based sector and sub-sector.

| Column name   | Description   |
|:---------------|:---------------|
| low_risk  | Number of occupations in the 'Low risk' category  |
| other  | Number of occupations in the 'Other' category    |
| high_risk  | Number of occupations in the 'High risk' category    |
