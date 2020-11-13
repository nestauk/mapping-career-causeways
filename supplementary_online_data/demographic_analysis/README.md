# Demographic analysis

To identify the characteristics of at-risk workers, we focused on three European countries: the United Kingdom (UK), France and Italy. We built up a picture of workers (including their gender mix, income and education levels) and their working patterns (i.e. full time or part time) using microdata from the [EU Labour Force Survey](https://ec.europa.eu/eurostat/web/microdata/european-union-labour-force-survey).

## Contents

- [National demographic data](#nat_demo)
- [Regional demographic data](#reg_demo)
- [Employment by ISCO occupational groups](#isco)

<a name="nat_demo"></a>
## National demographic data

**`all_count_demo_breakdown/`**

Contains tables with national employment figures (in thousands) in the UK, Italy and France in years 2014-2018 across different demographic variables. Each file is named by following the pattern:

```
{country_code}_combined_breakdown_k_{variable_name}.csv
```

Each table features a column `RISK_CAT` that indicates the automation risk category: 0=high risk, 1=low risk, and 2=other. Workers were grouped in these risk categories based on our estimates of automation risk of the workers' ISCO three-digit occupational codes. Specifically, occupational groups in the fourth quartile of overall automation risk and first quartile of prevalence of bottleneck tasks are 'High risk'; conversely, occupational groups in the first quartile of risk and last quartile of prevalence are 'Low risk'. Other occupations are labelled as 'Other' (see [Automation risk](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/automation_risk/) folder for the risk category of each ISCO code, and see the project [report](https://www.nesta.org.uk/report/mapping-career-causeways-supporting-workers-risk/) for more information about the methodology).

 The variables that we examined are listed in the table below:

| variable_name   | Description   |  
|:---------------|:---------------|
| age   | Age group, includes +/- 2 years (for example: 17=age group of 15-19 year-olds). |   
|ftpt  | Full-time vs. part-time distinction (1=full-time, and 2=part-time). |   
| hatlev1d | Level of education (L=Lower secondary, M=Upper secondary and post-secondary non-tertiary, H=Tertiary). |
| incdecil | Income decile. |
| nace1d | NACE industry codes; see the list of industries [here](https://en.wikipedia.org/wiki/Statistical_Classification_of_Economic_Activities_in_the_European_Community). |
| sex | 1=male, 2=female. |  

Note that all cells that show data from 20 or less workers have been masked out with an 'x', in accordance with the EU LFS publishing guidelines. For more information about the different variables, see also the [EU LFS User Guide](https://ec.europa.eu/eurostat/documents/1978984/6037342/EULFS-Database-UserGuide.pdf).

&nbsp;  
**`all_prop_demo_breakdown/`**

Contains the same data as `all_count_demo_breakdown/` but expressed as percentages. The files follow a similar naming pattern as shown above.

<a name="reg_demo"></a>
## Regional demographic data

&nbsp;
**`{country_code}_nuts_LQ.csv`**  

Tables with employment (in thousands) in the three risk categories for different regions in the UK, Italy and France, in years 2014-2018.  Data is available at the 2nd level of the Nomenclature of Territorial Units for Statistics (NUTS) for France and Italy, and at the 1st level for the UK.

| Column name | Description   |  
|:---------------|:---------------|
| region | Region code of workers' household. |   
| NUTS  | NUTS code of the region (for maps, see [Eurostat github](https://github.com/eurostat/Nuts2json)). |   
| region_name | Name of the region. |
| risk_category | [Automation risk](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/automation_risk/) category. |
| 2014 - 2018 | Employment (in thousands) in the particular region and automation risk category, in the years 2014-2018. |
| average | Average employment in the particular region and automation risk category in years 2014-2018. |
| percentage | Percentage of the average total employment in years 2014-2018 in a particular automation risk category.  |
| LQ | Local quotient, i.e. the ratio between the local percentage of workers in a particular automation risk category and the nationwide share of workers in the same risk category. |

&nbsp;  
**`regional_demo_count_breakdown/`**

Detailed demographic profiles of select regions: Île de France (France), Lazio and Lombardy (Italy), and Scotland and London (UK). Tables show national employment figures (in thousands) in years 2014-2018 across different demographic variables. Each file is named by following the pattern:

```
{region_code}_{variable_name}_k.csv
```

We examined the same variables as for the national data; the examined regions and their codes are listed below:

| region_code   | Region name (Country)   |  
|:---------------|:---------------|
| 10   | Île de France (France) |   
| C4  | Lombardia (Italy) |   
| I0 | London (UK) |
| I4 | Lazio (Italy) |
| M0 | Scotland (UK)  |

&nbsp;  
**`regional_demo_prop_breakdown/`**

Contains the same data as `regional_demo_count_breakdown/` but expressed as percentages. The files follow a similar naming pattern as shown above.

<a name="isco"></a>
## Employment by ISCO occupational groups

**`national_count_isco/`**

Contains tables with national employment figures (in thousands) for each three-digit ISCO minor occupational group, in years 2014-2018.

| Column name   | Description   |  
|:---------------|:---------------|
|isco_code   | Three-digit ISCO-08 code of the minor group. Find more information about ISCO on [ilo.org](https://www.ilo.org/public/english/bureau/stat/isco/isco08/). |
| 2014 - 2018 | Employment (in thousands). |
|isco_minor_group | Title of the occupational group. |
| RISK_CAT| [Automation risk](https://github.com/nestauk/mapping-career-causeways/tree/main/supplementary_online_data/automation_risk/) category of the ISCO minor group (0=high risk, 1=low risk, and 2=other). |

&nbsp;  
**`regional_count_isco/`**

Contains tables with regional employment figures (in thousands) for the regions listed above, for each three-digit ISCO minor occupational group. For the titles and risk categories of ISCO minor groups, see the tables in `national_count_isco/`.

| Column name   | Description   |  
|:---------------|:---------------|
|isco_code   | Three-digit ISCO-08 code of the minor group.|
| 2014 - 2018 | Employment (in thousands) in the years 2014-2018.  |
