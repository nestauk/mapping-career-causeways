# Measuring occupation similarity

To measure the fit between different occupations, we compared the occupational profiles across four different facets: essential skills, optional skills, work activity types and work context. The outputs of these notebooks are similarity matrices, which are stored in the `codebase/data/processed/sim_matrices` folder.

For more background on the methodology, see pages 85-93 of the project [research report](https://www.nesta.org.uk/report/mapping-career-causeways-supporting-workers-risk/). Note that while here we calculate distances between all 2942 ESCO occupations, in the report we mainly focused on a subset of transitions pertaining to the "top level" 1627 ESCO occupations.

**`Compare_occupations_by_ESCO_skills.ipynb`**  
Use NLP-adjusted overlap to make flexible comparisons of ESCO occupations' skills sets and arrive at the first two occupation similarity measures based on essential skills and optional skills.

**`Compare_occupations_by_ESCO_work_activities.ipynb`**  
Compare ESCO occupations based on their alignment of typical work activities. Here, we're using the recently released official ESCO skills hierarchy whose categories are related to the intermediate work activities of O*NET.

**`Compare_occupations_by_ONET_Work_context.ipynb`**  
Compare ESCO occupations based on their interpersonal, physical, and structural work aspects - these aspects are inferred from the O*NET work context features that are linked to ESCO occupations via our crosswalk.

**`Combine_similarity_measures.ipynb`**  
Combine the four similarity measures, and inspect the properties of the combined similarity measure.
