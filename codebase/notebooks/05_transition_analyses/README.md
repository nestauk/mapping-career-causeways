# Analyses of career transitions

Here, we use the calculated occupation similarities and our career transition
recommendation algorithm to analyse transition options.

## Clustering occupations into skills-based sectors

**`Clustering_01_ESCO_occupations.ipynb`**  
Applies graph-based clustering on occupation similarities to find and analyse groups
of related occupations that share similar job requirements and work characteristics.
The resulting grouping of occupations organises the 2942 ESCO occupations at two hierarchical levels, which we call skills-based sectors and sub-sectors.

**`Clustering_02_Manual_review.ipynb`**  
Notebook documenting the manual inspection, adjustment and labelling of the clusters.

## Transitions analysis

**`Transitions_01_Calibrate_viability_threshold`**  
Data-driven calibration of a threshold for viable and highly viable transitions.

**`Transitions_02_Number_of_options.ipynb`**  
Calculates the number of transition options for the 1627 occupations presented
in the Mapping Career Causeways report.

**`Transitions_03_Analysis.ipynb`**  
Reproduces the analyses in Part B of the Mapping Career Causeways report,
where we characterise the career transition options for workers in high-risk occupations.

## Skills gaps

**`Transitions_04_Skills_gaps.ipynb`**  
Identifies the most prevalent gaps across multiple transitions. Specifically,
showcases the skills gap analysis of transitions out of high-risk occupations
in the sales and services skills-based sector.

## Core skills and upskilling analysis  

**`Upskilling_01_Core_skills.ipynb`**   
Uses metrics from network science to identify 'core' ESCO skills that reflect
the central competencies for a wide range of jobs. The top 100 skills with the
highest 'coreness' are preselected for further evaluation.

**`Upskilling_02_Analysis.ipynb`**  
Identifies the most effective core skills, by evaluating the impact of adopting
a new core skill. This is done by adding them, one at a time, to
the skillset of each high-risk occupation and recalculating the similarities with
all other occupations.
