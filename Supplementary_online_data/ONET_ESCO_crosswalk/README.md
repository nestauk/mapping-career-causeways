# Crosswalk between O\*NET and ESCO occupations

To combine the rich insights from [O\*NET](https://www.onetonline.org/) and [ESCO](https://ec.europa.eu/esco), we developed, to the best of our knowledge, the first direct crosswalk between these frameworks. The crosswalk establishes a mapping between O\*NET occupational codes and those in ESCO. While many of the mappings are one to one, there are instances where a single O*NET occupation is matched to more than one ESCO occupation.

To derive the crosswalk, we leveraged several strategies. First, we used an existing crosswalk between O\*NET and ISCO (i.e. broader occupational groups than ESCO) to identify the most likely matches. Second, we applied techniques from Natural Language Processing – a subfield of machine learning – to identify for each ESCO occupation corresponding O\*NET occupations with the highest semantic similarity of occupational group descriptors and known job titles. Finally, we manually assigned occupational codes in instances where the two automated approaches didn’t agree.

For further details on the methodology, consult the Appendix of the [Mapping Career Causeways](https://www.nesta.org.uk/project/mapping-career-causeways/) project report, as well as the `ONET_to_ESCO_crosswalk.ipynb` notebook. The development of this crosswalk was led by Jyldyz Djumalieva, with special thanks to Stef Garasto and Karlis Kanders.

Anyone is welcome to use and build upon the crosswalk; to refer to this crosswalk in your work, you can cite the Mapping Career Causeways report.

## Crosswalk files

**`esco_onet_crosswalk_Nov2020.csv`**

The most recent version of the crosswalk (officially released on November 23rd, 2020). It details the best matching O\*NET occupation for 1680 ESCO occupations. These occupations build up the top level of the ESCO hierarchy of occupations (the next intermediate level after ISCO four-digit unit groups). Other, lower level occupations (level 6 to level 8) may inherit the matching O\*NET code of their parent occupation.

| Column name   | Description   |  
|:---------------|:---------------|
|id   | Unique integer identifier of the ESCO occupation; used only internally, within the scope of this project. |   
|esco_occupation  | Preferred label of the ESCO occupation.   |   
|isco_code   | Four-digit ISCO-08 code, indicating the broader ISCO unit group to which the occupation belongs; the code is provided by the ESCO API. Find more information about ISCO on [ilo.org](https://www.ilo.org/public/english/bureau/stat/isco/isco08/).    |
|onet_code | O\*NET code pertaining to the ESCO occupation. |
|onet_occupation | Title of the O*NET occupation matched to the ESCO occupation. |
|matching_job_titles| List of the known job titles that both occupations have in common. These are inferred from the so-called 'alternative labels' of ESCO occupations (see `alt_labels` column in `lookups/esco_occup_level5.csv`) and 'alternate titles' of O\*NET (`lookups/Alternate Titles.xlsx`).
|semantic_similarity | Semantic similarity between the matched ESCO and O*NET occupations, based on the known job titles and occupation descriptions.  |
| confidence | Our level of confidence in the assigned match, with three distinct levels: 0.5, 1 and 2 (highest confidence). |
|concept_uri | Universal identifier of the ESCO occupation used by the ESCO API. Find more information in the [ESCO documentation](https://ec.europa.eu/esco/api/doc/esco_api_doc.html#rest-calls-get-conceptschemes-by-uris). |  

**`esco_onet_crosswalk_Nov2020.xlsx`**  
Excel version of the crosswalk specified in `esco_onet_crosswalk_Nov2020.csv`.

**`esco_onet_crosswalk_MCC.xlsx`**  
Version of the crosswalk that was used in the analyses of the Mapping Career Causeways project. It is practically identical to the crosswalk described above, with the sole exception being the mapping of the 'member of parliament' ESCO occupation.

**`ONET_to_ESCO_crosswalk.ipynb`**    
Jupyter notebook for generating pre-validated matches between ESCO and O\*NET occupations. The notebook also contains information about the other files and folders of this directory.

To save some compute time, you can download the intermediate outputs (such as pre-computed Sentence-BERT embeddings of job titles and descriptions) by running `download_outputs.py` script:

```shell
$ python download_outputs.py
```

The expected output:
```
Downloading outputs.zip (264.7 MB)... Done!
Extracting the archive in outputs/... Done!
```
