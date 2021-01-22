#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16, 2020

@author: karliskanders

"""

import pandas as pd
import numpy as np
import pickle
import os
import mapping_career_causeways

## Paths
useful_paths = mapping_career_causeways.Paths()
processed_folder = useful_paths.data_dir + 'processed/'

class Data:
    """
    Utility class for loading and inspecting occupation and skills data.
    """

    def __init__(self, data_folder=processed_folder):

        # Path to the processed data folder
        self.dir = data_folder

        # Occupations and skills
        self._occupations = None
        self._occupation_hierarchy = None
        self._skills = None
        self._concepts = None
        self._occupation_to_skills = None
        self._node_to_essential_items = None
        self._node_to_all_items = None
        self._isco_titles = None

        # Linked data
        self._occ_risk = None
        self._occ_jobzones = None
        self._occ_earnings_and_hours = None
        self._occ_remote = None
        self._occ_exposure = None
        self._occ_employment = None

        # Skills-based sectors and sub-sectors
        self._occ_clusters = None
        self._clusters_level_1 = None
        self._clusters_level_2 = None
        self._skills_based_sectors = None
        self._sub_sectors = None

        # Master tables
        self.default_master_table = f'{self.dir}ESCO_occupation_profiles.csv'
        self._occ = None
        self._occ_top = None
        self._top_occ_ids = None
        self._occ_report = None
        self._report_occ_ids = None

    ### Helper functions
    def read_csv(self, path):
        """ Read in csv files """
        if os.path.exists(path):
            return pd.read_csv(path)
        else:
            print(f'{path} does not exist')
            return None

    def describe_occupation(self, job_i):
        """ Show occupation's description and alternative labels """
        print(f'occupation ID {job_i}: {self.occupations.loc[job_i].preferred_label}')
        print('\n---\nDescription:\n')
        print(self.occupations.loc[job_i].description)
        print('\n---\nAlternative occupation labels:\n')
        print(self.occupations.loc[job_i].alt_labels)

    def occupation_skills(self, job_i, skill_importance=None):
        """ Show occupation's skills """

        df = self.occupation_to_skills[self.occupation_to_skills.occupation_id==job_i]

        # Skill importance filter may be equal to None, 'Essential' or 'Optional'
        if type(skill_importance) != type(None):
            df = df[df.importance==skill_importance]
        df = df.merge(self.skills[['id', 'preferred_label', 'description']],
                      left_on='skill_id', right_on='id', how='left')
        df = df.drop('id', axis=1)
        return df.reset_index(drop=True)

    def occ_title_to_id(self, occ_title):
        """
        Finds occupation's ID based on a string input;
        This could be later enhanced by allowing imprecise inputs
        """
        if type(occ_title) == str:
            return self.occupations[self.occupations.preferred_label==occ_title].iloc[0].id
        else:
            return occ_title

    def skill_title_to_id(self, skill_title):
        """
        Finds skill's ID based on a string input;
        This could be later enhanced by allowing imprecise inputs
        """
        if type(skill_title) == str:
            return self.skills[self.skills.preferred_label==skill_title].iloc[0].id
        else:
            return skill_title

    def add_field_to_occupation(self, df, occupation_id_col='id',  field_name=None, new_prefix=None):
        """ Merges field 'field_name' from data.occ to the dataframe 'df', and optionally adds a prefix """
        df_new = df.copy().merge(self.occ[['id', field_name]], left_on=occupation_id_col, right_on='id', how='left')
        if new_prefix is not None:
            df_new = df_new.rename(columns = {field_name: new_prefix + '_' + field_name})
        if occupation_id_col != 'id':
            df_new = df_new.drop('id', axis=1)
        return df_new

    def add_field_to_skill(self, df, skill_id_col='id',  field_name=None, new_prefix=None):
        """ Merges field 'field_name' from data.skills to the dataframe 'df', and optionally adds a prefix """
        df_new = df.copy().merge(self.skills[['id', field_name]], left_on=skill_id_col, right_on='id', how='left')
        if new_prefix is not None:
            df_new = df_new.rename(columns = {field_name: new_prefix + '_' + field_name})
        if skill_id_col != 'id':
            df_new = df_new.drop('id', axis=1)
        return df_new

    ### Occupations and skills
    @property
    def occupations(self):
        """ ESCO occupations with their full descriptions """
        if self._occupations is None:
            self._occupations = self.read_csv(self.dir + 'ESCO_occupations.csv')
        return self._occupations

    @property
    def occupation_hierarchy(self):
        """ ESCO occupations and their place in the ISCO-ESCO hierarchy """
        if self._occupation_hierarchy is None:
            self._occupation_hierarchy = self.read_csv(self.dir + 'ESCO_occupational_hierarchy/ESCO_occupational_hierarchy.csv')
        return self._occupation_hierarchy

    @property
    def skills(self):
        """ ESCO skills and their place in the skills hierarchy """
        if self._skills is None:
            self._skills = self.read_csv(self.dir + 'ESCO_skills_hierarchy/ESCO_skills_hierarchy.csv')
        return self._skills

    @property
    def concepts(self):
        """ ESCO skills hierarchy categories and their relationships """
        if self._concepts is None:
            self._concepts = self.read_csv(self.dir + 'ESCO_skills_hierarchy/ESCO_skills_concepts_hierarchy.csv')
        return self._concepts

    @property
    def occupation_to_skills(self):
        """ Links between ESCO occupations and skills, and indication of whether the skills are Essential or Optional  """
        if self._occupation_to_skills is None:
            self._occupation_to_skills = self.read_csv(self.dir + 'ESCO_occupation_to_skills.csv')
        return self._occupation_to_skills

    @property
    def node_to_essential_items(self):
        """
        Dataframe with essential skills IDs for each occupation,
        formatted for NLP adjusted overlap measurements
        """
        if self._node_to_essential_items is None:
            self._node_to_essential_items = pickle.load(open(self.dir + 'occupation_to_essential_skills.pickle','rb'))
        return self._node_to_essential_items

    @property
    def node_to_all_items(self):
        """
        Dataframe with essential and optional skills IDs for each occupation,
        formatted for NLP adjusted overlap measurement
        """
        if self._node_to_all_items is None:
            self._node_to_all_items = pickle.load(open(self.dir + 'occupation_to_all_skills.pickle','rb'))
        return self._node_to_all_items

    @property
    def isco_titles(self):
        """ ISCO occupational category codes and titles """
        if self._isco_titles is None:
            self._isco_titles = self.read_csv(self.dir + 'ISCO_occupation_titles.csv')
        return self._isco_titles

    ### Linked data ###

    @property
    def occ_risk(self):
        """
        ESCO occupations and estimates of their exposure to automation risks
        """
        if self._occ_risk is None:
            self._occ_risk = self.read_csv(self.dir + 'ESCO_automation_risk_full.csv')
        return self._occ_risk

    def occ_risk_report(self):
        """
        ESCO occupations and estimates of their exposure to automation risks
        """
        if self._occ_risk_report is None:
            self._occ_risk_report = self.read_csv(self.dir + 'ESCO_automation_risk.csv')
        return self._occ_risk_report

    @property
    def occ_jobzones(self):
        """
        ESCO occupations and their Job Zones, and levels of education, related
        work experience and on the job training (inferred from O*NET via our crosswalk)
        """
        if self._occ_jobzones is None:
            self._occ_jobzones = self.read_csv(self.dir + 'linked_data/ESCO_occupations_Job_Zones.csv')
        return self._occ_jobzones

    @property
    def occ_earnings_and_hours(self):
        """
        ESCO occupations and their estimates of annual earnings and paid hours
        (estimates pertain to the UK and were inferred from the ASHE tables)
        """
        if self._occ_earnings_and_hours is None:
            self._occ_earnings_and_hours = self.read_csv(self.dir + 'linked_data/ESCO_occupations_UK_earnings_and_hours_imputed.csv')
        return self._occ_earnings_and_hours

    @property
    def occ_employment(self):
        """
        Estimates of employment for of ESCO occupations in the UK
        (Note: only for the 'top level' 1627 occupations)
        """
        if self._occ_employment is None:
            self._occ_employment = self.read_csv(self.dir + 'linked_data/ESCO_top_occupations_UK_employment.csv')
        return self._occ_employment

    @property
    def occ_remote(self):
        """
        ESCO occupations and their Remote Labor Index
        """
        if self._occ_remote is None:
            self._occ_remote = self.read_csv(self.dir + 'linked_data/ESCO_occupations_Remote_Labor_Index.csv')
        return self._occ_remote

    @property
    def occ_exposure(self):
        """
        ESCO occupations and estimates of their potential exposure to impacts from COVID-19
        """
        if self._occ_exposure is None:
            self._occ_exposure = self.read_csv(self.dir + 'linked_data/ESCO_occupations_COVID_Exposure.csv')
        return self._occ_exposure

    ### Skills-based sectors and sub-sectors ###

    @property
    def occ_clusters(self):
        """ ESCO occupations and their skills-based sectors and sub-sectors """
        if self._occ_clusters is None:
            self._occ_clusters = self.read_csv(self.dir + 'clusters/ESCO_occupation_clusters_v1_1_curated.csv')
        return self._occ_clusters

    @property
    def clusters_level_1(self):
        """ Skills-based sectors, characterised by integer labels, automatic keywords and manual labels """
        if self._clusters_level_1 is None:
            self._clusters_level_1 = self.read_csv(self.dir + 'clusters/ESCO_occupation_clusters_v1_1_LEVEL1.csv')
        return self._clusters_level_1

    @property
    def clusters_level_2(self):
        """ Skills-based sub-sectors, characterised by integer labels, automatic keywords and manual labels """
        if self._clusters_level_2 is None:
            self._clusters_level_2 = self.read_csv(self.dir + 'clusters/ESCO_occupation_clusters_v1_1_LEVEL2.csv')
        return self._clusters_level_2

    @property
    def skills_based_sectors(self):
        """ Same information as in clusters_level_1 in a more polished form """
        if self._skills_based_sectors is None:
            self._skills_based_sectors = self.occ_clusters.drop_duplicates(['level_1']).sort_values('level_1')[[
                'skills_based_sector_code', 'skills_based_sector','level_1']].reset_index(drop=True)
        return self._skills_based_sectors

    @property
    def sub_sectors(self):
        """ Same information as in clusters_level_2 in a more polished form """
        if self._sub_sectors is None:
            self._sub_sectors = self.occ_clusters.drop_duplicates(['level_1', 'level_2']).sort_values(['level_1', 'level_2'])[[
                'sub_sector_code', 'sub_sector', 'skills_based_sector_code', 'skills_based_sector', 'level_1', 'level_2']].reset_index(drop=True)
        return self._sub_sectors

    ### Master tables containing full occupational profiles ###

    def generate_occupation_profiles(self, export=True, save_path = None):
        """ Joins up relevant datasets to create occupational profiles """

        merge_params = {'on':'id', 'how':'left'}
        occ = self.occupation_hierarchy[['id', 'concept_type', 'concept_uri', 'preferred_label',
                                         'isco_level_1', 'isco_level_2','isco_level_3', 'isco_level_4', 'is_top_level']].copy()
        occ = occ.merge(self.occ_jobzones[['id', 'job_zone','education_level', 'related_work_experience', 'on_the_job_training']], **merge_params)
        occ = occ.merge(self.occ_earnings_and_hours[['id', 'annual_earnings', 'total_paid_hours']], **merge_params)
        occ = occ.merge(self.occ_employment[['id', 'employment_count', 'employment_share']], **merge_params)
        occ = occ.merge(self.occ_risk[['id', 'risk', 'prevalence', 'risk_category', 'onet_code', 'onet_occupation']], **merge_params)
        occ = occ.merge(self.occ_clusters[['id', 'level_1','level_2', 'skills_based_sector_code', 'sub_sector_code', 'skills_based_sector', 'sub_sector']], **merge_params)
        occ = occ.merge(self.occ_remote[['id','remote_labor_index']], **merge_params)
        occ = occ.merge(self.occ_exposure[['id', 'physical_proximity', 'exposure_score']], **merge_params)

        if export:
            if save_path is None:
                save_path = self.default_master_table
            occ.to_csv(save_path, index=False)
        return occ

    @property
    def occ(self):
        """ All ESCO occupations (n=2942) """
        if self._occ is None:
            self._occ = self.read_csv(self.default_master_table)
        return self._occ

    @property
    def occ_top(self):
        """ Master table of all 'top level' ESCO occupations (approx. 1700) """
        if self._occ_top is None:
            self._occ_top = self.occ[self.occ.is_top_level==True].copy().reset_index(drop=True)
        return self._occ_top

    @property
    def occ_report(self):
        """ Master table of the 'top level' ESCO occupations analysed in the report (n=1627) """
        if self._occ_report is None:
            df = self.read_csv(self.dir + 'ESCO_automation_risk.csv')
            self._occ_report = self.occ[self.occ.id.isin(df.id.to_list())].copy().reset_index(drop=True)
        return self._occ_report

    @property
    def top_occ_ids(self):
        """ Integer identifiers of the approx. 1700 top ESCO occupations """
        if self._top_occ_ids is None:
            self._top_occ_ids = self.occ_top.id.to_list()
        return self._top_occ_ids

    @property
    def report_occ_ids(self):
        """ Integer identifiers of the 1627 occupations analysed in the report """
        if self._report_occ_ids is None:
            self._report_occ_ids = self.occ_report.id.to_list()
        return self._report_occ_ids




class Similarities:
    """
    Utility class for loading occupation similary measures
    """

    def __init__(self, data_folder=processed_folder):
        # Path to the processed data folder
        self.dir = data_folder

        self._W_combined = None
        self._W_essential = None
        self._W_all_to_essential = None
        self._W_activities = None
        self._W_work_context = None

    def select_similarity_matrix(self, similarity_measure='combined'):
        if similarity_measure == 'combined':
            return self.W_combined
        elif similarity_measure == 'essential_skills':
            return self.W_essential
        elif similarity_measure == 'optional_skills':
            return self.W_all_to_essential
        elif similarity_measure == 'work_activities':
            return self.W_activities
        elif similarity_measure == 'work_context':
            return self.W_work_context

    @staticmethod
    def select_subset(W, subset_ids):
        """
        Select a subset of the full similarity matrix
        """
        W_subset = W[subset_ids, :].copy()
        W_subset = W_subset[:, subset_ids]
        return W_subset

    @property
    def W_combined(self):
        """
        Combined occupation similarities
        """
        if self._W_combined is None:
            self._W_combined = np.load(self.dir + 'sim_matrices/OccupationSimilarity_Combined.npy')
        return self._W_combined

    @property
    def W_essential(self):
        """
        'Essential skills' similarities: Occcupation similarities based on the NLP-adjusted overlap of essential skills
        """
        if self._W_essential is None:
            self._W_essential = np.load(self.dir + 'sim_matrices/OccupationSimilarity_EssentialSkillsDescription_asymmetric.npy')
        return self._W_essential

    @property
    def W_all_to_essential(self):
        """
        'Optional skills' similarities: Occupation similarities that include optional skills as well; specifically, the NLP-adjusted
        overlap of essential and optional skills at the origin occupation vs. essential skills at the destination occupation.
        """
        if self._W_all_to_essential is None:
            self._W_all_to_essential = np.load(self.dir + 'sim_matrices/OccupationSimilarity_AllToEssentialSkillsDescription_asymmetric.npy')
        return self._W_all_to_essential

    @property
    def W_activities(self):
        """
        'Work activities' similarities: Occupation similarities based on the categories of ESCO skills categories,
        that pertain to different types of work activities
        """
        if self._W_activities is None:
            self._W_activities = np.load(self.dir + 'sim_matrices/OccupationSimilarity_ESCO_clusters_Level_2.npy')
        return self._W_activities

    @property
    def W_work_context(self):
        """
        'Work context' similarities: Similarities based on ONET's work context features capturing structural, interpersonal,
        and physical work aspects.
        """
        if self._W_work_context is None:
            self._W_work_context = np.load(self.dir + 'sim_matrices/OccupationSimilarity_ONET_Work_Context.npy')
        return self._W_work_context
