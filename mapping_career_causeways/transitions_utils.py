#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in September 2020

@author: karliskanders

Helper functions to generate transition recommendations
"""

import pandas as pd
import numpy as np
import pickle
from time import time
import yaml

import  mapping_career_causeways
import  mapping_career_causeways.compare_nodes_utils as compare_nodes_utils
import  mapping_career_causeways.load_data_utils as load_data
find_closest = compare_nodes_utils.find_closest

useful_paths = mapping_career_causeways.Paths()
data = load_data.Data()
sim = load_data.Similarities()

### SET UP DEFAULT TRANSITION FILTERING CRITERIA ###

with open(f'{useful_paths.codebase_dir}configs/default_transition_params.yaml', 'r') as f:
    def_transition_params = yaml.load(f, Loader=yaml.FullLoader)

# Viability: Similarity threshold for viable transitions (default = 0.3)
MIN_VIABLE_DEF = def_transition_params['MIN_VIABLE']
# Viability: Similarity threshold for highly viable transitions (default = 0.4)
HIGHLY_VIABLE_DEF = def_transition_params['HIGHLY_VIABLE']
# Viability: Max absolute difference in job zones (default = 1)
MAX_JOB_ZONE_DIF_DEF = def_transition_params['MAX_JOB_ZONE_DIF']
# Desirability: Threshold for differences in earnings (default = 0.75)
MIN_EARNINGS_RATIO_DEF = def_transition_params['MIN_EARNINGS_RATIO']

def occupations_to_check(id_to_check):
    """ Helper function for selecting list of occupations for checking"""
    if (type(id_to_check)==type(None)) or (id_to_check=='report'):
        id_to_check = data.report_occ_ids
    elif id_to_check == 'top':
        id_to_check = data.top_occ_ids
    elif id_to_check == 'all':
        id_to_check = data.occ.id.to_list()
    return id_to_check

def get_transitions(
    origin_ids = None,
    MIN_VIABLE = MIN_VIABLE_DEF,
    HIGHLY_VIABLE = HIGHLY_VIABLE_DEF,
    MAX_JOB_ZONE_DIF = MAX_JOB_ZONE_DIF_DEF,
    MIN_EARNINGS_RATIO = MIN_EARNINGS_RATIO_DEF,
    destination_ids = None):

    """
    Function to find transitions according to the specified filters

    Parameters
    ----------
    origin_ids (list of int):
        List of origin occupation IDs, for which to check the transitions. If None,
        we only check the subset of occupations analysed in the report
    MIN_VIABLE (float):
        Similarity threshold for viable transitions (default = 0.3)
    HIGHLY_VIABLE (float):
        Similarity threshold for highly viable transitions (default = 0.4)
    MAX_JOB_ZONE_DIF (int):
        Max absolute difference in job zones (default = 1)
    MIN_EARNINGS_RATIO (float):
        Threshold for differences in earnings (default = 0.75)
    destination_ids (list of int):
        List of permissible destination occupation IDs. If None, we check only
        the occupations subset analysed in the report
    """

    columns = {
        'origin_id': [],
        'origin_label': [],
        'destination_id': [],
        'destination_label': [],
        'similarity': [],
        'is_jobzone_ok': [],
        'is_earnings_ok': [],
        'is_not_high_risk': [],
        'is_safer': [],
        'is_strictly_safe': [],
        'job_zone_dif': [],
        'earnings_ratio': [],
        'risk_dif': [],
        'prop_dif': [],
        'W_skills': [],
        'W_work': [],
        'W_essential_skills': [],
        'W_optional_skills': [],
        'W_activities': [],
        'W_work_context': []
    }

    origin_ids = occupations_to_check(origin_ids)
    destination_ids = occupations_to_check(destination_ids)

    # For each occupation in consideration...
    print('Finding all transitions...', end=' ')
    t_now = time()
    for j, j_id in enumerate(origin_ids):

        # Find the most similar occupations
        df = find_closest(j_id, sim.W_combined, data.occ[['id']])

        # Filter out self
        df = df[df.id!=j_id]

        # Filter out occupations that we're not supposed to check
        df = df[df.id.isin(destination_ids)]

        # Filter out non-viable transitions
        df = df[df.similarity > MIN_VIABLE]

        # Viable IDs
        viable_ids = df.id.to_list()
        N = len(viable_ids)

        ### FILTERS
        # For viability, obtain: job_zone
        # For desirability, obtain: annual_earnings
        # For safety, obtain risk, prevalence & risk_category

        origin_job_zone = data.occ.loc[j_id].job_zone
        origin_earnings = data.occ.loc[j_id].annual_earnings
        origin_risk = data.occ.loc[j_id].risk
        origin_prevalence = data.occ.loc[j_id].prevalence
        origin_label = data.occ.loc[j_id].risk_category

        job_zone_dif = origin_job_zone - data.occ.loc[viable_ids].job_zone
        earnings_ratio = data.occ.loc[viable_ids].annual_earnings / origin_earnings
        risk_dif = origin_risk - data.occ.loc[viable_ids].risk
        prevalence_dif = data.occ.loc[viable_ids].prevalence - origin_prevalence

        # Job Zone difference not larger than MAX_JOB_ZONE_DIF
        is_jobzone_ok = np.abs(job_zone_dif) <= MAX_JOB_ZONE_DIF
        # Earnings at destination larger than MIN_EARNINGS_RATIO
        is_earnings_ok = earnings_ratio > MIN_EARNINGS_RATIO
        # Destination is not a high risk occupation
        is_not_high_risk = (data.occ.loc[viable_ids].risk_category != 'High risk')
        # Destination has a smaller risk and a larger prevalence of bottleneck tasks
        is_safer = (risk_dif > 0) & (prevalence_dif > 0)
        # Combine both safety filters
        is_strictly_safe = is_safer & is_not_high_risk

        # Summarise similarities
        W_skills = 0.5*sim.W_essential[j_id, viable_ids] + 0.5*sim.W_all_to_essential[j_id, viable_ids]
        W_work = 0.5*sim.W_activities[j_id, viable_ids] + 0.5*sim.W_work_context[j_id, viable_ids]

        # Save the row data
        columns['origin_id'] += [j_id] * N
        columns['origin_label'] += [data.occ.loc[j_id].preferred_label] * N
        columns['destination_id'] += viable_ids
        columns['destination_label'] += data.occ.loc[viable_ids].preferred_label.to_list()
        columns['similarity'] += df.similarity.to_list()

        columns['is_jobzone_ok'] += list(is_jobzone_ok)
        columns['is_earnings_ok'] += list(is_earnings_ok)
        columns['is_not_high_risk'] += list(is_not_high_risk)
        columns['is_safer'] += list(is_safer)
        columns['is_strictly_safe'] += list(is_strictly_safe)

        columns['job_zone_dif'] += list(job_zone_dif)
        columns['earnings_ratio'] += list(earnings_ratio)
        columns['risk_dif'] += list(risk_dif)
        columns['prop_dif'] += list(prevalence_dif)

        columns['W_skills'] += list(W_skills)
        columns['W_work'] += list(W_work)

        columns['W_essential_skills'] += list(sim.W_essential[j_id, viable_ids])
        columns['W_optional_skills'] += list(sim.W_all_to_essential[j_id, viable_ids])
        columns['W_activities'] += list(sim.W_activities[j_id, viable_ids])
        columns['W_work_context'] += list(sim.W_work_context[j_id, viable_ids])

    print(f'Done!\nThis took {(time()-t_now):.2f} seconds.')

    # Collect everything
    trans_df = pd.DataFrame(data=columns)

    # Transition viability category
    trans_df['sim_category'] = ''
    trans_df.loc[trans_df.similarity <= HIGHLY_VIABLE, 'sim_category'] = 'min_viable'
    trans_df.loc[trans_df.similarity > HIGHLY_VIABLE, 'sim_category'] = 'highly_viable'

    trans_df['is_viable'] = trans_df['is_jobzone_ok']
    trans_df['is_desirable'] = trans_df['is_viable'] & trans_df['is_earnings_ok']
    trans_df['is_safe_desirable'] = trans_df['is_desirable'] & trans_df['is_not_high_risk']
    trans_df['is_safer_desirable'] = trans_df['is_desirable'] & trans_df['is_strictly_safe']

    return trans_df

def create_filtering_matrices(
    origin_ids = None,
    MIN_VIABLE = MIN_VIABLE_DEF,
    HIGHLY_VIABLE = HIGHLY_VIABLE_DEF,
    MAX_JOB_ZONE_DIF = MAX_JOB_ZONE_DIF_DEF,
    MIN_EARNINGS_RATIO = MIN_EARNINGS_RATIO_DEF,
    destination_ids = None):

    """
    !!! CAUTION: NOT YET REVIEWED !!!

    Function to find transitions according to the specified filters

    Parameters
    ----------
    origin_ids (list of int):
        List of origin occupation IDs, for which to check the transitions. If None,
        we only check the subset of occupations analysed in the report
    MIN_VIABLE (float):
        Similarity threshold for viable transitions (default = 0.3)
    HIGHLY_VIABLE (float):
        Similarity threshold for highly viable transitions (default = 0.4)
    MAX_JOB_ZONE_DIF (int):
        Max absolute difference in job zones (default = 1)
    MIN_EARNINGS_RATIO (float):
        Threshold for differences in earnings (default = 0.75)
    destination_ids (list of int):
        List of permissible destination occupation IDs. If None, we check only
        the occupations subset analysed in the report
    """

    # Select the occupations to check
    origin_ids = occupations_to_check(origin_ids)
    destination_ids = occupations_to_check(destination_ids)

    # Select the similarities corresponding to the specified occupations
    W_combined_select = sim.W_combined[origin_ids, :].copy()
    W_combined_select = W_combined_select[:, destination_ids]

    # Filter matrices
    N = len(origin_ids)
    N2 = len(destination_ids)

    # Matrices to inidicate...
    # ...compatibility of job zones
    F_jobzone = np.zeros((N,N2)).astype(bool)
    # ...compatability of earnings
    F_earnings = np.zeros((N,N2)).astype(bool)
    # ...reduction of risk and increase of the prevalence of bottleneck tasks
    F_safer = np.zeros((N,N2)).astype(bool)
    # ...that destination is not of high risk
    F_not_high_risk = np.zeros((N,N2)).astype(bool)

    F_not_self = np.zeros((N,N2)).astype(bool)

    print('Creating filtering matrices...', end=' ')
    t_now = time()
    for i in range(N):
        row_i = data.occ.iloc[origin_ids[i]]

        for j in range(N2):
            row_j = data.occ.iloc[destination_ids[j]]
            is_jobzone_ok = np.abs(row_i.job_zone - row_j.job_zone) <= MAX_JOB_ZONE_DIF
            is_earnings_ok = (row_j.annual_gross_pay / row_i.annual_gross_pay) > MIN_EARNINGS_RATIO
            is_safer = (row_i.risk > row_j.risk) & (row_i.prop_bottleneck_tasks < row_j.prop_bottleneck_tasks)
            is_not_high_risk = (row_j.risk_cat_label != 'High risk')

            F_jobzone[i][j] = is_jobzone_ok
            F_earnings[i][j] = is_earnings_ok
            F_not_high_risk[i][j] = is_not_high_risk
            F_safer[i][j] = is_safer

            F_not_self[i][j] = row_i.id != row_j.id

    print(f'Done!\nThis took {(time()-t_now):.2f} seconds.')

    # Matrices indicating viable and highly viable transitions
    F_viable = F_jobzone & (W_combined_select > MIN_VIABLE)
    F_highly_viable = F_jobzone & (W_combined_select > HIGHLY_VIABLE)
    F_min_viable = F_jobzone & (W_combined_select > MIN_VIABLE) & (W_combined_select <= HIGHLY_VIABLE)

    # Matrix indicating viable and desirable transitions
    F_desirable = F_viable & F_earnings

    # Matrix indicating safe transitions
    F_strictly_safe = F_safer & F_not_high_risk

    # Matrices indicating safe and desirable transitions
    F_safe_desirable = F_desirable & F_not_high_risk # 1st definition
    F_safer_desirable = F_desirable & F_strictly_safe # 2nd (stricter) definition

    # Export filtering matrices
    filter_matrices = {
        'F_viable': F_viable,
        'F_min_viable': F_min_viable,
        'F_highly_viable': F_highly_viable,
        'F_desirable': F_desirable,
        'F_jobzone': F_jobzone,
        'F_earnings': F_earnings,
        'F_not_high_risk': F_not_high_risk,
        'F_safer': F_safer,
        'F_full_safe': F_full_safe,
        'F_not_self': F_not_self,
        'F_safe_desirable': F_safe_desirable,
        'F_safer_desirable': F_safer_desirable,
    }

    # Remove transitions to self
    for key in list(filter_matrices.keys()):
        filter_matrices[key] = filter_matrices[key] & F_not_self

    filter_matrices['origin_ids'] = origin_ids
    filter_matrices['destination_ids'] = destination_ids

    return filter_matrices

def show_skills_overlap(
    job_i,
    job_j,
    data, sim,
    embeddings,
    skills_match = 'optional',  # either 'optional' or 'essential'
    matching_method='one_to_one',
    verbose=True):

    """
    NLP-adjusted overlap of skill sets between occupations job_i and job_j
    """

    if verbose: print(f"from {data.occ.loc[job_i].preferred_label} to {data.occ.loc[job_j].preferred_label}")

    # Create the input dataframe in the required format
    if skills_match == 'optional':
        node_to_items_ = pd.concat([data.node_to_all_items.loc[[job_i]],
                                    data.node_to_essential_items.loc[[job_j]]])
        w = sim.W_all_to_essential[job_i, job_j]
    elif skills_match == 'essential':
        node_to_items_ = pd.concat([data.node_to_essential_items.loc[[job_i]],
                                    data.node_to_essential_items.loc[[job_j]]])
        w = sim.W_essential[job_i, job_j]

    # Check for empty arrays
    assert((data.node_to_essential_items.loc[[job_j]].items_list.values[0]) != 0)

    # Compare occupations
    df, score = compare_nodes_utils.two_node_comparison(
        node_to_items_, job_i, job_j,
        data.skills[['id','preferred_label']],
        embeddings,
        metric='cosine',
        matching_method=matching_method,
        symmetric=False)

    N_matched = len(df)

    # Tidy up the dataframe
    df.rename(columns={
        'id_x': 'origin_skill_id',
        'preferred_label_x': 'origin_skill',
        'id_y': 'destination_skill_id',
        'preferred_label_y': 'destination_skill',
        'similarity': 'score',
        'similarity_raw': 'similarity'}, inplace=True)
    df = df[['origin_skill_id', 'origin_skill',
             'destination_skill_id', 'destination_skill',
             'similarity', 'score']]

    # Add leftover skills from the destination occupation
    all_destination_skills = data.occupation_to_skills[
        (data.occupation_to_skills.occupation_id==job_j) &
        (data.occupation_to_skills.importance=='Essential')].skill_id.to_list()
    skills_to_add = set(all_destination_skills).difference(set(df.destination_skill_id))

    if len(skills_to_add) != 0:
        append_df = {
            'origin_skill_id':[],
            'origin_skill':[],
            'destination_skill_id':[],
            'destination_skill':[],
            'similarity':[],
            'score':[]
        }

        for s in skills_to_add:
            append_df['origin_skill_id'].append('-')
            append_df['origin_skill'].append('-')
            append_df['destination_skill_id'].append(s)
            append_df['destination_skill'].append(data.skills.loc[s].preferred_label)
            append_df['similarity'].append(0)
            append_df['score'].append(0)
        df = df.append(pd.DataFrame(data=append_df), ignore_index=True)

    if verbose:
        print('--------')
        print(f'{N_matched}/{len(data.node_to_essential_items.loc[[job_j]].items_list.values[0])} destination skills matched')
        print(f'NLP-adjusted overlap = {w:.2f} (total combined similarity: {sim.W_combined[job_i, job_j]:.2f})')

    return df

class CompareFeatures():
    """
    Class to inspect feature vector differences between occupations
    """

    def __init__(self, data_folder=useful_paths.data_dir):

        ### Import work context vectors ###

        self.work_context_vectors = np.load(data_folder + 'interim/work_context_features/ESCO_work_context_vectors.npy')
        self.work_context_features = pd.read_csv(data_folder + 'processed/work_context_vector_features.csv')
        self.work_context_features['category'] = self.work_context_features.element_id.apply(lambda x: int(x[4]))

        # Add work context feature category label
        def categorise(x):
            if x == 1: return 'interpersonal'
            if x == 2: return 'physical'
            if x == 3: return 'structural'
        self.work_context_features['category'] = self.work_context_features['category'].apply(lambda x: categorise(x))

        ### Import ESCO skills category vectors ###

        self.esco_vectors_1 = np.load(data_folder + 'interim/work_activity_features/esco_hierarchy_vectors_level_1.npy')
        self.esco_features_1 = pickle.load(open(data_folder + 'interim/work_activity_features/esco_hierarchy_codes_level_1.pickle', 'rb'))

        self.esco_vectors_2 = np.load(data_folder + 'interim/work_activity_features/esco_hierarchy_vectors_level_2.npy')
        self.esco_features_2 = pickle.load(open(data_folder + 'interim/work_activity_features/esco_hierarchy_codes_level_2.pickle', 'rb'))

        self.esco_vectors_3 = np.load(data_folder + 'interim/work_activity_features/esco_hierarchy_vectors_level_3.npy')
        self.esco_features_3 = pickle.load(open(data_folder + 'interim/work_activity_features/esco_hierarchy_codes_level_3.pickle', 'rb'))

    def get_work_context_deltas(self, job_i, job_j):
        # Calculate vector deltas and add category labels
        delta_vector = self.work_context_vectors[job_j] - self.work_context_vectors[job_i]
        df = self.work_context_features.copy()
        df['job_i'] = self.work_context_vectors[job_i]
        df['job_j'] = self.work_context_vectors[job_j]
        df['deltas'] = delta_vector
        df['deltas_abs'] = np.abs(delta_vector)
        return df.sort_values('deltas_abs', ascending=False)

    def get_esco_cluster_deltas(self, job_i, job_j, level=2):
        # Select the level of ESCO hierarchy
        if level==1:
            esco_vectors = self.esco_vectors_1
            esco_features = self.esco_features_1
        elif level==2:
            esco_vectors = self.esco_vectors_2
            esco_features = self.esco_features_2
        elif level==3:
            esco_vectors = self.esco_vectors_3
            esco_features = self.esco_features_3

        # Calculate vector deltas and add category labels
        delta_vector = esco_vectors[job_j] - esco_vectors[job_i]
        df = data.concepts[data.concepts.code.isin(esco_features)][['code','title']].sort_values('code').copy()
        df['job_i'] = esco_vectors[job_i]
        df['job_j'] = esco_vectors[job_j]
        df['deltas'] = delta_vector
        df['deltas_abs'] = np.abs(delta_vector)
        return df.sort_values('deltas_abs', ascending=False)
