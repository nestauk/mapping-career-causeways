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
import os
from ast import literal_eval
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist, cosine
from scipy.stats import wilcoxon
from collections import defaultdict

import mapping_career_causeways
import mapping_career_causeways.compare_nodes_utils as compare_nodes_utils
import mapping_career_causeways.load_data_utils as load_data
from mapping_career_causeways.scripts import pickle_large_files
find_closest = compare_nodes_utils.find_closest

useful_paths = mapping_career_causeways.Paths()
data = load_data.Data()
sim = load_data.Similarities()

# Import default skills description embeddings
embeddings = np.load(f'{useful_paths.data_dir}interim/embeddings/embeddings_skills_description_SBERT.npy')

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
    destination_ids = None,
    verbose=False):

    """
    Function to find viable, desirable and safe transitions according to the specified filters;
    NB: This function outputs only transitions whose occupation similarity is above MIN_VIABLE threshold

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

    columns = initialise_transition_table_columns()

    origin_ids = occupations_to_check(origin_ids)
    destination_ids = occupations_to_check(destination_ids)

    # For each occupation in consideration...
    if verbose: print('Finding all transitions...', end=' ')
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
        # Collect data about each transition from j_id to viable_ids
        columns = transition_data_processing(
            columns, j_id, viable_ids,
            MIN_VIABLE,
            HIGHLY_VIABLE,
            MAX_JOB_ZONE_DIF,
            MIN_EARNINGS_RATIO)
    if verbose: print(f'Done!\nThis took {(time()-t_now):.2f} seconds.')
    trans_df = pd.DataFrame(data=columns)
    # Add filtering variables
    trans_df = transition_data_filtering(trans_df, MIN_VIABLE, HIGHLY_VIABLE)
    return trans_df.reset_index(drop=True)

def initialise_transition_table_columns():
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
    return columns

def transition_data_processing(
    columns, j_id, viable_ids,
    MIN_VIABLE = MIN_VIABLE_DEF,
    HIGHLY_VIABLE = HIGHLY_VIABLE_DEF,
    MAX_JOB_ZONE_DIF = MAX_JOB_ZONE_DIF_DEF,
    MIN_EARNINGS_RATIO = MIN_EARNINGS_RATIO_DEF):
    """
    FILTERS:
    - For viability, obtain: job_zone
    - For desirability, obtain: annual_earnings
    - For safety, obtain: risk, prevalence & risk_category
    """
    N = len(viable_ids)

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
    columns['similarity'] += list(sim.W_combined[j_id, viable_ids])

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

    return columns

def transition_data_filtering(trans_df, MIN_VIABLE, HIGHLY_VIABLE):
    """ Adds filtering variables """
    trans_df['sim_category'] = ''
    trans_df.loc[trans_df.similarity <= HIGHLY_VIABLE, 'sim_category'] = 'min_viable'
    trans_df.loc[trans_df.similarity > HIGHLY_VIABLE, 'sim_category'] = 'highly_viable'
    trans_df.loc[trans_df.similarity <= MIN_VIABLE, 'sim_category'] = 'not_viable'
    trans_df['is_viable'] = trans_df['is_jobzone_ok'] & (trans_df['sim_category'] != 'not_viable')
    trans_df['is_desirable'] = trans_df['is_viable'] & trans_df['is_earnings_ok']
    trans_df['is_safe_desirable'] = trans_df['is_desirable'] & trans_df['is_not_high_risk']
    trans_df['is_strictly_safe_desirable'] = trans_df['is_desirable'] & trans_df['is_strictly_safe']
    return trans_df

def get_transition_data(
    transition_pairs,
    MIN_VIABLE = MIN_VIABLE_DEF,
    HIGHLY_VIABLE = HIGHLY_VIABLE_DEF,
    MAX_JOB_ZONE_DIF = MAX_JOB_ZONE_DIF_DEF,
    MIN_EARNINGS_RATIO = MIN_EARNINGS_RATIO_DEF,
    verbose=False):
    """
    Adds transition data for each transition pair; final output table follows the same
    format as the output of get_transitions()
    """
    columns = initialise_transition_table_columns()

    if verbose: print('Finding data for all transitions...', end=' ')
    t_now = time()

    transition_pair_dict = defaultdict(list)
    for pair in transition_pairs:
        transition_pair_dict[pair[0]].append(pair[1])

    # For each transition pair in consideration...
    for j_id in list(transition_pair_dict.keys()):
        viable_ids = transition_pair_dict[j_id]
        columns = transition_data_processing(
            columns, j_id, viable_ids,
            MIN_VIABLE,
            HIGHLY_VIABLE,
            MAX_JOB_ZONE_DIF,
            MIN_EARNINGS_RATIO)
    if verbose: print(f'Done!\nThis took {(time()-t_now):.2f} seconds.')
    trans_df = pd.DataFrame(data=columns)
    trans_df = transition_data_filtering(trans_df, MIN_VIABLE, HIGHLY_VIABLE)
    return trans_df.reset_index(drop=True)

def create_filtering_matrices(
    origin_ids = None,
    MIN_VIABLE = MIN_VIABLE_DEF,
    HIGHLY_VIABLE = HIGHLY_VIABLE_DEF,
    MAX_JOB_ZONE_DIF = MAX_JOB_ZONE_DIF_DEF,
    MIN_EARNINGS_RATIO = MIN_EARNINGS_RATIO_DEF,
    destination_ids = None,
    export_path = None):

    """
    Creates boolean matrices for tagging transitions as 'safe', 'desirable', 'viable'
    'highly viable' and combinations of these.

    These boolean matrices are then used for analysing the number of different
    types of transitions for each occupation.

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

    # Boolean natrices to indicate...
    # ...compatibility of job zones
    F_jobzone = np.zeros((N,N2)).astype(bool)
    # ...compatability of earnings
    F_earnings = np.zeros((N,N2)).astype(bool)
    # ...reduction of risk and increase of the prevalence of bottleneck tasks
    F_safer = np.zeros((N,N2)).astype(bool)
    # ...that destination is not of high risk
    F_not_high_risk = np.zeros((N,N2)).astype(bool)
    # ...that the transition is not to self
    F_not_self = np.zeros((N,N2)).astype(bool)

    print('Creating filtering matrices...', end=' ')
    t_now = time()
    # Brute force approach (for each transition...)
    for i in range(N):
        row_i = data.occ.iloc[origin_ids[i]]

        for j in range(N2):
            row_j = data.occ.iloc[destination_ids[j]]

            is_jobzone_ok = np.abs(row_i.job_zone - row_j.job_zone) <= MAX_JOB_ZONE_DIF
            is_earnings_ok = (row_j.annual_earnings / row_i.annual_earnings) > MIN_EARNINGS_RATIO
            is_safer = (row_i.risk > row_j.risk) & (row_i.prevalence < row_j.prevalence)
            is_not_high_risk = (row_j.risk_category != 'High risk')

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

    # Matrix indicating desirable transitions
    F_desirable = F_viable & F_earnings

    # Matrix indicating safe transitions
    F_strictly_safe = F_safer & F_not_high_risk

    # Matrices indicating safe and desirable transitions
    F_safe_desirable = F_desirable & F_not_high_risk # 1st definition
    F_strictly_safe_desirable = F_desirable & F_strictly_safe # 2nd (stricter) definition

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
        'F_strictly_safe': F_strictly_safe,
        'F_not_self': F_not_self,
        'F_safe_desirable': F_safe_desirable,
        'F_strictly_safe_desirable': F_strictly_safe_desirable,
    }

    # Remove transitions to self
    for key in list(filter_matrices.keys()):
        filter_matrices[key] = filter_matrices[key] & F_not_self

    filter_matrices['origin_ids'] = origin_ids
    filter_matrices['destination_ids'] = destination_ids

    # Export filtering matrices
    if export_path is not None:
        if os.path.exists(export_path) == False:
            pickle.dump(filter_matrices, open(export_path, 'wb'))
            print(f'Filtering matrices saved at {export_path}')
        else:
            print('File already exists! (not saved)')

    return filter_matrices

def show_skills_overlap(
    job_i,
    job_j,
    data=data, sim=sim,
    embeddings=embeddings,
    skills_match = 'optional',  # either 'optional' or 'essential'
    matching_method='one_to_one',
    verbose=True,
    rounding=True):
    """
    NLP-adjusted overlap of skill sets between occupations job_i and job_j
    """

    job_i = data.occ_title_to_id(job_i)
    job_j = data.occ_title_to_id(job_j)

    if verbose: print(f"from {data.occ.loc[job_i].preferred_label} (id {job_i}) to {data.occ.loc[job_j].preferred_label} (id {job_j})")

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
        symmetric=False,
        rounding=rounding)

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
        self.esco_features_1 = data.concepts[data.concepts.code.isin(self.esco_features_1)][['code','title']].sort_values('code').copy()

        self.esco_vectors_2 = np.load(data_folder + 'interim/work_activity_features/esco_hierarchy_vectors_level_2.npy')
        self.esco_features_2 = pickle.load(open(data_folder + 'interim/work_activity_features/esco_hierarchy_codes_level_2.pickle', 'rb'))
        self.esco_features_2 = data.concepts[data.concepts.code.isin(self.esco_features_2)][['code','title']].sort_values('code').copy()

        self.esco_vectors_3 = np.load(data_folder + 'interim/work_activity_features/esco_hierarchy_vectors_level_3.npy')
        self.esco_features_3 = pickle.load(open(data_folder + 'interim/work_activity_features/esco_hierarchy_codes_level_3.pickle', 'rb'))
        self.esco_features_3 = data.concepts[data.concepts.code.isin(self.esco_features_3)][['code','title']].sort_values('code').copy()

    def select_esco_level(self, level=2):
        # Select the level of ESCO hierarchy
        if level==1:
            self.vectors = self.esco_vectors_1
            self.features = self.esco_features_1
        elif level==2:
            self.vectors = self.esco_vectors_2
            self.features = self.esco_features_2
        elif level==3:
            self.vectors = self.esco_vectors_3
            self.features = self.esco_features_3
        elif level is None:
            self.vectors = self.work_context_vectors
            self.features = self.work_context_features

    def get_feature_deltas(self, job_i, job_j, esco_level=2):
        """ Useful for checking what are the biggest differences between the two occupations """
        self.select_esco_level(esco_level)
        # Calculate vector deltas and add category labels
        delta_vector = self.vectors[job_j] - self.vectors[job_i]
        df = self.features.copy()
        df['job_i'] = self.vectors[job_i]
        df['job_j'] = self.vectors[job_j]
        df['deltas'] = delta_vector
        df['deltas_abs'] = np.abs(delta_vector)
        return df.sort_values('deltas_abs', ascending=False)

    def most_impactful_features(self, job_i, job_j, esco_level=2):
        """
        Useful for checking what makes both occupations similar; calculates 'impact'
        which relates to how much an element contributes to similarity

        Parameters
        ----------
        esco_level (int or boolean):
            ESCO hierarchy level (normally use level 2); if esco_level is None, uses work context vectors
        """
        self.select_esco_level(esco_level)
        original_destination_vector = self.vectors[job_j,:]
        origin_vector = normalize(self.vectors[job_i,:].reshape(1,-1))
        original_sim = cosine(normalize(original_destination_vector.reshape(1,-1)), origin_vector)
        impacts = []
        for j in range(len(original_destination_vector)):
            new_vector = original_destination_vector.copy()
            new_vector[j] = 0
            new_vector = normalize(new_vector.reshape(1,-1))
            impact = original_sim - cosine(new_vector, origin_vector)
            impacts.append(-impact)
        df = self.features.copy()
        df['impact'] = impacts
        return df.sort_values('impact', ascending=False)


class SkillsGaps():
    """
    Class for characterising prevalent skills gaps for a collection of transitions
    """

    def __init__(self, trans_to_analyse, verbose=True):
        """
        trans_to_analyse (pandas.DataFrame):
            Table with transitions, with columns 'origin_id' and 'destination_id'
            indicating the occupations involved in the transition.
        """
        self.trans_to_analyse = trans_to_analyse
        self.get_skills_scores(verbose=verbose)
        self.skill_similarities_all = None
        self._skills_gaps = None
        self.cluster_gaps = None

    @property
    def skills_gaps(self):
        if self._skills_gaps is None:
            self._skills_gaps = self.get_skills_gaps()
        return self._skills_gaps

    def get_skills_scores(self, verbose=True):
        """
        Compare skillsets and get matching scores for each comparison
        """

        ## List of lists (a list for each transition)
        # Skills IDs for all transitions
        self.destination_skills_id_ALL = []
        self.origin_skills_id_ALL = []
        # All matching scores
        self.destination_skills_id_score_ALL = []
        self.origin_skills_id_score_ALL = []
        # All semantic similarity values (not used in the final analysis)
        self.destination_skills_id_sim_ALL = []
        self.origin_skills_id_sim_ALL = []

        t = time()
        for j, row in self.trans_to_analyse.iterrows():

            # Get job IDs
            job_i = row.origin_id
            job_j = row.destination_id

            # Create the input dataframe in the required format
            df = show_skills_overlap(job_i, job_j, verbose=False)

            ###### DESTINATION SKILLS ######
            # Save the skill IDs and similarity values
            self.destination_skills_id_ALL.append(df.destination_skill_id.to_list())
            self.destination_skills_id_score_ALL.append(df.score.to_list())
            self.destination_skills_id_sim_ALL.append(df.similarity.to_list())

            ###### ORIGIN SKILLS ######
            # Exclude unmatched destination skill rows
            origin_skills = df[df.origin_skill_id.apply(lambda x: type(x)!=str)]

            # Extract the oriign skill IDs, matching scores and similarity values
            self.origin_skills_id_ALL.append(origin_skills.origin_skill_id.to_list())
            self.origin_skills_id_score_ALL.append(origin_skills.score.to_list())
            self.origin_skills_id_sim_ALL.append(origin_skills.similarity.to_list())

        t_elapsed = time() - t
        if verbose: print(f'Time elapsed: {t_elapsed :.2f} sec ({t_elapsed/len(self.trans_to_analyse): .3f} per transition)')


    def setup(self, transition_indices=None, skills_type='destination', skill_items=None):
        """
        Parameters:
        ----------
        transition_indices (list of int)
            Transitions that we wish to analyse, specified by the row indices of 'trans_to_analyse'
        skills_type (str):
            Sets up which skills type are we checking ('destination' vs 'origin')
        """

        # Store the analysis parameters
        if type(transition_indices)==type(None):
            self.transition_indices = range(0, len(self.trans_to_analyse))
        else:
            self.transition_indices = transition_indices

        self.skills_type = skills_type

        # Number of transitions we have
        self.n_trans = len(self.transition_indices)
        # Get all skills occurrences and matching scores
        self.skill_similarities_all = self.merge_lists()

        # Select only specific skill items (either 'K' for knowledge, 'S' for skills or 'A' for attitude)
        if skill_items is None:
            pass
        else:
            df = self.skill_similarities_all.merge(data.skills[['id','skill_category']], left_on='skills_id', right_on='id', how='left')
            self.skill_similarities_all = self.skill_similarities_all[df.skill_category.isin(skill_items)]

        self._skills_gaps = self.get_skills_gaps()

    def prevalent_skills_gaps(self, top_x=10, percentile=False):
        """
        Show most prevalent skills gaps
        """
        # Return the top most prevalent skills
        return self.get_most_prevalent_gaps(self.skills_gaps, top_x=top_x, percentile=percentile)

    def prevalent_cluster_gaps(self, level='level_3', top_x=10, percentile=False):

        self.cluster_gaps = self.get_cluster_gaps(level)
        prevalent_clusters = self.get_most_prevalent_gaps(self.cluster_gaps, top_x=top_x, percentile=percentile)
        return self.most_prevalent_cluster_skills(prevalent_clusters)

    def merge_lists(self):

        """
        Creates dataframe with all skills occurrences, their matched similarities and scores.
        It is possible to analyse a subset of all supplied transitions, by specifying
        the row indices of 'trans_to_analyse' table using 'transition_indices'
        """

        # Merge lists
        list_skills = []
        list_score = []
        list_similarity = []

        for i in self.transition_indices:
            if self.skills_type=='destination':
                list_skills += self.destination_skills_id_ALL[i]
                list_score += self.destination_skills_id_score_ALL[i]
                list_similarity += self.destination_skills_id_sim_ALL[i]
            elif self.skills_type=='origin':
                list_skills +=  self.origin_skills_id_ALL[i]
                list_score += self.origin_skills_id_score_ALL[i]
                list_similarity += self.origin_skills_id_sim_ALL[i]

        skill_similarities_all = pd.DataFrame(data={
            'skills_id': list_skills,
            'score': list_score,
            'similarity': list_similarity})

        # If a skill was not matched, then set it to 0
        skill_similarities_all.loc[skill_similarities_all.score.isnull(), 'score'] = 0

        return skill_similarities_all

    def count_and_agg_scores(self, skill_similarities_all, groupby_column):

        """ Aggregates scores for each skill or cluster (depending on groupby_column) """

        # Counts
        skill_counts = skill_similarities_all.groupby(groupby_column).count()
        # Mean similarity
        skill_similarities = skill_similarities_all.groupby(groupby_column).mean()
        # Create the dataframe
        skill_similarities['counts'] = skill_counts['score']
        skill_similarities['stdev'] = skill_similarities_all.groupby(groupby_column).std()['score']
        skill_similarities.reset_index(inplace=True)

        return skill_similarities

    def get_skills_gaps(self):

        """ Agregates scores for skills """

        # Aggregate scores
        skill_similarities = self.count_and_agg_scores(self.skill_similarities_all, 'skills_id')
        skill_similarities['prevalence'] = skill_similarities['counts'] / self.n_trans
        # Add information about skills
        skill_similarities = skill_similarities.merge(
            data.skills[['id', 'preferred_label', 'level_1', 'level_2', 'level_3']],
            left_on='skills_id', right_on='id', how='left')
        # Clean up the dataframe
        skill_similarities = self.clean_up_df(skill_similarities)
        skill_similarities = skill_similarities[['id', 'preferred_label', 'level_1', 'level_2', 'level_3', 'counts', 'prevalence', 'score' , 'stdev']]

        return skill_similarities

    def get_cluster_gaps(self, level='level_1'):

        """ Agregates scores for ESCO skills clusters """

        # Save the level of analysis
        self.level = level

        # Add skills cluster information
        skill_similarities_all_clust = self.skill_similarities_all.merge(data.skills[[
            'id', 'preferred_label', 'level_1', 'level_2', 'level_3', 'code']], left_on='skills_id', right_on='id')
        # Aggregate scores
        skill_similarities = self.count_and_agg_scores(skill_similarities_all_clust, level)
        skill_similarities['prevalence'] = skill_similarities['counts'] / self.n_trans
        # Add skills cluster title
        skill_similarities = skill_similarities.merge(data.concepts[['code','title']], left_on=level, right_on='code')
        # Clean up the dataframe
        skill_similarities = self.clean_up_df(skill_similarities)
        skill_similarities = skill_similarities[['code', 'title', 'counts', 'prevalence', 'score', 'stdev']]

        return skill_similarities

    def clean_up_df(self, df):
        """ Clean up the dataframe for presentation """
        df.prevalence = df.prevalence.round(3)
        df.similarity = df.similarity.round(3)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_most_prevalent_gaps(self, skills_gaps, top_x=10, percentile=False):
        """ Select only the most prevalent skills """
        if percentile:
            df = skills_gaps[skills_gaps.prevalence > np.percentile(skills_gaps.prevalence, top_x)]
            df = df.sort_values('score', ascending=False)
            return df
        else:
            return skills_gaps.sort_values('prevalence', ascending=False).head(top_x).sort_values('score', ascending=False)

    def most_prevalent_cluster_skills(self, prevalent_clusters, top_n=3):
        """ For each cluster, find top_n most prevalent skills and add to the dataframe """
        x = []
        for j, row in prevalent_clusters.iterrows():
            dff = self.skills_gaps[self.skills_gaps[self.level]==row.code]
            dff = dff.sort_values('prevalence', ascending=False).iloc[0:top_n]
            xx = []
            # Add matching scores
            for jj, rrow in dff.iterrows():
                xx.append(f'{rrow.preferred_label} ({np.round(rrow.score,2)})')
            x.append(', '.join(xx))

        prevalent_clusters_ = prevalent_clusters.copy()
        prevalent_clusters_['skills'] = x
        return prevalent_clusters_

class Upskilling():
    """
    Tests upskilling by adding new ESCO skills to occupations' skillsets and
    re-evaluating viable transitions
    """

    def __init__(self,
                 origin_ids='report',
                 new_skillsets=[None],
                 destination_ids='report',
                 verbose=False,
                 load_data_path=False,
                 ):
        """
        Parameters
        ----------
        origin_ids (list of int, or str):
            Origin occupation integer identifiers
        new_skillsets (list of int):
            List of the new skills (or combinations of skills) to be tested
        destination_ids (list of int, or str):
            Destination occupation integer identifiers
        """
        self.verbose = verbose

        # List of perturbed matrices
        self.new_W_combined = None
        # Upskilling analysis results
        self.upskilling_effects = None

        if load_data_path:
            self.load_data_path = load_data_path
            result_dict = self.load_results()
            self.new_W_combined = result_dict['new_W_combined']
            origin_ids = result_dict['origin_ids']
            destination_ids = result_dict['destination_ids']
            new_skillsets = result_dict['new_skillsets']
            if 'upskilling_effects' in list(result_dict.keys()):
                self.upskilling_effects = result_dict['upskilling_effects']

        # Origin and destination occupations
        self.origin_ids = occupations_to_check(origin_ids)
        self.destination_ids = occupations_to_check(destination_ids)
        # Prep a list of lists of skills (allowing us to add multiple skill combinations)
        self.list_of_new_skills = [skill if type(skill)==list else [skill] for skill in new_skillsets]

        self.n_origin_occupations = len(self.origin_ids)
        self.n_destination_occupations = len(self.destination_ids)
        self.n_new_skills = len(self.list_of_new_skills)

        # Dictionaries mapping matrix element indices to the original occupation IDs
        self.origin_ids_to_row_indices = dict(zip(self.origin_ids, list(range(len(self.origin_ids)))))
        self.destination_ids_to_col_indices = dict(zip(self.destination_ids, list(range(len(self.destination_ids)))))
        self.row_indices_to_origin_ids = dict(zip(list(range(len(self.origin_ids))),self.origin_ids))
        self.col_indices_to_destination_ids = dict(zip(list(range(len(self.destination_ids))),self.destination_ids))

        ## Required variables for re-calculating similarities (Note: should eventually do further refactoring) ##
        # Variables for recalculating work activity feature vector similarity
        activity_vector_dir = f'{useful_paths.data_dir}interim/work_activity_features/'
        self.element_codes_2 = np.array(pickle.load(open(f'{activity_vector_dir}esco_hierarchy_codes_level_2.pickle', 'rb')))
        self.normalisation_params = pickle.load(open(f'{activity_vector_dir}esco_hierarchy_norm_params.pickle', 'rb'))
        self.occupation_vectors_level_2_abs = np.load(f'{activity_vector_dir}esco_hierarchy_vectors_level_2_abs.npy')
        self.occupation_vectors_level_2 = np.load(f'{activity_vector_dir}esco_hierarchy_vectors_level_2.npy')
        # Variables including work context similarities into the combined measure
        esco_to_work_context_vector = pd.read_csv(useful_paths.data_dir + 'interim/work_context_features/occupations_work_context_vector.csv')
        esco_with_work_context = esco_to_work_context_vector[esco_to_work_context_vector.has_vector==True].id.to_list()
        occ_no_work_context = set(data.occupations.id.to_list()).difference(set(esco_with_work_context))
        self.origin_indices_no_work_context = self.indices_of_specified_elements(self.origin_ids, occ_no_work_context)
        self.destination_indices_no_work_context = self.indices_of_specified_elements(self.destination_ids, occ_no_work_context)
        # Parameters for combining the different similarity measures
        with open(f'{useful_paths.codebase_dir}configs/default_combined_similarity_params.yaml', 'r') as f:
            self.combining_params = yaml.load(f, Loader=yaml.FullLoader)

    @staticmethod
    def indices_of_specified_elements(list_of_ids, list_of_specified_ids):
        """ Outputs indices of elements in list_of_ids which are also in the list_of_specified_ids """
        indices = []
        for j, element_j in enumerate(list_of_ids):
            if element_j in list_of_specified_ids:
                indices.append(j)
        return indices

    def effectiveness(self,
                      safe_definition='default',
                      significance_test_tolerance=False,
                      select_origin_ids=None,
                      select_destination_ids=None):
        """
        Summarise the effectiveness of the tested skills across the specified transitions
        (by default, characterise across all transitions)
        """
        if self.upskilling_effects is None:
            self.new_transitions()

        # Compile a table with summary stats for each skill
        skills_analysis_results = []
        for n, new_skill in enumerate(self.list_of_new_skills):
            upskilling_dict = self.upskilling_effects[n]
            analysis_dict = {}
            analysis_dict['new_skill'] = upskilling_dict['new_skill']
            analysis_dict['new_skill_label'] = upskilling_dict['new_skill_label']

            # Analyse novel transitions
            transition_df = upskilling_dict['transition_table']
            transition_df = transition_df[transition_df.is_new]
            # Select only the transition destinations of interest
            if select_destination_ids is not None:
                selected_transition_df = transition_df[transition_df.destination_id.isin(select_destination_ids)]
            else:
                selected_transition_df = transition_df
            # Select safe and desirable
            if safe_definition=='default':
                selected_transition_df = selected_transition_df[selected_transition_df.is_safe_desirable]
            elif safe_definition=='strict':
                selected_transition_df = selected_transition_df[selected_transition_df.is_strictly_safe_desirable]
            elif safe_definition==None:
                selected_transition_df = selected_transition_df[selected_transition_df.is_desirable]
            df = self.count_transitions(selected_transition_df)

            if select_origin_ids is not None:
                df = df[df.origin_id.isin(select_origin_ids)]
            analysis_dict['n_mean'] = df.counts.mean()
            analysis_dict['n_median'] = df.counts.median()

            if significance_test_tolerance is not False:
                analysis_dict['p_value'] = wilcoxon(df.counts.to_list(), correction=True).pvalue
                analysis_dict['is_significant'] = analysis_dict['p_value'] < significance_test_tolerance
            skills_analysis_results.append(analysis_dict)

        skills_analysis_df = pd.DataFrame(data=skills_analysis_results)
        skills_analysis_df = self.clean_up_df(skills_analysis_df)
        return skills_analysis_df.sort_values('n_mean', ascending=False)

    @staticmethod
    def clean_up_list(old_list):
        new_list = []
        contains_combinations = False
        for x in old_list:
            if len(x) == 1:
                new_list.append(x[0])
            else:
                new_list.append(x)
                contains_combinations = True
        return new_list, contains_combinations

    @staticmethod
    def add_skills_categories(df):
        df = data.add_field_to_skill(df, 'new_skill', 'level_1')
        df = df.merge(data.concepts[['code', 'title']], left_on='level_1', right_on='code', how='left').drop('code', axis=1).rename(columns={'title': 'ESCO skill category'})
        df = data.add_field_to_skill(df, 'new_skill', 'level_2')
        df = df.merge(data.concepts[['code', 'title']], left_on='level_2', right_on='code', how='left').drop('code', axis=1).rename(columns={'title': 'ESCO skill subcategory'})
        return df

    def clean_up_df(self, df):
        df.new_skill, contains_combinations = self.clean_up_list(df.new_skill.to_list())
        df.new_skill_label, _ = self.clean_up_list(df.new_skill_label.to_list())
        if not contains_combinations: df = self.add_skills_categories(df)
        return df

    def new_transitions(self,
        MIN_VIABLE = MIN_VIABLE_DEF,
        HIGHLY_VIABLE = HIGHLY_VIABLE_DEF,
        MAX_JOB_ZONE_DIF = MAX_JOB_ZONE_DIF_DEF,
        MIN_EARNINGS_RATIO = MIN_EARNINGS_RATIO_DEF):
        """
        Evaluates the new transitions after upskilling
        """

        if self.new_W_combined is None:
            self.recalculate_similarities()

        W_combined_baseline = sim.W_combined[self.origin_ids,:].copy()
        W_combined_baseline = W_combined_baseline[:, self.destination_ids]

        self.upskilling_effects = []
        for n, new_skill in enumerate(self.list_of_new_skills):
            W_new_combined = self.new_W_combined[n]
            # Get new transitions above similarity threshold
            viable_transitions = np.where((W_new_combined > MIN_VIABLE) & (W_combined_baseline <= MIN_VIABLE))
            # Get new transition similarities
            new_similarities = W_new_combined[viable_transitions]
            # Fetch other data about the transition
            transition_pairs_indices = [(viable_transitions[0][x], viable_transitions[1][x]) for x in range(len(viable_transitions[0]))]
            transition_pairs_ids = [(self.row_indices_to_origin_ids[i], self.col_indices_to_destination_ids[j]) for i, j in transition_pairs_indices]
            transition_df = get_transition_data(transition_pairs_ids, verbose=self.verbose)
            # Organise the dataframe
            transition_df = transition_df.drop(['W_skills', 'W_work', 'W_essential_skills', 'W_optional_skills', 'W_activities', 'W_work_context'], axis=1)

            transition_df['baseline_viable'] = transition_df['is_viable'].copy()
            # Find the novel transitions
            transition_df['new_similarity'] = new_similarities
            transition_df['is_viable'] = (transition_df['new_similarity']>MIN_VIABLE) & transition_df['is_jobzone_ok']
            transition_df['is_desirable'] = transition_df['is_viable'] & transition_df['is_earnings_ok']
            transition_df['is_safe_desirable'] = transition_df['is_desirable'] & transition_df['is_not_high_risk']
            transition_df['is_strictly_safe_desirable'] = transition_df['is_desirable'] & transition_df['is_strictly_safe']
            # Flag for brand new viable transitions
            transition_df['is_new'] = transition_df['is_viable'] & (transition_df['baseline_viable'] == False)

            # Count new safe and desirable transitions for each occupation
            counts_safe_desirable = self.count_transitions(transition_df[transition_df.is_new & transition_df.is_safe_desirable])
            counts_strictly_safe_desirable = self.count_transitions(transition_df[transition_df.is_new & transition_df.is_strictly_safe_desirable])

            # List of new transition destinations for each occupation
            new_transitions = []
            for job_i in self.origin_ids:
                df = transition_df[transition_df.origin_id==job_i]
                job_i_trans = {'origin_id': job_i,
                               'origin_label': job_i,
                               'destination_id': [],
                               'destination_label': []}
                if len(df) != 0:
                    for j, row in df.iterrows():
                        job_i_trans['destination_label'].append(row.destination_label)
                        job_i_trans['destination_id'].append(row.destination_id)
                new_transitions.append(job_i_trans)

            # Store all the information about effects of adding the skills
            self.upskilling_effects.append(
                {
                    'new_skill': new_skill,
                    'new_skill_label': [data.skills.loc[s].preferred_label for s in new_skill],
                    'new_transitions': new_transitions,
                    'counts_new_safe_desirable': counts_safe_desirable,
                    'counts_new_strictly_safe_desirable': counts_strictly_safe_desirable,
                    'transition_table': transition_df}
            )

    def recalculate_similarities(self, load_data=False):
        """ Recalculates all similarities and combines them """
        # Recalculate all skills and work activity similarities with the new sets of skills
        self.new_W_essential_skills = self.recalculate_skills_similarities(skills_match = 'essential')
        self.new_W_optional_skills = self.recalculate_skills_similarities(skills_match = 'optional')
        self.new_W_activities = self.recalculate_work_activity_similarities()
        # Get work context similarities (don't need to be recalculated)
        self.W_work_context = self.fetch_work_context_similarities()
        # For each set of skills, combine the new similarity matrices
        self.new_W_combined = []
        for n, new_skills in enumerate(self.list_of_new_skills):
            # Calculate the new combined, perturbed similarity matrix
            W_combined = self.combine_similarity_measures(
                self.new_W_essential_skills[n],
                self.new_W_optional_skills[n],
                self.new_W_activities[n],
                self.W_work_context,
                self.combining_params
            )
            self.new_W_combined.append(W_combined)

    def combine_similarity_measures(self, W_essential, W_optional, W_activities, W_context, params):
        """ Calculates the combined similarity measure, according to parameters in params """
        # Combined similarity matrix
        W_combined = (params['p_essential_skills'] * W_essential) + (params['p_optional_skills'] * W_optional) + (params['p_work_activities'] * W_activities) + (params['p_work_context'] * W_context)
        # Adjust for cases where work context doesn't exist for either origin or destination occupation
        p_essential_skills_x = params['p_essential_skills']/(1-params['p_work_context'])
        p_optional_skills_x = params['p_optional_skills']/(1-params['p_work_context'])
        p_work_activities_x = params['p_work_activities']/(1-params['p_work_context'])
        for i in self.origin_indices_no_work_context:
            for j in range(len(W_combined)):
                W_combined[i][j] = (p_essential_skills_x * W_essential[i][j]) + (p_optional_skills_x * W_optional[i][j]) + (p_work_activities_x * W_activities[i][j])
        for i in range(len(W_combined)):
            for j in self.destination_indices_no_work_context:
                W_combined[i][j] = (p_essential_skills_x * W_essential[i][j]) + (p_optional_skills_x * W_optional[i][j]) + (p_work_activities_x * W_activities[i][j])
        return W_combined

    def recalculate_skills_similarities(self, skills_match = 'optional'):
        """
        Add skills to occupations' skillsets and recalculate NLP-adjusted overlaps
        """
        if self.verbose: print(f'Recalculating {skills_match} skills similarities...')
        # Origin occupations' skills lists
        if skills_match == 'optional':
            origin_node_to_items = data.node_to_all_items.loc[self.origin_ids].copy()
        elif skills_match == 'essential':
            origin_node_to_items = data.node_to_essential_items.loc[self.origin_ids].copy()
        origin_node_to_items.sector = 'origin'
        # Adjust IDs of the origin items
        origin_node_to_items = self.adjust_node_ids(origin_node_to_items)

        # Destination occupations' skills lists (always the 'essential' skills only)
        destination_node_to_items =  data.node_to_essential_items.loc[self.destination_ids].copy()
        destination_node_to_items.sector = 'destination'
        # Adjust IDs of the destination items
        destination_node_to_items = self.adjust_node_ids(destination_node_to_items, id_offset = self.n_origin_occupations)

        # List with all perturbed similarity matrices
        list_of_new_W = []
        # Go through each new skill in question and test them out!
        for new_skills in self.list_of_new_skills:
            if self.verbose: print(f'Adding skill(s) {new_skills} to origin occupations.')
            # Add skills items to each origin occupation's skills list
            perturbed_origin_node_to_items = origin_node_to_items.copy()
            new_items_list = [] # New skills lists
            for job_i, row in perturbed_origin_node_to_items.iterrows():
                # Original skillset of the origin occupation
                original_skillset = set(row.items_list)
                # Add the set of new skills
                new_skillset = original_skillset.union(set(new_skills))
                new_items_list.append(str(sorted(list(new_skillset))))
            # Update the origin skills lists
            perturbed_origin_node_to_items.items_list = new_items_list
            # Re-evaluate all items lists so that they are treated as lists
            perturbed_origin_node_to_items.items_list = perturbed_origin_node_to_items.items_list.apply(lambda x: literal_eval(x))
            # Combine both origin and destination lists of skills
            node_to_items = pd.concat([perturbed_origin_node_to_items, destination_node_to_items]).reset_index(drop=True)

            # Perform the comparison!
            Comp = compare_nodes_utils.CompareSectors(
                node_to_items,
                embeddings,
                combos=[('origin','destination')],
                metric='cosine',
                symmetric=False,
                verbose=False)
            t = time()
            if self.verbose: print('Running comparisons...', end=' ')
            Comp.run_comparisons(dump=False)
            Comp.collect_comparisons()
            t_elapsed = time()-t
            if self.verbose: print(f'Done in {t_elapsed:.0f} seconds!')

            # Processing the outputs (select only the relevant edges, starting from origin occupations)
            W = Comp.D
            i_edges = [edge[0] for edge in Comp.real_edge_list]
            origin_edges = np.array(Comp.real_edge_list)[np.where(np.array(i_edges)<self.n_origin_occupations)[0]]

            W_perturbed = np.zeros((self.n_origin_occupations,self.n_destination_occupations))
            for edge in origin_edges:
                W_perturbed[edge[0], edge[1]-self.n_origin_occupations] = W[edge[0],edge[1]]
            # Take care of nulls (might appear if destination occupation had no essential skills)
            W_perturbed[np.isinf(W_perturbed)] = 0


            # Store the new, perturbed similarity matrix
            list_of_new_W.append(W_perturbed)
        return list_of_new_W

    @staticmethod
    def adjust_node_ids(node_to_items, id_offset=0):
        """ Helper function for self.recalculate_skills_similarities() """
        node_to_items['original_id'] = node_to_items.id.copy()
        node_to_items['id'] = np.array(list(range(0, len(node_to_items)))) + id_offset
        node_to_items.reset_index(drop=True)
        return node_to_items

    def recalculate_work_activity_similarities(self):
        """
        Recalculates similarity between work activity vectors
        """
        t = time()
        if self.verbose: print('Recalculating work activity feature vector alignments...', end=' ')

        # List with all perturbed similarity matrices
        list_of_new_W = []
        # Go through each new set of skills in question and test them out!
        for new_skills in self.list_of_new_skills:
            # Re-calculated similarities
            W_perturbed = np.zeros((self.n_origin_occupations,self.n_destination_occupations))

            # For each origin occupation
            for i, job_i in enumerate(self.origin_ids):

                # Existing work activity feature vector
                new_feature_vector = self.occupation_vectors_level_2_abs[job_i].copy()

                origin_skillset = data.node_to_essential_items.loc[[job_i]].items_list.values[0]
                # For each single skill in the new set of skills
                for new_skill_id in new_skills:
                    # Find the skill's hierarchy code
                    skill_code = data.skills.loc[new_skill_id].level_2
                    # Check if the skill is already in the skill set
                    if new_skill_id in origin_skillset:
                        pass
                    # Check if the skill is a knowledge or attitude item (these are not included in the measure)
                    elif skill_code[0] in ['K', 'A']:
                        pass
                    # Add the skill to the job_i feature vector
                    else:
                        # Find the element number for the skill
                        element_id = np.where(self.element_codes_2==skill_code)[0][0]
                        # Increment the element by one
                        new_feature_vector[element_id] += 1

                # Create a new normalised feature vector
                new_feature_vector = new_feature_vector.reshape(1,-1)
                new_feature_vector_norm = normalize(new_feature_vector)

                # Re-calculate the similarity
                new_d = cdist(new_feature_vector_norm, self.occupation_vectors_level_2[self.destination_ids,:], 'euclidean')
                new_d = (new_d - self.normalisation_params['d_MIN_LEVEL2'])/(self.normalisation_params['d_MAX_LEVEL2']-self.normalisation_params['d_MIN_LEVEL2'])
                new_similarities = 1-new_d # Vector of the new similarities

                # Store the similarities in the perturbed similarity matrix
                for j, new_sim in enumerate(new_similarities[0,:]):
                    W_perturbed[i, j] = new_sim

            W_perturbed[np.isinf(W_perturbed)] = 0 # just in case
            # Store the new, perturbed similarity matrix
            list_of_new_W.append(W_perturbed)

        t_elapsed = time()-t
        if self.verbose: print(f'Done in {t_elapsed:.0f} seconds!')
        return list_of_new_W

    def fetch_work_context_similarities(self):
        W_work_context = sim.W_work_context[self.origin_ids, :].copy()
        W_work_context = W_work_context[:, self.destination_ids]
        return W_work_context

    def count_transitions(self, transition_df):
        # Numbers for each occupation
        df = transition_df.groupby('origin_id').agg({'destination_id': 'count'}).reset_index().rename(columns={'destination_id': 'counts'})
        # Add occupations without any new transitions
        df_ids = pd.DataFrame(data={'origin_id': self.origin_ids})
        df_ids = df_ids.merge(df, how='left')
        df_ids.loc[df_ids.counts.isnull(), 'counts'] = 0
        return df_ids

    def dump_results(self, filename='upskilling_results.pickle', dir=f'{useful_paths.data_dir}interim/upskilling_analysis/'):
        """
        Dumps the recalculated, perturbed skills matrices for later reuse
        """
        if self.verbose: print(f'Dumping in {dir+filename}')
        result_dict = {
            'origin_ids': self.origin_ids,
            'destination_ids': self.destination_ids,
            'new_skillsets': self.list_of_new_skills,
            'new_W_combined': self.new_W_combined,
            'upskilling_effects': self.upskilling_effects
        }
        pickle_large_files.pickle_dump(result_dict, dir+filename)

    def load_results(self):
        """
        Loads pre-computed perturbed skills matrices
        """
        if self.verbose: print(f'Loading data form {self.load_data_path}')
        return pickle_large_files.pickle_load(self.load_data_path)


def get_flow_matrix(trans_clust, level):
    """ Number of transitions between clusters (e.g. sectors and sub-sectors) """
    n_clust = len(np.unique(data.occ[level]))
    flow_matrix = np.zeros((n_clust, n_clust))
    for j, row in trans_clust.iterrows():
        clust_origin = row['origin_' + level]
        clust_destination = row['destination_' + level]
        flow_matrix[clust_origin, clust_destination] += 1
    return flow_matrix

def normalise_rows(A):
    A = A.copy()
    for j in range(len(A)):
        A[j,:] = A[j,:] / np.sum(A[j,:])
    return A


def assess_transition_options(
    filter_matrices_path=f'{useful_paths.data_dir}interim/transitions/filter_matrices_Report_occupations.pickle',
    filter_crowd_feasibility=False):
    """ Update for more flexibility """

    # Import or create a set of filtering matrices
    export_path = filter_matrices_path
    if os.path.exists(export_path):
        filter_matrices = pickle.load(open(export_path,'rb'))
        print(f'Imported filtering matrices from {export_path}')
    else:
        # May take about 30 mins
        filter_matrices = create_filtering_matrices(
            origin_ids='report',
            destination_ids= 'report',
            export_path = export_path)
    
    # Desirable transitions
    n_desirable = np.sum(filter_matrices['F_desirable'], axis=1)
    # Highly viable, desirable transitions
    n_desirable_and_highly_viable = np.sum(
        (filter_matrices['F_desirable'] & filter_matrices['F_highly_viable']), axis=1)
    # Safe and desirable transitions
    n_safe_desirable = np.sum(filter_matrices['F_safe_desirable'], axis=1)
    # Highly viable, safe and desirable transitions
    n_safe_desirable_and_highly_viable = np.sum(
        (filter_matrices['F_safe_desirable'] & filter_matrices['F_highly_viable']), axis=1)
    # Strictly safer and desirable transitions
    n_safe_desirable_strict = np.sum(filter_matrices['F_strictly_safe_desirable'], axis=1)
    # Highly viable, strictly safe
    n_safe_desirable_strict_and_highly_viable = np.sum(
        (filter_matrices['F_strictly_safe_desirable'] & filter_matrices['F_highly_viable']), axis=1)

    occ_transitions = data.occ_report[['id', 'concept_uri', 'preferred_label', 'isco_level_4', 'risk_category']].copy()
    occ_transitions['n_desirable'] = n_desirable
    occ_transitions['n_desirable_and_highly_viable'] = n_desirable_and_highly_viable
    occ_transitions['n_safe_desirable'] = n_safe_desirable
    occ_transitions['n_safe_desirable_and_highly_viable'] = n_safe_desirable_and_highly_viable
    occ_transitions['n_safe_desirable_strict'] = n_safe_desirable_strict
    occ_transitions['n_safe_desirable_strict_and_highly_viable'] = n_safe_desirable_strict_and_highly_viable

    # Rename the risk categories, as used in the report
    occ_transitions.loc[occ_transitions.risk_category.isin(['Low risk', 'Other']), 'risk_category'] = 'Lower risk'

    return occ_transitions
