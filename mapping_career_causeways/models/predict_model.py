import os
import joblib
import numpy as np
import pandas as pd
import yaml

from scipy.spatial.distance import cosine
from pathlib import Path

import mapping_career_causeways
import mapping_career_causeways.transitions_utils as trans_utils


useful_paths = mapping_career_causeways.Paths()

occupation_embeddings = np.load(os.path.join(
    useful_paths.data_dir, 
    'interim/embeddings/embeddings_occupation_description_SBERT_bert-base-nli-mean-tokens.npy')
    )

with open(os.path.join(useful_paths.config_dir, 'model_config.yaml'), 'rt') as f:
    config = yaml.safe_load(f.read())


def aggregate_mean_crowd_feasibility(data):
    """aggregate_crowd_ratings
    Aggregate crowd feasibility ratings to get the mean rating
    for each transition.
    
    Args:
    
    Returns:
    """
    data = (data
            .groupby(['origin_id', 'destination_id'])
            .agg({'feasibility_1-5': 'mean'})
            .reset_index()
            .rename(columns={'feasibility_1-5': f'feasibility_mean'}))

    data = data.set_index(['origin_id', 'destination_id'])
    return data


def create_transition_features(origin_ids, destination_ids):
    """create_transition_features
    Obtain model features from the transitions data.
    
    Args:
    
    Returns:
    
    """
    transition_pairs = zip(origin_ids, destination_ids)
    transitions_df = trans_utils.get_transition_data(transition_pairs)
    cols = ['origin_id', 'destination_id', 'W_work', 'W_skills']
    transitions_df = transitions_df[cols]
    transitions_df = transitions_df.set_index(['origin_id', 'destination_id'])
    return transitions_df


def create_skill_similarity_features(origin_ids, destination_ids):
    """create_similarity_distribution_features
    
    Args:
    
    Returns:
    
    """
    skill_stats = []
    for o, d in zip(origin_ids, destination_ids):
        stats = {
                'origin_id': o,
                'destination_id': d,
        }
        try:
            skill_matches = trans_utils.show_skills_overlap(
                o, d, skills_match='optional', verbose=False)
            sims = skill_matches['similarity']
            stats.update({
                'skill_similarity_10pc': np.percentile(sims, 10),
                'skill_similarity_90pc': np.percentile(sims, 90),
                'skill_similarity_mean': sims.mean(),
            }) 
        except:
            stats.update({
                'skill_similarity_10pc': config['features']['mean_skill_similarity_90pc'],
                'skill_similarity_90pc': config['features']['mean_skill_similarity_10pc'],
                'skill_similarity_mean': config['features']['mean_skill_similarity_mean'],
            })
            
        skill_stats.append(stats)
        
    skill_stats = pd.DataFrame.from_records(skill_stats)
    for col in ['skill_similarity_10pc', 'skill_similarity_90pc', 'skill_similarity_mean']:
        skill_stats[col] = skill_stats[col].fillna(skill_stats[col].mean())
    
    skill_stats = skill_stats.set_index(['origin_id', 'destination_id'])
    return skill_stats

        
def create_job_similarity_feature(origin_ids, destination_ids):
    """create_job_similarity_feature
    
    Args:
    
    Returns:
    
    """
    
    description_sims = []
    for o, d in zip(origin_ids, destination_ids):
        v_o = occupation_embeddings[o]
        v_d = occupation_embeddings[d]
        description_sims.append(cosine(v_o, v_d))
        
    sims = pd.DataFrame({'origin_id': origin_ids, 
                         'destination_id': destination_ids,
                         'description_similarity': description_sims,
                        })
    sims = sims.set_index(['origin_id', 'destination_id'])
    return sims

        
def create_features(origin_ids, destination_ids):
    features = [
        create_transition_features(origin_ids, destination_ids),
        create_skill_similarity_features(origin_ids, destination_ids),
        create_job_similarity_feature(origin_ids, destination_ids)
    ]
    
    feature_df = pd.concat(features, axis=1)

    feature_order = [
	'W_work',
	'description_similarity',
	'W_skills',
	'skill_similarity_mean',
	'skill_similarity_10pc', 
	'skill_similarity_90pc']
    return feature_df[feature_order]

    
def predict(origin_ids, destination_ids):
    """predict
    
    Args:
    
    Returns:
    """
    model = joblib.load(os.path.join(useful_paths.models_dir, 'feasibility_model.pkl'))
    
    features = create_features(origin_ids, destination_ids)
    
    ratings = model.predict(features)
    min_feasibility = config['predict']['min_feasibility']
    is_feasible = ratings >= min_feasibility

    pred_df = pd.DataFrame({
        'origin_id': origin_ids,
        'destination_id': destination_ids,
        'feasibility_predicted': ratings,
        'is_feasible_predicted': is_feasible,
        })

    return pred_df
