import pandas as pd
import numpy as np
from time import time
from itertools import combinations_with_replacement
import pickle
import mapping_career_causeways.compare_nodes_utils as compare_nodes_utils
import os
import boto3
from ast import literal_eval
import sys

## SETUP

# Get the skill integer to check
if len(sys.argv) < 2:
    print('Core skill integer missing!')
    raise
else:
    j_skill = int(sys.argv[1])

# Set up AWS params
df_keys = pd.read_csv('../../private/karlisKanders_accessKeys.csv')
os.environ["AWS_ACCESS_KEY_ID"] = df_keys['Access key ID'].iloc[0]
os.environ["AWS_SECRET_ACCESS_KEY"] = df_keys['Secret access key'].iloc[0]

bucket_name = 'ojd-temp-storage'
s3_output_folder = 'outputs_Essential/'

s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')
my_bucket = s3_resource.Bucket(name=bucket_name)

# Set up folder for temporary data
data_folder = '../../data/temp_files/'
if os.path.exists(data_folder) == False:
    os.mkdir(data_folder)

# Load embeddings and lists of skills items of each occupation
files_to_download = [
    'embeddings_skills_description_SBERT.npy',
    'topOccupation_to_all_skills.pickle',
    'topOccupation_to_essential_skills.pickle',
    'sorted_core_skills_id.pickle']

for file in files_to_download:
    if os.path.exists(data_folder + file) == False:
        s3_resource.Object(bucket_name=bucket_name, key=file).download_file(data_folder + file)

embeddings = np.load(data_folder + files_to_download[0])
node_to_essential_items_Top = pickle.load(open(data_folder + files_to_download[2], 'rb'))
node_to_all_items_Top = pickle.load(open(data_folder + files_to_download[1], 'rb'))
sorted_core_skills = pickle.load(open(data_folder + files_to_download[3], 'rb'))

n_occupations = len(node_to_essential_items_Top)

## ANALYSIS

## Select skill to add to origin occupation's skill-set
skill_id = sorted_core_skills[j_skill]

## Set up "origin" sector and "destination" sectors

# Origin nodes: here, only ESSENTIAL items
from_node_to_items =  node_to_essential_items_Top.copy()
from_node_to_items.sector = 'origin'

# Add the extra skill to job_i skillset
t = time()
skill_added_to = []
new_items_list = []
for job_i, row in from_node_to_items.iterrows():

    # Original skillset of the origin occupation
    origin_skillset = row.items_list.copy()

    # Check if skills is not already in the skillset
    if skill_id not in origin_skillset:
        list_of_skills = sorted([skill_id] + origin_skillset)
        new_items_list.append(str(list_of_skills))
        skill_added_to.append(row.original_id)
    else:
        new_items_list.append(str(origin_skillset))

# Re-evaluate all items lists so that they are treated as lists
from_node_to_items.items_list = new_items_list
from_node_to_items.items_list = from_node_to_items.items_list.apply(lambda x: literal_eval(x))
t_elapsed = time()-t
print(f"Added skill #{skill_id} to {len(skill_added_to)} occupations in {t_elapsed:.2f} seconds")

# Destination nodes: only ESSENTIAL items
to_node_to_items =  node_to_essential_items_Top.copy()
to_node_to_items.sector = 'destination'
to_node_to_items.id = to_node_to_items.id + n_occupations

# Combine all into one dataframe
node_to_items = pd.concat([from_node_to_items, to_node_to_items]).reset_index(drop=True)

# Set up the combination of sectors to check
combos = [('origin','destination')]

# Perform the comparison!
comp_all_to_essential = compare_nodes_utils.CompareSectors(
    node_to_items,
    embeddings,
    combos,
    metric='cosine',
    symmetric=False)

t = time()
comp_all_to_essential.run_comparisons(dump=False)
comp_all_to_essential.collect_comparisons()
t_elapsed = time()-t
print('===============')
print(f"Total time elapsed: {t_elapsed:.0f} seconds")

# Select only the edges from origin to destination occupations
W_all_to_essential = comp_all_to_essential.D
print(W_all_to_essential.shape)
i_edges = [edge[0] for edge in comp_all_to_essential.real_edge_list]
from_edges = np.array(comp_all_to_essential.real_edge_list)[np.where(np.array(i_edges)<n_occupations)[0]]

W_perturbed = np.zeros((n_occupations,n_occupations))
for edge in from_edges:
    W_perturbed[edge[0], edge[1]-n_occupations] = W_all_to_essential[edge[0],edge[1]]

# Take care of nulls
W_perturbed[np.isinf(W_perturbed)] = 0

# Save
output_file_name = f"W_perturbed_essential_{j_skill}_Skill_{skill_id}.npy"
np.save(data_folder + output_file_name, W_perturbed)

# Upload to S3
s3_resource.Object(bucket_name, s3_output_folder + output_file_name).upload_file(Filename=data_folder + output_file_name)
