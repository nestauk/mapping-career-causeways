# -*- coding: utf-8 -*-
import logging
import mapping_career_causeways
import mapping_career_causeways.load_data_utils as load_data
import pandas as pd
import pickle

data = load_data.Data()
data_path = f'{mapping_career_causeways.Paths().data_dir}'

logger = logging.getLogger(__name__)

def main():
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data
    """
    occupation_skill_lists()
    return

def occupation_skill_lists():
    """
    Create essential and optional skill lists for each occupation, and
    dataframes with skills of different types (necessary for skills comparison)
    """

    # Import data on occupations and skills
    occupations = data.occupations
    skills = data.skills
    occupation_to_skills = data.occupation_to_skills

    # Lists of essential and optional skills IDs for each parent node (occupations)
    all_essential_items = []
    all_optional_items = []

    for i, row in occupations.iterrows():
        essential_items = occupation_to_skills[
            (occupation_to_skills.occupation_id==row.id) &
            (occupation_to_skills.importance=='Essential')
        ].skill_id.to_list()
        all_essential_items.append(sorted(essential_items))

        optional_items = occupation_to_skills[
            (occupation_to_skills.occupation_id==row.id) &
            (occupation_to_skills.importance=='Optional')
        ].skill_id.to_list()
        all_optional_items.append(sorted(optional_items))

    # Lists of all skill IDs for each occupation
    all_items = []
    for j in range(len(occupations)):
        all_items.append(sorted(list(
            set(all_essential_items[j]).union(set(all_optional_items[j]))
            )))

    logger.info('Lists of essential, optional and all skills have been created!')

    ### Create dataframes with occupation nodes and their children nodes (skills)

    # Dataframe for essential children nodes
    node_to_essential_items = pd.DataFrame(data={
        'id': occupations.id,
        'occupation': occupations.preferred_label,
        'items_list': all_essential_items,
        'sector': data.occ.isco_level_1
    })

    # Dataframe for optional children nodes
    node_to_optional_items = pd.DataFrame(data={
        'id': occupations.id,
        'occupation': occupations.preferred_label,
        'items_list': all_optional_items,
        'sector': data.occ.isco_level_1
    })

    # Dataframe for all children nodes
    node_to_all_items = pd.DataFrame(data={
        'id': occupations.id,
        'occupation': occupations.preferred_label,
        'items_list': all_items,
        'sector': data.occ.isco_level_1
    })

    # Save the lists of processed dataframes
    pickle.dump(node_to_essential_items, open(f'{data_path}processed/occupation_to_essential_skills.pickle', 'wb'))
    pickle.dump(node_to_optional_items, open(f'{data_path}processed/occupation_to_optional_skills.pickle', 'wb'))
    pickle.dump(node_to_all_items, open(f'{data_path}processed/occupation_to_all_skills.pickle', 'wb'))
    logger.info('Dataframes of children nodes have been created!')

if __name__ == "__main__":

    try:
        msg = f"Creating occupation-skills dataframes..."
        logger.info(msg)
        main()
    except (Exception, KeyboardInterrupt) as e:
        logger.exception("Failed!", stack_info=True)
        raise e
