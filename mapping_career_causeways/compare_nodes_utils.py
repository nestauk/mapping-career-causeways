#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:06:16 2019

@author: karliskanders

Last updated on 31/03/2020
"""

import numpy as np
from numba import njit
from scipy.spatial.distance import pdist, squareform
import time
import pickle
import re
import ast

class PrepInputs:
    """
    Class that prepares inputs for comparing parent data items ("nodes") based
    on their children nodes ("items"). For example, to compare technical skills
    and competencies based on their underlying knowledge items.
    An instance of this class is then fed as an input to the CompareNode class
    constructor (see further below).
    """

    def __init__(self, dataframe, embeddings, sectors=None, metric='cosine'):
        """
        Parameters
        ----------
        dataframe (pandas.DataFrame):
            Dataframe that needs to have three columns called 'id', 'sector' and
            'items_list'. Column 'id' specifies the parent node's unique ID number.
            Column 'sector' specifies higher-level categories of the parent nodes
            (for example, sector to which the technical skills and competency has
            been assigned to).  Column 'items_list' contains ID numbers of the
            children nodes that form the basis of the comparison.
        embeddings (numpy.ndarray):
            Array containing embeddings or similarity matrix of the children nodes
            featured in the 'items_list'.
        metric (string):
            Specifies the method used to compare the embeddings; if metric is set to
            'precomputed' then 'embeddings' will be interpreted as a similarity matrix.
        sectors (list of tuples of two elements):
            Specifies parent nodes that will be included in the comparison (based
            on the 'sector' column of the dataframe desribed above).

        """
        self.node_to_items = self.select_sector_data(dataframe, sectors)
        self.sectors = sectors
        self.metric = metric
        # Prepare the inputs immediately in the constructor so that we don't
        # have to store the embeddings in the memory
        self.prep_inputs(embeddings)

    def prep_inputs(self,embeddings):
        """
        Selects the parent nodes that we wish to compare (specified by the
        self.sectors variable) and then computes the similarity
        matrix of the children nodes pertaining to the selected parent nodes.
        Finally, it also keeps track of the original node IDs and how they map
        to the rows/columns of the similarity matrix.
        """
        # All items (children) that we will use
        items_sample = np.array(sorted(list(set([k for items in self.node_to_items.items_list.to_list() for k in items]))))

        # Obtain similarity matrix for the sampled item embeddings
        if self.metric == 'precomputed':
            self.D = embeddings[items_sample,:]
            self.D = self.D[:, items_sample]
        else:
            emb_sample = embeddings[items_sample,:]
            self.D = 1 - squareform(pdist(emb_sample, metric=self.metric))

        # Link between the original item ids and the smaller distance matrix row/column indices
        self.items_lookup = dict(zip(items_sample, range(len(items_sample))))

        # For each sector, collect sector's node's items in lists of lists
        # Note that the items will be numbered according to the distance matrix row/column indices
        self.sector_node_to_matrix = []
        # Additionally, collect each sector nodes' id in separate lists
        self.sector_node_ids = []
        for sector in self.sectors:
            # All lists of items belonging to nodes of a specific sector
            node_to_items_list = self.node_to_items[self.node_to_items.sector==sector].items_list.to_list()
            node_ids = self.node_to_items[self.node_to_items.sector==sector].id.to_list()
            node_to_matrix = []
            for x in node_to_items_list:
                node_to_matrix.append([self.items_lookup[i] for i in x])
            self.sector_node_to_matrix.append(node_to_matrix)
            self.sector_node_ids.append(node_ids)

    def select_sector_data(self, dataframe, sectors):
        """
        Select the chosen sectors from the dataframe
        """
        if sectors is not None:
            # Select sectors
            f = np.array(dataframe.sector==sectors[0])
            for s in sectors[1:]:
                f = f | np.array(dataframe.sector==s)
            return dataframe[f].reset_index()
        else:
            return dataframe


#############################
@njit
def all_indices(shape):
    """
    Lists of all possible indices for a matrix of the given shape;
    I found it necessary to write this trivial function to work around
    the limitations of numba. See CompareNodes.find_best_matches() for usage.
    """
    rows = np.array([0] * shape[0]*shape[1])
    cols = np.array([0] * shape[0]*shape[1])
    k=0
    for i in range(shape[0]):
        for j in range(shape[1]):
            rows[k] = i
            cols[k] = j
            k += 1
    return (rows,cols)

class CompareNodes:
    """
    Class that implements the pair-wise comparison and similarity measurement
    between parent nodes based on the similarities of their children nodes.
    """

    def __init__(self, prep, weighting_params=[-43.5, 50],
                 save_matched_items=False,
                 matching_method='one_to_one'):
        """
        Parameters
        ----------
        prep (instance of PrepInputs class):
            Contains the inputs for calculating comparisons between parent nodes.
        weighting_params (iterable of two floats OR None):
            Parameters of a sigmoidal function that is used to filter the
            computed similarity values. The default values of [-43.5, 50] have
            been manually selected to provide rather conservative estimates
            of similarity. These values are practically filtering out raw
            similarity values below 0.8; we noticed that for our sentence-BERT
            embedding comparisons, the quality of the matches started declining
            quite rapidly around cosine similarity ~ 0.8. However, you might
            want to tweak these parameter values for your specific application.
            If weighting_params==None then unweighted, raw similarity
            values will be used for deriving the final similarity values.
        save_matched_items (boolean):
            If True, saves the raw similarity scores for matched items for each
            pair of parent nodes (might be useful for debugging).
        matching_method (string):
            Can be either 'one_to_one' or 'one_to_many'. Determines the
            approach used to find the best matches (see the corresponding functions
            further below).

        """
        self.D = prep.D
        # Similarity matrix encoding similarities between the children nodes

        self.node_to_matrix_sector_i = prep.sector_node_to_matrix[0]
        self.node_to_matrix_sector_j = prep.sector_node_to_matrix[1]
        # List of lists, of the same length as the number of parent nodes, where
        # each sub-list contains the rows/columns of D pertaining to the parent node

        self.sector_node_ids = prep.sector_node_ids
        # Original parent node IDs

        self.save_matched_items = save_matched_items
        self.matched_items = []
        # Variables for fine-grained outputs (useful for debugging)

        self.n_i = [len(ii) for ii in prep.sector_node_to_matrix[0]]
        self.n_j = [len(ii) for ii in prep.sector_node_to_matrix[1]]
        # Number of items per each node

        self.matching_method = matching_method
        self.w = weighting_params
        self._weighted_match_score = None
        self._unweighted_match_score = None
        self.sim=[]
        self.sim_edge_list = []
        self.edge_list = []
        self.edge_matched_scores = []

    ### Item matching methods
    def compare_nodes(self):
        """
        Routine to compare and find best pairwise matches of "children" nodes’ embeddings
        """
        # Number of nodes
        N_i = len(self.node_to_matrix_sector_i)
        N_j = len(self.node_to_matrix_sector_j)

        for node_i in range(N_i):
            ii = self.node_to_matrix_sector_i[node_i]
            n_i = self.n_i[node_i]
            for node_j in range(N_j):

                self.edge_list.append([node_i, node_j])

                jj = self.node_to_matrix_sector_j[node_j]
                n_j = self.n_j[node_j]

                # In the edge case if there are nodes with no items
                if (n_i==0) or (n_j==0):
                    self.edge_matched_scores.append(np.array([0]))
                    if self.save_matched_items:
                        self.matched_items.append(('none'))
                    continue

                # Select the distances between the relevant items
                D_ = self.D[ii,:]
                D_ = D_[:,jj]
                if self.matching_method == 'one_to_one':
                    matched_scores, i_item_matched, j_item_matched = np.array(self.find_best_matches(D_, n_i, n_j))
                elif self.matching_method == 'one_to_many':
                    matched_scores, i_item_matched, j_item_matched = np.array(self.one_to_many_matches(D_, n_i, n_j))

                # Save matched items if fine-grained outputs are required
                # (Note: this will increase memory overhead)
                if self.save_matched_items:
                    self.matched_items.append((i_item_matched.astype(int), j_item_matched.astype(int)))

                self.edge_matched_scores.append(matched_scores)

    @staticmethod
    @njit
    def find_best_matches(D_, n_i, n_j):
        """
        Finds best one-to-one matches for two sets of children nodes based on
        their pairwise similarities.

        Parameters
        ----------
        D_ (numpy.ndarray):
            Similarity matrix between the specific children nodes pertaining
            to the two parent nodes.
        n_i (int):
            Number of children nodes for parent i.
        n_j (int):
            Number of children nodes for parent j.

        Returns
        -------
        matched_scores (list of floats):
            Similarity values for the best matches
        i_item_matched (list of int):
            IDs of the matched child nodes from parent i.
        j_item_matched (list of int):
            IDs of the matched child nodes from parent j.

        """
        # Sort the distances
        indices = all_indices(D_.shape)
        scores = D_.flatten()
        order = np.argsort(scores)
        ordered_i_items = indices[0][order]
        ordered_j_items = indices[1][order]
        ordered_scores = scores[order]
        # Find best matching items
        i_item_matched = []
        j_item_matched = []
        matched_scores = []
        k = len(ordered_scores)
        while (len(i_item_matched) < n_i) and (len(j_item_matched) < n_j):
            k -= 1
            if ((ordered_i_items[k] in i_item_matched) or (ordered_j_items[k] in j_item_matched)):
                continue # Destination item already matched or origin item already used
            i_item_matched.append(ordered_i_items[k])
            j_item_matched.append(ordered_j_items[k])
            matched_scores.append(ordered_scores[k])
        return matched_scores, i_item_matched, j_item_matched

    @staticmethod
    @njit
    def one_to_many_matches(D_, n_i, n_j):
        """
        Finds best, asymmetric one-to-many matches between each of parent's j
        child nodes and any of parent's i child nodes, based on their pairwise
        similarities. Therefore, all of child nodes of parent j will be matched
        exactly once, whereas some of the parent i child nodes might be used
        several times.
        Note: This function is almost entirely a duplicate of find_best_matches(),
        and thus these two should probably be merged.

        Parameters
        ----------
        D_ (numpy.ndarray):
            Similarity matrix between the specific children nodes pertaining
            to the two parent nodes.
        n_i (int):
            Number of children nodes for parent i.
        n_j (int):
            Number of children nodes for parent j.

        Returns
        -------
        matched_scores (list of floats):
            Similarity values for the best matches
        i_item_matched (list of int):
            IDs of the matched child nodes from parent i.
        j_item_matched (list of int):
            IDs of the matched child nodes from parent j.

        """
        # Sort the distances
        indices = all_indices(D_.shape)
        scores = D_.flatten()
        order = np.argsort(scores)
        ordered_i_items = indices[0][order]
        ordered_j_items = indices[1][order]
        ordered_scores = scores[order]
        # Find best matching items
        i_item_matched = []
        j_item_matched = []
        matched_scores = []
        k = len(ordered_scores)
        while (len(j_item_matched) < n_j):
            k -= 1
            if (ordered_j_items[k] in j_item_matched):
                continue # Destination item already matched
            i_item_matched.append(ordered_i_items[k])
            j_item_matched.append(ordered_j_items[k])
            matched_scores.append(ordered_scores[k])
        return matched_scores, i_item_matched, j_item_matched

    @staticmethod
    @njit
    def weighting_function(x, intercept=-18.95608773, coeff=26.92733767):
        """
        Sigmoidal function used for filtering raw similarity values.
        The default coefficients have been inferred from analysing ESCO skill
        title synonyms - they should not be expected to work well with other datasets.
        """
        return 1/(1 + np.exp(-(intercept + x*coeff)))

    @property
    def weighted_match_score(self):
        """
        Aggregates the filtered matched scores for each pair of parent nodes, to be
        further used in calculating the final similarity values.
        """
        if self._weighted_match_score is None:
            self._weighted_match_score = np.zeros((len(self.edge_matched_scores),))
            for i, matched_scores in enumerate(self.edge_matched_scores):
                self._weighted_match_score[i] = np.sum(self.weighting_function(matched_scores,intercept=self.w[0],coeff=self.w[1]))
        return self._weighted_match_score

    @property
    def unweighted_match_score(self):
        """
        Aggregates the raw matched scores for each pair of parent nodes, to be
        further used in calculating the final similarity values.
        """
        if self._unweighted_match_score is None:
            self._unweighted_match_score = np.zeros((len(self.edge_matched_scores),))
            for i, matched_scores in enumerate(self.edge_matched_scores):
                self._unweighted_match_score[i] = np.sum(matched_scores)
        return self._unweighted_match_score

    def symmetric_similarity(self):
        """
        Calculates a symmetric similarity measure by taking the aggregated
        matched scores and dividing them by the max number of children nodes
        across both parents. In a way, this functions like a graded version of
        the Jaccard similarity coefficient.
        """
        # List that will hold the final similarity values
        self.sim = [0] * len(self.edge_list)
        self.sim_edge_list = self.edge_list.copy() # Not sure if this is used anymore
        # List that will hold the original IDs of the parent nodes
        self.real_edge_list = [0] * len(self.edge_list)

        # Go through each comparison between parent nodes
        for i, indices in enumerate(self.edge_list):
            self.real_edge_list[i] = [self.sector_node_ids[0][indices[0]], self.sector_node_ids[1][indices[1]]]

            # For comparison with self
            if self.real_edge_list[i][0] == self.real_edge_list[i][1]:
                self.sim[i] = 1
                continue

            # Calculate the final similarity value
            denominator = max(self.n_i[indices[0]], self.n_j[indices[1]])
            if self.w is not None:
                self.sim[i] = self.weighted_match_score[i] / denominator
            else:
                self.sim[i] = self.unweighted_match_score[i] / denominator

        self.real_edge_list += [[edge[1],edge[0]] for edge in self.real_edge_list]
        self.sim += self.sim

    def asymmetric_similarity(self):
        """
        Calculates an asymmetric similarity measure by taking the aggregated
        matched scores and dividing them by the number of children nodes
        each parent has. In this way, we account for cases where there might be
        a different number of children nodes present for each parent.
        """
        # List that will hold the final similarity values
        self.sim = [0] * (2*len(self.edge_list))
        self.sim_edge_list = self.edge_list.copy() # Not sure if this is used anymore
        # List that will hold the original IDs of the parent nodes
        self.real_edge_list = [0] * (2*len(self.edge_list))

        # Go through each comparison between parent nodes
        k = 0
        for i, indices in enumerate(self.edge_list):

            self.real_edge_list[k] = [self.sector_node_ids[0][indices[0]], self.sector_node_ids[1][indices[1]]]
            self.real_edge_list[k+1] = [self.sector_node_ids[1][indices[1]], self.sector_node_ids[0][indices[0]]]

            # For comparison with self
            if self.real_edge_list[k][0] == self.real_edge_list[k][1]:
                self.sim[k] = 1
                self.sim[k+1] = 1
            else:
                # Calculate the final similarity values
                if self.w is not None:
                    sim = self.weighted_match_score[i]
                else:
                    sim = self.unweighted_match_score[i]
                # from i to j
                self.sim[k] = sim / self.n_j[indices[1]]
                # from j to i
                self.sim[k+1] = sim / self.n_i[indices[0]]
            k += 2

    def similarity_matrix(self):
        """
        Generates a similarity matrix between all parent nodes
        (Doesn't work when working with asymetric similarities!)
        """
        D_tsc = np.zeros((len(self.node_to_matrix_sector_i),len(self.node_to_matrix_sector_j)))
        for i, indices in enumerate(self.edge_list):
            sim = self.sim[i]
            D_tsc[indices[0],indices[1]] = sim
        return D_tsc


def find_closest(i, similarity_matrix, df):
    """
    Method for reporting the closest neighbours to a node i given a similarity matrix;
    useful during exploratory data analysis.

    Parameters
    ----------
    i (int OR None):
        Determines for which node where are assessing the closest neighbours;
        if i==None, a random node is chosen.
    similarity_matrix (numpy.ndarray):
        Similarity matrix determining the closeness between each pair of nodes.
    df (pandas.DataFrame):
        Dataframe to be used for reporting the closest neighbours; must have then
        same number of rows as the similarity matrix

    Returns
    -------
    df (pandas.DataFrame):
        The same input dataframe with an added column for similarity values
        between node i and the rest of the nodes, ordered in a descending order
        of similarity.

    """
    if type(i) == type(None):
        i = np.random.randint(similarity_matrix.shape[0])
    most_similar = np.flip(np.argsort(similarity_matrix[i,:]))
    similarity = np.flip(np.sort(similarity_matrix[i,:]))

    df = df.copy().loc[most_similar]
    df['similarity'] = similarity
    return df


def two_node_comparison(node_to_items,
                        node_i, node_j,
                        item_df, item_embeddings,
                        metric='cosine',
                        matching_method='one_to_one',
                        weighting_params=[-43.5, 50],
                        symmetric=True):
    """
    Method to compare two parent nodes based on their children, and output a
    detailed breakdown of the comparison; useful for exploratory data analysis.

    Parameters
    ----------
    node_to_items (pandas.DataFrame):
        See the description of parameter 'dataframe' of the
        PrepInputs.__init__() function.
    node_i (int):
        ID of the parent i
    node_j (int):
        ID of the parent j
    item_df (pandas.DataFrame):
        Dataframe containing the titles/labels of the children nodes
    item_embeddings (numpy.ndarray):
        See the description of parameter 'embeddings' of the
        PrepInputs.__init__() function.
    metric (string):
        See the description of parameter 'metric' of the
        PrepInputs.__init__() function.
    matching_method (string):
        See the description of parameter 'matching_method' of the
        CompareNodes.__init__() function.
    weighting_params (iterable of two floats OR None):
        See the description of parameter 'weighting_params' of the
        CompareNodes.__init__() function.
    symmetric (boolean):
        Determines which type of final similarity value will be calculated.

    Returns
    -------
    item_comparison (pandas.DataFrame):
        Dataframe with all matched children nodes and their matching scores
    final_score (float):
        Final similarity value between the parent nodes node_i and node_j
    """

    # Rename sectors to be the two seperate nodes
    node_to_items_ = node_to_items.loc[np.array([node_i, node_j])].copy().reset_index(drop=True)
    node_to_items_['sector'] = 0
    node_to_items_.loc[0, 'sector'] = 'node_i'
    node_to_items_.loc[1, 'sector'] = 'node_j'
    combos = ('node_i', 'node_j')

    if metric=='precomputed':
        w = None
    else:
        w = weighting_params

    prep = PrepInputs(
        node_to_items_,
        item_embeddings,
        list(combos),
        metric=metric)

    # Get the matched items and their scores
    compare_pair = CompareNodes(
        prep,
        weighting_params=w,
        save_matched_items=True,
        matching_method=matching_method)

    compare_pair.compare_nodes()
    sim = compare_pair.edge_matched_scores[0]

    if metric=='precomputed':
        score = sim
    else:
        score = compare_pair.weighting_function(
            sim, intercept=compare_pair.w[0], coeff=compare_pair.w[1])

    # Get the items
    items_i = np.array(prep.node_to_items.items_list.loc[0])[compare_pair.matched_items[0][0]]
    items_j = np.array(prep.node_to_items.items_list.loc[1])[compare_pair.matched_items[0][1]]

    # Create the comparison dataframe
    items_i_df = item_df.loc[items_i].copy()
    items_i_df['item'] = list(range(len(items_i)))

    items_j_df = item_df.loc[items_j].copy()
    items_j_df['item'] = list(range(len(items_j)))

    item_comparison = items_i_df.merge(items_j_df, on='item')
    item_comparison['similarity'] = np.round(score,3)
    if metric != 'precomputed':
        item_comparison['similarity_raw'] = sim

    if symmetric==True:
        final_score = np.sum(score)  / max(len(node_to_items_.loc[0].items_list), len(node_to_items_.loc[1].items_list))
    else:
        final_score = np.sum(score) / len(node_to_items_.loc[1].items_list)

    return item_comparison, final_score

def match_items_to_destination_TSC(job_i, job_j, data, item='knowledge', item_embeddings=None):
    """
    A specialised function for assessing the skills gap for a job transition from
    job_i (the origin) to job_j (the destination). The function performs matching
    between all knowledge/abilities of job_i (across all TSCs) to the knowledge/abilities
    of each separate TSC of job_j. This, therefore, allows for knowledge/abilities
    from different TSCs at the origin to mix, and to obtain matching estimates for
    all destination's TSCs.

    Note that, in contrast to other functions of this package, this function is
    more specialised for one particular purpose. It could, however, be generalised
    if necessary.

    Parameters
    ----------
    job_i (int):
        ID number of the transition's origin job role
    job_j (int):
        ID number of the transition's destination job role
    data (import_data_utils.SSGdata):
        An instance of SSGdata() class
    item (string):
        Either 'knowledge' or 'abilities' depending on the desired comparison
    item_embeddings (numpy.ndarray):
        Sentence embeddings of the knowledge/abilities items

    Returns
    -------
    destination_tsc_scores (numpy.ndarray):
        Array of matching scores for each TSC
    destination_tsc (pandas.DataFrame):
        ID numbers of the destination TSCs
    tsc_comparison_dataframes (list of pandas.DataFrame):
        Dataframes with the matched knowledge/abilities per each destination TSC
    """

    if item == 'knowledge':
        items_df = data.knowledge
    elif item == 'abilities':
        items_df = data.abilities

    # All origin job role's items
    origin_items = data.role_to_items.loc[job_i][item+'_id']

    # Destination role's TSC items
    destination_tsc = data.role_to_tscl.loc[job_j].tscl_list
    destination_tsc_scores = np.zeros((len(destination_tsc),))
    tsc_comparison_dataframes = []

    for j, tsc in enumerate(destination_tsc):
        # Get all tsc items
        node_to_items = data.tsc_to_items[['tscl_id','sector',item+'_id_nested']].copy()
        # Select only the destination job role's tsc items (and a dummy row for origin's items)
        node_to_items = node_to_items.loc[[0, tsc]].reset_index(drop=True)
        # Prepare for comparison
        node_to_items.rename(columns={'tscl_id':'id',item+'_id_nested':'items_list'}, inplace=True)
        node_to_items.id = node_to_items.id.apply(lambda x: int(x[4:]))
        # Add origin items to match
        node_to_items.loc[0,'items_list'] = str(origin_items)
        # Do some juggling to accommodate nested lists
        node_to_items.items_list = node_to_items.items_list.apply(lambda x: str(x))
        node_to_items.items_list = node_to_items.items_list.apply(lambda x: ast.literal_eval(x))

        node_to_items.loc[1, 'id'] = 1

        df, score = two_node_comparison(node_to_items, 0, 1, items_df, item_embeddings, metric='cosine', symmetric=False)

        destination_tsc_scores[j] = score
        tsc_comparison_dataframes.append(df)

    return destination_tsc_scores, destination_tsc, tsc_comparison_dataframes

#############################
class CompareSectors():
    """
    Class that is used do parwise comparisons of parent nodes on a sector-vs-sector
    basis (each parent node must have a label, which we call here its "sector").
    The result of this comparison is a similarity value or a "weight" that describes
    the edges (connections) between parent nodes.

    This is the main class that we are practically interfacing with and the rest
    of the classes and methods are called from here.

    We perform the comparisons on a sector-vs-sector basis because the similarity
    calculations between all parent nodes might take a long time, and hence
    we use sector-vs-sector comparisons as intermediate outputs that can be saved
    seperately as binary .pickle files and then later collected together.

    The typical workflow can be as follows:
    $ comp = CompareSectors( <input your parameters here> )
    $ comp.compare()
    $ similarity_matrix = comp.D

    """

    def __init__(self, node_to_items, embeddings, combos,
                 metric='cosine', weighting_params=[-43.5, 50],
                 symmetric=True, save_name='comparison'):
        """
        Parameters
        ----------
        node_to_items (pandas.DataFrame):
            See the description of parameter 'dataframe' of the
            PrepInputs.__init__() function.
        embeddings (numpy.ndarray):
            See the description of parameter 'embeddings' of the
            PrepInputs.__init__() function.
        combos (iterable of iterable of two strings):
            Combinations of sectors that we want to check. For example, it could
            look like [('accountancy','accountancy'), ('accountancy', 'design'),
            ('design', 'design')]. These sector labels must exist in the 'sector'
            column of the 'node_to_items' dataframe.
        metric (string):
            See the description of parameter 'metric' of the
            PrepInputs.__init__() function.
        weighting_params (iterable of two floats OR None):
            See the description of parameter 'weighting_params' of the
            CompareNodes.__init__() function.
        symmetric (boolean):
            Determines which type of final similarity value will be calculated.
        save_name (string):
            File name to be used if saving the intermediate sector-vs-sector outputs
        """
        self.combos = combos
        self.node_to_items = node_to_items
        self.emb = embeddings
        self.metric = metric
        self.save_name = save_name
        self.w = weighting_params
        self.symmetric = symmetric

        self.prep_list = []
        self.results_list = []
        self._D = None

    def compare(self):
        """
        Shortcut method to run all comparisons and collect them together
        """
        self.run_comparisons()
        self.collect_comparisons()

    @property
    def D(self):
        """
        Final similarity matrix between the parent nodes
        """
        if self._D is None:
            self.similarity_matrix()
        return self._D

    @staticmethod
    def fname(combo, save_name):
        """
        Generates file name for saving the intermediate sector-vs-sector outputs
        """
        return save_name+'_'+re.sub(' ','_',combo[0])+'_vs_'+re.sub(' ','_',combo[1])+'.pickle'

    def run_comparisons(self, dump=False):
        """
        Performs comparisons of parent nodes for each provided combination of sectors

        Parameters
        ----------
        dump (boolean):
            If dump==True, saves the intermediate sector-vs-sector outputs as
            binary .pickle files. These can then be later manually loaded and
            provided to the collect_comparisons() method. This option might be
            useful if calculations take a long time.
        """

        # For each provided combination of sectors
        for i, combo in enumerate(self.combos):
            t = time.time()
            print(combo)
            # Prepare inputs for the comparison analysis
            prep = PrepInputs(self.node_to_items, self.emb, list(combo), metric=self.metric)
            # Do the comparison of parent nodes
            comp = CompareNodes(prep, weighting_params=self.w)
            comp.compare_nodes()
            if self.symmetric == True:
                comp.symmetric_similarity()
            else:
                comp.asymmetric_similarity()
            # Save only variables that are necessary for the next analysis steps
            res = {'sector_node_ids': comp.sector_node_ids,
                   'sim': comp.sim,
                   'sectors': combo,
                   'sim_edge_list': comp.sim_edge_list,
                   'real_edge_list': comp.real_edge_list}
            if not dump:
                self.results_list.append(res)
            else:
                pickle.dump(res, open(self.fname(combo, self.save_name), 'wb'))
            print(f"{time.time()-t:.2f}")

            # Explicitly clean memory
            del prep, comp

    def collect_comparisons(self):
        """
        Collects the different sector-vs-sector comparisons into one pooled list
        of similarity values and keeps track of the original parent node IDs of
        each edge.

        """
        unique_indices = set()
        # Keep track of the original parent node IDs that are included in the
        # analysis
        self.real_edge_list = []
        # List that will hold original IDs of the parent nodes for each edge
        self.pooled_sim = []
        # List that will hold the similarity values (weights) corresponding to
        # the edges in self.real_edge_list

        # Pool together all edges and their similarity values
        for i, res in enumerate(self.results_list):
            unique_indices = unique_indices.union(set(res['sector_node_ids'][0]).union(set(res['sector_node_ids'][1])))

            # Pool edges pertaining to similarity values (using their real indices)
            self.real_edge_list += res['real_edge_list']
            self.pooled_sim += res['sim']

        # List all unique real indices
        self.unique_indices = np.sort(list(unique_indices))
        self.n_nodes = len(self.unique_indices)

        # Select rows of 'node_to_items' dataframe that pertain to the parent nodes
        # included in this comparison
        self.nodes = self.node_to_items[self.node_to_items.id.isin(self.unique_indices)]

    def similarity_matrix(self):
        """
        Creates a similarity matrix that stores the weights between each pair of parent nodes
        """
        # Map real edges to the similarity matrix
        self.unique_indices_lookup = dict(zip(self.unique_indices, range(self.n_nodes)))
        matrix_edge_list = [[]] * len(self.real_edge_list)
        for i, edge in enumerate(self.real_edge_list):
            matrix_edge_list[i] = [self.unique_indices_lookup[edge[0]], self.unique_indices_lookup[edge[1]]]

        self._D = np.zeros((self.n_nodes,self.n_nodes))
        for i, index in enumerate(matrix_edge_list):
            self._D[index[0],index[1]] = self.pooled_sim[i]
