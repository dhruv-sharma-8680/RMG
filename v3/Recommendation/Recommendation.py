import numpy as np
import pandas as pd
import implicit
import random 
# from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score
import scipy.sparse as sparse
from tqdm import tqdm

class Recommendation:
    def __init__(self, input_dataframe, feature_columns={"item_column":None, 
                                                         "user_column": None, 
                                                         "score_column":None,
                                                         "item_description_column":None}, isPreprocess=False, isSum=True):
        self.feature_columns = feature_columns
        self.df_columns = list(self.feature_columns.values())
        self.input_dataframe = input_dataframe[self.df_columns]
        self.isPreprocess = isPreprocess
        self.isSum=isSum
        if self.isPreprocess:
            self.preprocess(isSum=isSum)
        if len(self.feature_columns) == 4:
            self.item_lookup = self.input_dataframe[[self.feature_columns["item_column"], self.feature_columns["item_description_column"]]].drop_duplicates()
    
    def preprocess(self, isSum=True):
        if isSum:
            cleaned_df = self.input_dataframe.groupby([self.feature_columns['user_column'], 
                                                   self.feature_columns['item_column']]).sum().reset_index()
        else:
            cleaned_df = self.input_dataframe.groupby([self.feature_columns['user_column'], 
                                                   self.feature_columns['item_column']]).mean().reset_index()
            
        cleaned_df[self.feature_columns['score_column']].loc[cleaned_df[self.feature_columns['score_column']] == 0] = 1
        grouped_purchased = cleaned_df.query(self.feature_columns['score_column']+' > 0')
        
        # Creating the sparse matrix

        self.user = list(np.sort(grouped_purchased[self.feature_columns['user_column']].unique())) # Get our unique customers
        self.item = list(grouped_purchased[self.feature_columns['item_column']].unique()) # Get our unique products that were purchased
        self.score = list(grouped_purchased[self.feature_columns['score_column']]) # All of our purchases


        # Get the associated row & column indices
        self.row_indices = grouped_purchased[self.feature_columns['user_column']].astype('category', categories = self.user).cat.codes 
        self.col_indices = grouped_purchased[self.feature_columns['item_column']].astype('category', categories = self.item).cat.codes 

        self.sparse_matrix = sparse.csr_matrix((self.score, (self.row_indices, self.col_indices)), shape=(len(self.user), len(self.item)))
        self.isPreprocess=True
    
    def describe_sparse_matrix(self, isSum=True):
        if self.isPreprocess == False:
            self.preprocess(isSum=isSum)
        self.matrix_size = self.sparse_matrix.shape[0]*self.sparse_matrix.shape[1] 
        num_purchases = len(self.sparse_matrix.nonzero()[0]) 
        self.sparsity = 100*(1 - (num_purchases/self.matrix_size))
        print("Maximum possible sparsity for collaborative filtering approach is 99.5%.")
        print()
        if self.sparsity < 99.5:
            print("Sparsity is ~"+str(round(self.sparsity,2))+", we can expect decent results.")
        else:
            print("Sparsity is "+str(round(self.sparsity,2)) + "%, Try to reduce as per the standard")
    
    def create_split_data(self, score, pct_test = 0.2):
        
        self.test_set = score.copy() 
        self.test_set[self.test_set != 0] = 1 

        self.training_set = score.copy() 

        nonzero_inds = self.training_set.nonzero() 
        nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) 

        random.seed(0) 

        num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) 
        samples = random.sample(nonzero_pairs, num_samples) 

        self.user_inds = [index[0] for index in samples] 
        self.item_inds = [index[1] for index in samples] 

        self.training_set[self.user_inds, self.item_inds] = 0 
        self.training_set.eliminate_zeros() 

        return self.training_set, self.test_set, self.user_inds
    
    def train(self, factors, alpha, regularization=0.1, iterations=50, num_threads=1, isEvaluate=False):
        self.describe_sparse_matrix(isSum=self.isSum)
        train_set, _, _ = self.create_split_data(self.sparse_matrix)
        self.als = implicit.als.AlternatingLeastSquares(factors=factors, 
                                     regularization = regularization, 
                                     iterations = iterations,
                                    num_threads=num_threads)
        self.als.fit(train_set.transpose() * alpha)
        print("Training completed...")
        if isEvaluate:
            _f1_score, _roc_auc_score = self.get_f1_score()
            print(" F1 Score: ",_f1_score, " | AUC ROC score: ", _roc_auc_score)
            return _f1_score, _roc_auc_score
    
    def get_recommendation(self, user_id, is_lookup_items=False, N=10):
        lookup_flag = False
        if is_lookup_items :
            if len(self.feature_columns) == 4:
                lookup_flag=True
            else:
                print("Instantiate object with four columns. See the docs for item lookups.")
            
        idx = np.where(np.array(self.user)==int(user_id))[0][0]
        recommendations = self.als.recommend(idx, self.training_set, N)
        recommendations_list = []
        score_list = []
        item_lookup_list = []
        for name_id, score in recommendations:
            recommendations_list.append(self.item[name_id])
            score_list.append(score)
            if lookup_flag:
                item_lookup_list.append(self.item_lookup[self.feature_columns["item_description_column"]].loc[self.item[name_id] == self.item_lookup[self.feature_columns["item_column"]]].iloc[0])
        return recommendations_list, score_list, item_lookup_list
    
    def get_similar_items(self, item_id, is_lookup_items=False, N=10):
        lookup_flag = False
        if is_lookup_items :
            if len(self.feature_columns) == 4:
                lookup_flag=True
            else:
                print("Instantiate object with four columns. See the docs for item lookups.")
        idx = np.where(np.array(self.item)==str(item_id))[0][0]
        items = self.als.similar_items(idx, N)
        item_list = []
        score_list = []
        item_lookup_list = []
        for name_id, score in items:
            item_list.append(self.item[name_id])
            score_list.append(score)
            if lookup_flag:
                item_lookup_list.append(self.item_lookup[self.feature_columns["item_description_column"]].loc[self.item[name_id] == self.item_lookup[self.feature_columns["item_column"]]].iloc[0])
        return item_list, score_list, item_lookup_list
    
    def calc_f1_score(self, predictions, test):
        score = f1_score(test, predictions, average="weighted")
        return score
    
    def get_f1_score(self):
        f1_score_list = []
        roc_auc_score_list =[]
        for user in tqdm(self.user_inds):
            train_rows = self.training_set[user].toarray().reshape(-1)
            zero_idxs = np.where(train_rows == 0)

            #get prediction based on user/item sparse matrix
            user_vec = self.als.user_factors[user,:]
            pred = user_vec.dot(self.als.item_factors.T)[zero_idxs]
            pred_bin = np.ones(shape=pred.shape)
            pred_bin[np.where(pred<0.5)] = 0

            true_pred = self.test_set[user, :].toarray()[0,zero_idxs].reshape(-1)
            f1_score_list.append(f1_score(true_pred, pred_bin))
            roc_auc_score_list.append(roc_auc_score(true_pred, pred_bin))
        return np.mean(f1_score_list), np.mean(roc_auc_score_list)

        
    