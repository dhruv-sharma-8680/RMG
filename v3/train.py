from Recommendation.Recommendation import Recommendation as Rec
import pandas as pd
import numpy as np
import argparse
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--factors",
    type=int,
    default=150,
    help="Number of factors for matrix factorization. Argument for training"
)
parser.add_argument(
    "--alpha",
    type=float,
    default=23.0,
    help="Scaling factor for matrix"
)
parser.add_argument(
    "--regularization",
    type=float,
    default=0.1,
    help="regularization parameter for training"
)
parser.add_argument(
    "--iterations",
    type=int,
    default=200,
    help="training iterations for train def "
)
parser.add_argument(
    "--num_of_threads",
    type=int,
    default=1,
    help="number of threads used for training provided by system"
)
parser.add_argument(
    "--item_column_name",
    type=str,
    default="StockCode",
    help="Column name of item column in dataframe"
)
parser.add_argument(
    "--user_column_name",
    type=str,
    default="CustomerID",
    help="Column name of User column in dataframe"
)
parser.add_argument(
    "--score_column_name",
    type=str,
    default="Quantity",
    help="Column name of score column in dataframe"
)
parser.add_argument(
    "--item_description_column_name",
    type=str,
    default=None,
    help="Column name of item description column in dataframe"
)
parser.add_argument(
    "--csv_path",
    type=str,
    default="./rec.csv",
    help="CSV path of recommendation dataset"
)
parser.add_argument(
    "--isExcel",
    type=bool,
    default=True,
    help="Is dataset type excel"
)
parser.add_argument(
    "--model_save_path",
    type=str,
    default="recommendation.rec",
    help="It is a Recommendation class object saved in a pickle file with extension .rec"
)
parser.add_argument(
    "--isEvaluate",
    type=bool,
    default=True,
    help="Check the model auc roc score and F1 score after training"
)
args = parser.parse_args()

if args.isExcel:
    retail_data = pd.read_excel(args.csv_path)
else:
    retail_data = pd.read_csv(args.csv_path)

cleaned_retail = retail_data.loc[pd.isnull(retail_data[args.user_column_name]) == False]

print("Dataset loaded...")

if args.item_description_column_name == None:
    feature_columns = {"item_column":args.item_column_name, 
                    "user_column": args.user_column_name, 
                    "score_column":args.score_column_name}
else:
    feature_columns = {"item_column":args.item_column_name, 
                    "user_column": args.user_column_name, 
                    "score_column":args.score_column_name,
                    "item_description_column": args.item_description_column_name}
rec = Rec(cleaned_retail, feature_columns=feature_columns, isPreprocess=False)

rec.train(factors=args.factors, 
            alpha=args.alpha, 
            regularization=args.regularization, 
            iterations=args.iterations, 
            num_threads=args.num_of_threads,
            isEvaluate=args.isEvaluate)

#Save model
with open(args.model_save_path, 'wb') as output:
    pkl.dump(rec, output, pkl.HIGHEST_PROTOCOL)
    print("Model saved at :", args.model_save_path)