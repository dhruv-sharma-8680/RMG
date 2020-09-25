import pickle as pkl
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
MODEL_PATH = './models/model_v1.rec'
with open(MODEL_PATH, 'rb') as input:
    recommendation_obj = pkl.load(input)
    print("Model loaded...")

@app.route("/Recommendation", methods=["POST"])
def get_recommendation():
    if request.method =='POST':
        
        user_id = request.args.get('user_id')
        print(user_id)
        item, _, desc = recommendation_obj.get_recommendation(int(user_id), is_lookup_items=True, N=10)
        result = pd.DataFrame(data=np.transpose([item, desc]), columns=["StockCode","Description"]).to_json(orient='columns')
        print(result)
        return jsonify(result)

@app.route("/get_similar_items", methods=["POST"])
def get_similar_items():
    if request.method =='POST':
        
        item_id = str(request.args.get('item_id'))
        print(item_id)
        item, _, desc = recommendation_obj.get_similar_items(item_id, is_lookup_items=True, N=10)
        result = pd.DataFrame(data=np.transpose([item, desc]), columns=["StockCode","Description"]).to_json(orient='columns')
        print(result)
        return jsonify(result)
if __name__ == "__main__":
    app.run(port=2000, host="0.0.0.0", debug=True)
