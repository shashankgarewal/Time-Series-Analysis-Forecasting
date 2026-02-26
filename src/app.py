from flask import Flask, jsonify
import data as data_module
import config

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def train():
    df, df_train, df_test = data_module.load(config.TICKER, config.START_DATE)
    features_full = data_module.build_features(df)
    features_train = data_module.build_features(df_train)
    
    
    
    return jsonify({"rows": len(features_train)})

if __name__ == "__main__":
    app.run(debug=True)
    