import flask
import argparse
import XGBoost.XGBoost_infer as xgb_inference
import XGBoost.XGBoost_train as xgb_train

app = flask.Flask(__name__)

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"]


# GET route
@app.route("/")
def index():
    return flask.jsonify({"message": "IDSAPI is up and running!"})

# GET route
@app.route('/api/xgboost')
def api():
    # get the data from the request body
    request_json = flask.request.json
    if request_json:
        inputs = xgb_train.convert_to_training_format(request_json)
        prediction = xgb_inference.predict(
            duration=inputs[0],
            protocol_type=inputs[1],
            service=inputs[2],
            flag=inputs[3],
            src_bytes=inputs[4],
            dst_bytes=inputs[5],
            land=inputs[6],
            wrong_fragment=inputs[7],
            urgent=inputs[8],
            hot=inputs[9],
            num_failed_logins=inputs[10],
            logged_in=inputs[11],
            num_compromised=inputs[12],
            root_shell=inputs[13],
            su_attempted=inputs[14],
            num_root=inputs[15],
            num_file_creations=inputs[16],
            num_shells=inputs[17],
            num_access_files=inputs[18],
            num_outbound_cmds=inputs[19],
            is_host_login=inputs[20],
            is_guest_login=inputs[21],
            count=inputs[22],
            srv_count=inputs[23],
            serror_rate=inputs[24],
            srv_serror_rate=inputs[25],
            rerror_rate=inputs[26],
            srv_rerror_rate=inputs[27],
            same_srv_rate=inputs[28],
            diff_srv_rate=inputs[29],
            srv_diff_host_rate=inputs[30],
            dst_host_count=inputs[31],
            dst_host_srv_count=inputs[32],
            dst_host_same_srv_rate=inputs[33],
            dst_host_diff_srv_rate=inputs[34],
            dst_host_same_src_port_rate=inputs[35],
            dst_host_srv_diff_host_rate=inputs[36],
            dst_host_serror_rate=inputs[37],
            dst_host_srv_serror_rate=inputs[38],
            dst_host_rerror_rate=inputs[39],
            dst_host_srv_rerror_rate=inputs[40]
        )
        print(prediction)
        return flask.jsonify({
            "is_attack": str(prediction[0] > xgb_inference.threshold),
            "probability": str(prediction[0])
        })
    # If no request_json, return a 500 error
    return flask.Response(status=500)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False)
    args = parser.parse_args()

    # if --debug is passed, run in debug mode
    app.run(debug=args.debug)