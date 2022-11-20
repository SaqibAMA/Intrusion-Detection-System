import flask
import argparse

app = flask.Flask(__name__)

@app.route('/api/xgboost')
def index():
    return flask.jsonify({'message': 'Hello World!'})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False)
    args = parser.parse_args()

    # if --debug is passed, run in debug mode
    app.run(debug=args.debug)