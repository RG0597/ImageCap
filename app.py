
import flask
from flask import Flask,request,render_template
from flask_cors import CORS,cross_origin
import json
from train import predict
import base64

app=Flask(__name__)

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def get_answer():
    try:
        image=request.json['image']
        image=base64.b64decode(image)
        out=predict(image)
        return flask.jsonify(out)
    except Exception as e:
        out=str(e)
        return app.response_class(response=json.dumps(out),status=500,mimetype='application/json')



if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True,port=5000,use_reloader=False)