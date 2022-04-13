from flask import Flask, request
from abstraction_function import word2vec_equals


app = Flask(__name__)

import json
# call to route /equals executes equalRoute function
# use uRL, DOM String, Dom content and DOM syntax tree as params
@app.route('/equals', methods=('POST', ))
def equalRoute():
    # get params sent by java
    parametersJava = request.json
    obj1 = parametersJava['obj1']
    obj2 = parametersJava['obj2']

    # compute equality of DOM objects
    result = word2vec_equals(obj1, obj2)

    result = "true" if result else "false"
    
    # return true if the two objects are the same
    return result