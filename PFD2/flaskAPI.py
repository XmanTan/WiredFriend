from flask import Flask, jsonify, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import firebase_admin
from firebase_admin import firestore

#Initialize app
cred_obj = firebase_admin.credentials.Certificate('wiredhealth-b104a-firebase-adminsdk-sfm9e-3e450423b4.json')
default_app = firebase_admin.initialize_app(cred_obj)

#Initialize Firestore Client
db = firestore.client()

#Create Classes
class User(object):
    def __init__(self, compound10d, compound5d, activeness, lastUsed, compound,
                 occupation, name, orgID, organisation, password, timeSpentWorking, 
                 username, isSepcialist, telegramID = "", discordID = ""):
        self.compound10d = compound10d
        self.compound5d = compound5d
        self.activeness = activeness
        self.lastUsed = lastUsed
        self.compound = compound
        self.isSepcialist = isSepcialist
        self.name = name
        self.occupation = occupation
        self.orgID = orgID
        self.organisation = organisation
        self.password = password
        self.telegramID = telegramID
        self.timeSpentWorking = timeSpentWorking
        self.username = username
        self.discordID = discordID
        
class Text(object):
    def __init__(self, date, score, text):
        self.date = date
        self.score = score
        self.text = text

    def to_dict(self):
        return({"date":self.date,"score":self.score,"text":self.text})

#Adding data to dataset
#db.collection('test').add({"name":"test"})

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['FLASK_APP'] = "flaskAPI.py"

'''
############################################################################
-------------------------- Flask API For AppGyver --------------------------
############################################################################
-------------------------- Remove PFD2 FROM PATH ---------------------------
############################################################################
'''
database = "PFD2/Data.csv"

#Functions
def sentiment_scores(text):
    # polarity_scores method of SentimentIntensityAnalyzer
    sentiment_dict = SentimentIntensityAnalyzer().polarity_scores(text)
    return(sentiment_dict['compound'])

def calculate_scores(ref):
    score0d = []
    score5d = []
    score10d = []
    scorelst = []
    datelst = []
    datedifflst = []
    docs = ref.collection(u'text').get()
    #Get date and score data from collection
    for doc in docs:
        datelst.append(doc.to_dict()['date'])
        scorelst.append(doc.to_dict()['score'])
    datelst.sort(reverse=True)
    #Calculation and assignments of data
    datedifflst = [(datelst[0] - d).days for d in datelst]
    print(datedifflst)
    print(datelst)
    for score,datediff in zip(scorelst,datedifflst):
        if datediff == 0:
            score10d.append(score)
            score5d.append(score)
            score0d.append(score)
        elif datediff <=5:
            score10d.append(score)
            score5d.append(score)
        elif datediff <= 10:
            score10d.append(score)
    #Update scores
    ref.update({"compoundScore":sum(score0d)/len(score0d),"score5d":sum(score5d)/len(score5d),"score10d":sum(score10d)/len(score10d)})

#-----API-----
@app.route("/")
def homepage():
    return jsonify(message=f"API is Working!", status = True)

@app.route("/discord/<string:discid>/<string:text>")
def disc_chat(discid:str,text:str):
    try:
        #Updating Compoundf
        text = text
        text = " ".join(text.split("."))
        score = sentiment_scores(text)
        
        #Reading data from dataset
        doc_ref = db.collection(u'discordUsers').document(discid).get()
        if doc_ref.exists:
            ref = doc_ref.to_dict()['id']
            #Adding data to dataset
            txt = Text(firestore.SERVER_TIMESTAMP,score,text).to_dict()
            ref.collection(u'text').add(txt)
            calculate_scores(ref)
            return jsonify(msg = 'User Info Updated!',status = True)
        else:
            return jsonify(msg = 'No such telegram ID!',status = False)
    except:
        return jsonify(msg = 'Error!',status = False)

@app.route("/telegram", methods = ["POST"])
def tele_chat():
    print(request.json)
    teleId = str(request.json['teleId'])
    text = request.json['text']
    print(teleId, text)
    try:
        #Updating Compoundf
        score = sentiment_scores(text)

        #Reading data from dataset
        doc_ref = db.collection(u'telegramUsers').document(teleId).get()
        if doc_ref.exists:
            ref = doc_ref.to_dict()['id']
            #Adding data to dataset
            txt = Text(firestore.SERVER_TIMESTAMP,score,text).to_dict()
            ref.collection(u'text').add(txt)
            calculate_scores(ref)
            return jsonify(msg = 'User Info Updated!',status = True)
        else:
            return jsonify(msg = 'No such telegram ID!',status = False)
    except:
        return jsonify(msg = 'Error!',status = False)

@app.route("/chat", methods = ["POST"])
def chat():
    req = request.get_json(silent=True, force=True)

    text = req['queryResult']["queryText"]
    id = req['session'].split("/")[-1]

    try:
        #Updating Compoundf
        score = sentiment_scores(text)
        #Reading data from dataset
        ref = db.collection(u'users').document(id)
        doc_ref = ref.get()
        if doc_ref.exists:
            #Adding data to dataset
            txt = Text(firestore.SERVER_TIMESTAMP,score,text).to_dict()
            ref.collection(u'text').add(txt)
            calculate_scores(ref)
            return jsonify(msg = 'User Info Updated!',status = True)
        else:
            return jsonify(msg = 'No such user ID!',status = False)
    except:
        return jsonify(msg = 'Error!',status = False)

@app.route("/tconnect/<string:severId>/<string:teleId>")
def tele_connect(severId:str,teleId:str):
    try:
        ref = db.collection(u'users').document(severId)
        if ref.get().exists:
            if not ref.get().to_dict()['telegramID']:
                #Creation of document to hold telegram id as primary key
                db.collection(u'telegramUsers').document(teleId).set({'id':ref})
                ref.update({u'telegramID': teleId})
                return jsonify(msg = 'User Info Updated!',status = True)
            else:
                return jsonify(msg = 'Already has telegram ID!',status = False)
        else:
            return jsonify(msg = 'No such server ID!',status = False)
    except:
        return jsonify(msg = 'Error!',status = False)

@app.route("/dconnect/<string:severId>/<string:discId>")
def disc_connect(severId:str,discId:str):
    try:
        ref = db.collection(u'users').document(severId)
        if ref.get().exists:
            if not ref.get().to_dict()['discordID']:
                #Creation of document to hold telegram id as primary key
                db.collection(u'discordUsers').document(discId).set({'id':ref})
                ref.update({u'discordID': discId})
                return jsonify(msg = 'User Info Updated!',status = True)
            else:
                return jsonify(msg = 'Already has discord ID!',status = False)
        else:
            return jsonify(msg = 'No such server ID!',status = False)
    except:
        return jsonify(msg = 'Error!',status = False)

@app.route("/tsearch/<string:teleId>")
def tele_search(teleId:str):
    docs = db.collection(u'users').where(u'telegramID', u'==', teleId).get()
    if len(docs) > 0:
        return jsonify(msg = 'Found!',status = True)
    else:
        return jsonify(msg = 'Not Found!',status = False)

@app.route("/dsearch/<string:discId>")
def disc_search(discId:str):
    docs = db.collection(u'users').where(u'discordID', u'==', discId).get()
    if len(docs) > 0:
        return jsonify(msg = 'Found!',status = True)
    else:
        return jsonify(msg = 'Not Found!',status = False)

@app.route("/tdisconnect/<string:teleId>")
def tele_disconnect(teleId:str):
    docs = db.collection(u'users').where(u'telegramID', u'==', teleId).get()
    db.collection(u'telegramUsers').document(teleId).delete()
    if len(docs) > 0:
        doc=docs[0]
        ref = db.collection(u'users').document(doc.id)
        ref.update({u'telegramID': ""})
        return jsonify(msg = 'User Info Updated!',status = True)
    else:
        return jsonify(msg = 'Not Found!',status = False)

@app.route("/ddisconnect/<string:discId>")
def disc_disconnect(discId:str):
    docs = db.collection(u'users').where(u'discordID', u'==', discId).get()
    db.collection(u'discordUsers').document(discId).delete()
    if len(docs) > 0:
        doc=docs[0]
        ref = db.collection(u'users').document(doc.id)
        ref.update({u'discordID': ""})
        return jsonify(msg = 'User Info Updated!',status = True)
    else:
        return jsonify(msg = 'Not Found!',status = False)

@app.route("/tname/<string:teleId>")
def tele_name(teleId:str):
    docs = db.collection(u'users').where(u'telegramID', u'==', teleId).get()
    if len(docs) > 0:
        doc=docs[0]
        name = db.collection(u'users').document(doc.id).get().to_dict()["name"]
        return jsonify(msg = name,status = True)
    else:
        return jsonify(msg = 'Name Not Found!',status = False)

#signup()

if __name__ == '__main__':
    app.run()