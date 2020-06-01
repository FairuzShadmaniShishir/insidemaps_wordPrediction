'''
Author:Fairuz Shadmani Shishir
16.05.2020
insidemaps.com
'''

#import libraries

#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
#from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
from flask_cors import CORS

#import the excel file
#file_location=r'E:\InsideMaps Office\KnowledgeBase.xlsx'

df = pd.read_excel('KnowledgeBase.xlsx')
df['Answers']=df.Answers.fillna('This Answer is not found')


# TFIDF vectorizer
tfidf = TfidfVectorizer()
tfidf_features = tfidf.fit_transform(df.Query)
tfidf_features.get_shape()

#re for reular expressions
import re

def process_query(query):
    preprocessed_reviews = []
    sentance = re.sub("\S*\d\S*", "", query).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    #sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords.words('english'))
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower())
    preprocessed_reviews.append(sentance.strip())
    return preprocessed_reviews
    
def tfidf_search(tfidf, query):
    query = process_query(query)
    query_trans = tfidf.transform(query)
    pairwise_dist = pairwise_distances(tfidf_features, query_trans)
    
    indices = np.argsort(pairwise_dist.flatten())[0:5]
    df_indices = list(df.index[indices])
    return df_indices
    
def search(query, typ = "tfidf"):
    query_list=[]
    answer_list=[]
    if typ == "tfidf":
        val = tfidf_search(tfidf, query)
    
        
    for i in (val):   
        query_list.append(df.Query.iloc[i]) 
        answer_list.append(df.Answers.iloc[i])
    return query_list,answer_list


from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers import Embedding
from tensorflow.keras.models import load_model


# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    # generate a fixed number of words
    for _ in range (n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences ([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded = pad_sequences ([encoded], maxlen=max_length, padding='pre')
        # predict probabilities for each word
        yhat = model.predict_classes (encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items ():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    return in_text


# source text
data = """ Which devices are compatible with InsideMaps HERO 360 Rotor?
How do I establish a bluetooth connection between my iPhone and the HERO device?
How do I know if all my photos have uploaded successfully?
How much storage space do I need on my phone to capture a property?
What happens to the photos on my iPhone after they have been uploaded?
Will my projects be hosted by InsideMaps?
Can my projects be embedded on my own website?
What is the difference between a 3D tour and a 3D model?
How long will it take to capture a property?
My project was submitted over 24 hours ago, but it still has not finished, why?
What tripod should I use with InsideMaps?
How much free space should I have on my iPhone?
How is InsideMaps different than Matterport?
How can I change the starting point of my 3D Tour?
How do I pay for my projects?
Can I do a test project without paying?
Project Management and Customization
Scanning Your First Property
Equipment Setup
Creating Your First Project
Blocking Issue #1 | Missing Room/Hallways | Not enough Rooms to complete the project
Battery is loose or not making a good connection in the HERO
Panos are bad because the HERO is not rotating correctly / HERO issue
What do you charge for just the 3D Tour?
How to change the logo/remove branding from a project
Project Customization: This tutorial shows how to manage and customize all of your projects, including editing floor plans and 3D Tours.
Scanning Your First Property: Step by step instructions on how to properly capture your first property.
Equipment Setup Guide: Learn how to use the InsideMaps Capture equipment, including syncing your device with the HERO.
Creating Your First Project: Detailed instructions on how to use the InsideMaps Capture app to create a new project.
Why is it taking so long for my project to upload?
Does InsideMaps Charge For Hosting?
The 3D Tour, 3D Model, or Floor Plan are missing from the listing page
We are working on your ticket
Ticket solved - Customer is happy
What is the turnaround time for Standard / Pro / Floorplan / HDR package?
Customer asks for ETA or when their project will be ready
Photo Capture Tutorial
Photo Capture Guide
Customer asks if Android is supported
The map location at the bottom of the listing page for my project is not correct?
How to download a project's property report?
How to send "httpTasks" folder
Customer is asking why a project has not been finished, on the Photo Analyzer page for the project it says "Project is not locked"
What is pilot project?
How much free storage space do you need on your iPhone to use InsideMaps?
How to fix Cube faces with Photoshop
Customer Asks "Can I see more examples of your 3D Tours or 3D Models?"
How do I share my project with others / How do I embed a project on my website?
The client is asking about a project, but you cannot locate it.
Client is asking Sales related questions, I have no info
The client is reporting that rooms are missing from their project
The client asks why there are no photos in the HDR Photos section of their project's listing page
Can two people work on the same project simultaneously ?
How to clone the rooms / merge rooms from one project to another
How to clone a project
OD scanners list - Live InsideMaps Issue Tracker
Customer reports that their project or tour won't load, but it works for you.
How to correct the Address for "Recovered" projects.
How to make a project public
Ask User to Update the App
How to Delete a Client's Project When Requested
Contact Us Tickets - Customer is Asking for General Information About InsideMaps
[NEEDS REVIEW -do not use] Contact Us Tickets - Customer is Asking for General Information About InsideMaps
Ticket Solved - Customer Rated Service as "Good"
Ticket Solved - Customer is not Happy
Ticket Solved - Customer is Happy 2
Ask the client for information about their computer / mobile device and operating system
Ticket Solved - Customer is not happy 2
Client has reported a bug - We are working on your ticket.
How to accept invitation to become member of Organization
Customer is complaining about server issues, errors, or slow loading
How to download and send 8K Panos to a client
Difference between Splash & Set Start point option.
3D Video ( Walkthrough ) Price Model and Basic Information
Important Project Management User Guide
How to import fixed SQlite3 file back to the iPhone.
Room created from scratch - message template
How to Extract Raw Photos from a client's iPhone
Reffered User Signup | exitrealty - Solution
User Education for 3D Models - Is it possible to get 3D Models in different formats for offline viewing / What formats can 3D Models be exported to?
Client Inquiry: Project missing from the app
User Guide - Where to Place the HERO to get the best HDR Photos
How to handle projects with issue from VIP customers
About Matterport Vs. InsideMaps
[NEEDS REVIEW] Contact Us version #3
Customer asking for reference customers from specific region
Storage Failure | Your Phone is Out of Free Space | Out of Memory Message | Memory is NOT Full
Posting Facebook 360 Spin
Photos for Offerpad
Client says the upload is suspended.
Invites expire
Large client contacts us
Customer says the iPhone is not charging
Customer asks if she can upload her own photos for a specific project
Reprocess trifocal, to remove stitching errors
iCloud Storage - Out of Storage
Scan Request QA and Delivery
Agent emails us about $50 scans , or scheduling scans
Adding reference measurements
Blocked Projects Workaround
Client reports Storage failure issue
Merging Request
Mac user getting error "unable to expand into downloads" "operation not permitted" when unzipping file
If a user has not accepted an invite to an org, please use this template
Project won't upload - Upload stuck
How to disable spins in the 3D tour
Customer wants to have two accounts attached to one email
Info regarding floorplans
HERO CONNECTIVITY FLOW (and Connection troubleshooting guide)
RezFest Winners HomeAway Users (when they can't see 3D tour on VRBO or HomeAway)
Pricing for RezFest Winners
How to reset password
How to create Organization and invite Users and Admins
Move asset photos to HDR photos
No data transferred - message template
Clean install the app
User Guide LINKS
Insidemaps Customer Support Hotline Number - When Clients ask , "Do you have a phone number where I can call ? "
Packages that InsideMaps offer
HERO Guarantee & Refund Policy
"Rotation Failure" error - Troubleshoot guide
Adding Logo to every project in organization
Dropbox login information
iPhone sensor calibration
Website is down
Enabling app sound for InsideMaps Cappture app
Firmware HERO
Scanning equipment
Homeaway/Vrbo - Rent a HERO - My project supposed to be free (Paid by HomeAway) but it asks me to pay for it.
Scanning instructions: Bungalo
Scanning instructions: Tricon
What is a Curb Shot
How to Enable Mobile Data Photo Upload
How to bookmark a spin
Introducing Theta
Asset Capture Guide
Theta V Losing Connection - Troubleshooting
Rotation Failure Error Message
Rotation Failure Troubleshooting for CSS
Where can I access training materials?
HomeAway/Vrbo Paid Offer 2019.
VR support
Uploading in the background | Best practices to ensure timely uploads
Enabling upload over Mobile Data
Adjusting Auto-Lock (screen timeout) Setting
Blocking Issue #2 | Need video capture
Blocking Issue | Can't open 3D Floor Editor
Blocking Issue | Door height issue
Blocking Issue | We can't find the Main Project for Merge
Blocking Issue | Project cannot be finished
Blocking Issue | 3D Tour is not loading
Blocking Issue | All doors in the project are closed
Ceiling Fan/Flag is Creating Blur | Ceiling fan or a Flag causes blur in photos
Blocking Issue | Can't open or save Floor Editor
No Line of Sight between spins
Non Blocking Issue | Dirty lens
Non Blocking Issue | People in scans
Non Blocking Issue | Can't see some asset tools & information
Non Blocking Issue | Spins in the door frames
Non Blocking Issue | Room created from scratch (Missing Room)
Non Blocking Issue | JM Corp - Separate floor for a room without connection
Tricon | Marketing Angles Not Captured
Tricon | No exterior spins
Non Blocking Issue | The spin/spins appear to be tilted
Blocking Issue | Developer's issue
Tricon | Not all rooms were captured
Tricon | No good curb shot
Tricon | No connection between rooms
Tricon | Project is not ready for Marketing (not empty and clean)
Tricon | People in scan
Tricon | Dirty Lens
Tricon | Bad spins position for photo capture
Blocking Issue | No outside spins
Blocking Issue | Not all rooms were captured
Blocking Issue | People in scans
Blocking Issue | Ceiling fan or a Flag causes blur in photos
Blocking Issue | No good curb shot
Blocking Issue | Assets were not submitted by the scanner
Non Blocking Issue | Curb Shot was not submitted
FirstKey | Can't see some asset tools & information."""

# prepare the tokenizer on the source text
#print (type (data))
tokenizer = Tokenizer ()
tokenizer.fit_on_texts ([data])
# determine the vocabulary size
vocab_size = len (tokenizer.word_index) + 1
#print ('Vocabulary Size: %d' % vocab_size)
# create line-based sequences
sequences = list ()
for line in data.split ('\n'):
    encoded = tokenizer.texts_to_sequences ([line])[0]
    for i in range (1, len (encoded)):
        sequence = encoded[:i + 1]
        sequences.append (sequence)
#print ('Total Sequences: %d' % len (sequences))
# pad input sequences
max_length = max ([len (seq) for seq in sequences])
sequences = pad_sequences (sequences, maxlen=max_length, padding='pre')
#print ('Max Sequence Length: %d' % max_length)
# split into input and output elements
sequences = array (sequences)
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical (y, num_classes=vocab_size)
# define model
'''
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=200, verbose=2)
# evaluate model
'''

model = load_model ('model.h5')
#print (tokenizer)
#print (max_length - 1)






from flask import Flask, request, redirect, url_for, flash, jsonify
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def makecalc():
    data = request.get_json()
    Queries,Answers= search("insidemaps")
    return jsonify({'query':Queries,'answers':Answers})

@app.route('/getAll', methods=['POST'])
def get_all_data():
    data = request.get_json()
    query = data['key']
    Queries, Answers = search(query)
    return jsonify({'query': Queries, 'answers': Answers})

@app.route('/getNextWord',methods=['POST'])
def getWordPrediction():
    data = request.get_json ()
    query = data['key']
    word=generate_seq(model, tokenizer, max_length-1, query,1)
    return jsonify ({'prediction': word})


if __name__ == '__main__':
    app.run(debug=True)