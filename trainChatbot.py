#import tensorflow

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import nltk
from nltk.stem import WordNetLemmatizer  #Lemmatizer

lemmatizer=WordNetLemmatizer()

import json
import pickle #To Serialize/Deserialize
import numpy as np
from keras.models import Sequential  #Sequential Model Used
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

import random

words=[]
classes=[]
documents=[]
ignoreWords=['!','?']
fileName=open('intents.json','r')
print('This is working fine till now')
#dataFile=fileName.read()
try:
    intents=json.loads(fileName)
except ValueError as e:
    print(e)

print("Hello this is working fine")

#print(intents)
#print(type())
'''
import json

#dataFile=open('intents.json','r')
with open("intents.json","r") as readFile:
    data=json.load(readFile)
print(type(data))
'''
'''
intent={
  "intents": [
        {"tag": "greeting",
         "patterns": ["Hi there", "How are you", "Is anyone there?","Hey","Hola", "Hello", "Good day"],
         "responses": ["Hello, thanks for asking", "Good to see you again", "Hi there, how can I help?"],
         "context": [""]
        },
        {"tag": "goodbye",
         "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
         "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."],
         "context": [""]
        },
        {"tag": "thanks",
         "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
         "responses": ["Happy to help!", "Any time!", "My pleasure"],
         "context": [""]
        },
        {"tag": "noanswer",
         "patterns": [],
         "responses": ["Sorry, can't understand you", "Please give me more info", "Not sure I understand"],
         "context": [""]
        },
        {"tag": "options",
         "patterns": ["How you could help me?", "What you can do?", "What help you provide?", "How you can be helpful?", "What support is offered"],
         "responses": ["I can guide you through Adverse drug reaction list, Blood pressure tracking, Hospitals and Pharmacies", "Offering support for Adverse drug reaction, Blood pressure, Hospitals and Pharmacies"],
         "context": [""]
        },
        {"tag": "adverse_drug",
         "patterns": ["How to check Adverse drug reaction?", "Open adverse drugs module", "Give me a list of drugs causing adverse behavior", "List all drugs suitable for patient with adverse reaction", "Which drugs dont have adverse reaction?" ],
         "responses": ["Navigating to Adverse drug reaction module"],
         "context": [""]
        },
        {"tag": "blood_pressure",
         "patterns": ["Open blood pressure module", "Task related to blood pressure", "Blood pressure data entry", "I want to log blood pressure results", "Blood pressure data management" ],
         "responses": ["Navigating to Blood Pressure module"],
         "context": [""]
        },
        {"tag": "blood_pressure_search",
         "patterns": ["I want to search for blood pressure result history", "Blood pressure for patient", "Load patient blood pressure result", "Show blood pressure results for patient", "Find blood pressure results by ID" ],
         "responses": ["Please provide Patient ID", "Patient ID?"],
         "context": ["search_blood_pressure_by_patient_id"]
        },
        {"tag": "search_blood_pressure_by_patient_id",
         "patterns": [],
         "responses": ["Loading Blood pressure result for Patient"],
         "context": [""]
        },
        {"tag": "pharmacy_search",
         "patterns": ["Find me a pharmacy", "Find pharmacy", "List of pharmacies nearby", "Locate pharmacy", "Search pharmacy" ],
         "responses": ["Please provide pharmacy name"],
         "context": ["search_pharmacy_by_name"]
        },
        {"tag": "search_pharmacy_by_name",
         "patterns": [],
         "responses": ["Loading pharmacy details"],
         "context": [""]
        },
        {"tag": "hospital_search",
         "patterns": ["Lookup for hospital", "Searching for hospital to transfer patient", "I want to search hospital data", "Hospital lookup for patient", "Looking up hospital details" ],
         "responses": ["Please provide hospital name or location"],
         "context": ["search_hospital_by_params"]
        },
        {"tag": "search_hospital_by_params",
         "patterns": [],
         "responses": ["Please provide hospital type"],
         "context": ["search_hospital_by_type"]
        },
        {"tag": "search_hospital_by_type",
         "patterns": [],
         "responses": ["Loading hospital details"],
         "context": [""]
        }
   ]
}
'''
#print(type(intent))
