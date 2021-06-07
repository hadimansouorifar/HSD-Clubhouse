from googleapiclient import discovery
import csv
import pandas as pd
import json
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
import time
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    #tokenizer = RegexpTokenizer(r'\w+')
    #tokens = tokenizer.tokenize(rem_num)  
    #filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    #stem_words=[stemmer.stem(w) for w in filtered_words]
    #lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)

##### You need to replace your own API code
API_KEY = 'your API code'

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)


with open('Clubhouse.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    c1=0
    c2=0
    m=[]
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    s5=[]
    s6=[]
    s7=[]
    s8=[]
    s9=[]
    label=[]
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            try:
                analyze_request = {'comment': { 'text': row[0] },'requestedAttributes': {'TOXICITY': {}, 'SEVERE_TOXICITY': {}, 'IDENTITY_ATTACK': {}, 'INSULT': {}, 'PROFANITY': {}, 'THREAT': {}, 'SEXUALLY_EXPLICIT': {}, 'OBSCENE': {}, 'SPAM': {}    }}
                response = client.comments().analyze(body=analyze_request).execute()
                s1.append(json.dumps(response["attributeScores"]["TOXICITY"]["summaryScore"]["value"], indent=2))
                s2.append(json.dumps(response["attributeScores"]["SEVERE_TOXICITY"]["summaryScore"]["value"], indent=2))
                s3.append(json.dumps(response["attributeScores"]["IDENTITY_ATTACK"]["summaryScore"]["value"], indent=2))
                s4.append(json.dumps(response["attributeScores"]["INSULT"]["summaryScore"]["value"], indent=2))
                s5.append(json.dumps(response["attributeScores"]["PROFANITY"]["summaryScore"]["value"], indent=2))
                s6.append(json.dumps(response["attributeScores"]["THREAT"]["summaryScore"]["value"], indent=2))
                s7.append(json.dumps(response["attributeScores"]["SEXUALLY_EXPLICIT"]["summaryScore"]["value"], indent=2))
                s8.append(json.dumps(response["attributeScores"]["OBSCENE"]["summaryScore"]["value"], indent=2))
                s9.append(json.dumps(response["attributeScores"]["SPAM"]["summaryScore"]["value"], indent=2))
                time.sleep(10)









                line_count += 1
                print(f'Processed {line_count} lines.')
                m.append(row[0])
                label.append(row[1])

                
                
                if(row[1]=="0"):
                   c1=c1+1
                if(row[1]=="1"):
                   c2=c2+1

            except Exception as e:
                                  k2=0

           

       

        #print(f'\t{(row[0])}.')
        
data = {'text': m,
        'label': label,'TOXICITY':s1,'SEVERE_TOXICITY':s2,'IDENTITY_ATTACK':s3,'INSULT':s4,'PROFANITY':s5,'THREAT':s6,'SEXUALLY_EXPLICIT':s7,'OBSCENE':s8,'SPAM':s9}

df = pd.DataFrame(data, columns= ['text', 'label','TOXICITY','SEVERE_TOXICITY','IDENTITY_ATTACK','INSULT','PROFANITY','THREAT','SEXUALLY_EXPLICIT','OBSCENE','SPAM'])
print(df)
df.to_csv (r'clubhouse-scores3.csv', index = False, header=True)
print(c1)
print(c2)