from django.shortcuts import render
from django.http import HttpResponse
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re, collections
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score
import nltk.data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import numpy as np
from nlp.forms import sem
from nlp.forms import sen
from .models import Contact
'''It seems to me we are in the middle of no man's land with respect to the  following:  Opec production speculation, Mid east crisis and renewed  tensions, US elections and what looks like a slowing economy (?), and no real weather anywhere in the world. I think it would be most prudent to play  the markets from a very flat price position and try to day trade more aggressively. I have no intentions of outguessing Mr. Greenspan, the US. electorate, the Opec ministers and their new important roles, The Israeli and Palestinian leaders, and somewhat importantly, Mother Nature.  Given that, and that we cannot afford to lose any more money, and that Var seems to be a problem, let's be as flat as possible. I'm ok with spread risk  (not front to backs, but commodity spreads). The morning meetings are not inspiring, and I don't have a real feel for  everyone's passion with respect to the markets.  As such, I'd like to ask  John N. to run the morning meetings on Mon. and Wed.  Thanks. Jeff'''
def post(self,request):
    fro = sem(request.POST)
    if fro.is_valid():
        message = fro.cleaned_data['mess']
    return mess 
# Create your views here.
def index(request):
    templates= "index.html"
    mess=''
    all_entries=[]
    if request.method == 'POST':
        if request.POST.get('name') and request.POST.get('email') and request.POST.get('message'):
            post=Contact()
            post.name= request.POST.get('name')
            post.email= request.POST.get('email')
            post.message= request.POST.get('message')
            post.save()
            mess='ThankYou For Contatcting Us ... We will Contact you Shortly...'
    return render(request,templates,{'mess':mess});
def modules(request):
    templates= "modules.html"
    return render(request,templates);
def essay(request):
    templates= "essay.html"
    g1=0
    g2=0
    m1=0
    m2=0
    c=0
    if request.method == 'POST':
        essay=request.POST.get('essay')
        dataframe = pd.read_csv(essay, encoding = 'latin-1')
        data = dataframe[['essay_set','essay','domain1_score']].copy()
        def sentence_to_wordlist(raw_sentence):
            clean_sentence = re.sub("[^a-zA-Z0-9]"," ", raw_sentence)
            tokens = nltk.word_tokenize(clean_sentence)
            return tokens
        def tokenize(essay):
            stripped_essay = essay.strip()
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            raw_sentences = tokenizer.tokenize(stripped_essay)
            tokenized_sentences = []
            for raw_sentence in raw_sentences:
                if len(raw_sentence) > 0:
                    tokenized_sentences.append(sentence_to_wordlist(raw_sentence))
            return tokenized_sentences
        def avg_word_len(essay):
            clean_essay = re.sub(r'\W', ' ', essay)
            words = nltk.word_tokenize(clean_essay)
            return sum(len(word) for word in words) / len(words)
        def word_count(essay):
            clean_essay = re.sub(r'\W', ' ', essay)
            words = nltk.word_tokenize(clean_essay)
            return len(words)
        def sent_count(essay):
            sentences = nltk.sent_tokenize(essay)
            return len(sentences)
        # calculating number of lemmas per essay
        def count_lemmas(essay):
            tokenized_sentences = tokenize(essay)      
            lemmas = []
            wordnet_lemmatizer = WordNetLemmatizer()
            for sentence in tokenized_sentences:
                tagged_tokens = nltk.pos_tag(sentence) 
                for token_tuple in tagged_tokens:
                    pos_tag = token_tuple[1]
                    if pos_tag.startswith('N'): 
                        pos = wordnet.NOUN
                        lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
                    elif pos_tag.startswith('J'):
                        pos = wordnet.ADJ
                        lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
                    elif pos_tag.startswith('V'):
                        pos = wordnet.VERB
                        lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
                    elif pos_tag.startswith('R'):
                        pos = wordnet.ADV
                        lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
                    else:
                        pos = wordnet.NOUN
                        lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
    
            lemma_count = len(set(lemmas))
            return lemma_count
        def count_spell_error(essay):
            clean_essay = re.sub(r'\W', ' ', str(essay).lower())
            clean_essay = re.sub(r'[0-9]', '', clean_essay)
            #big.txt: It is a concatenation of public domain boo
            #and lists of most frequent words from Wiktionary and the British National Corpus.
            #It contains about a million words.
            data = open('big.txt').read()
            words_ = re.findall('[a-z]+', data.lower())
            word_dict = collections.defaultdict(lambda: 0)
            for word in words_:
                word_dict[word] += 1
            clean_essay = re.sub(r'\W', ' ', str(essay).lower())
            clean_essay = re.sub(r'[0-9]', '', clean_essay)
            mispell_count = 0
            words = clean_essay.split()
            for word in words:
                if not word in word_dict:
                     mispell_count += 1
            return mispell_count
        def count_pos(essay):
            tokenized_sentences = tokenize(essay)
            noun_count = 0
            adj_count = 0
            verb_count = 0
            adv_count = 0
            for sentence in tokenized_sentences:
                tagged_tokens = nltk.pos_tag(sentence)
                for token_tuple in tagged_tokens:
                    pos_tag = token_tuple[1]
                    if pos_tag.startswith('N'): 
                        noun_count += 1
                    elif pos_tag.startswith('J'):
                        adj_count += 1
                    elif pos_tag.startswith('V'):
                        verb_count += 1
                    elif pos_tag.startswith('R'):
                        adv_count += 1
            return noun_count, adj_count, verb_count, adv_count
        def get_count_vectors(essays):
            vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')
            count_vectors = vectorizer.fit_transform(essays)
            feature_names = vectorizer.get_feature_names()
            return feature_names, count_vectors
        feature_names_cv, count_vectors = get_count_vectors(data[data['essay_set'] == 1]['essay'])
        X_cv = count_vectors.toarray()
        y_cv = data[data['essay_set'] == 1]['domain1_score'].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X_cv, y_cv, test_size = 0.3)
        alphas = np.array([3, 1, 0.3, 0.1, 0.03, 0.01])
        lasso_regressor = Lasso()
        grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        # summarize the results of the grid search
        g1=grid.best_score_
        g2=grid.best_estimator_.alpha
        # The mean squared error
        m1=mean_squared_error(y_test, y_pred)
        # Explained variance score: 1 is perfect prediction
        m2=grid.score(X_test, y_test)
        # Cohen’s kappa score: 1 is complete agreement
        c=cohen_kappa_score(np.rint(y_pred), y_test)
    return render(request,templates,{'g1':g1*100,'g2':g2,'m1':m1,'m2':m2,'c':c});
def semantic(request):
    message=''''''
    s=[]
    sentences=[]
    templates= "semantic.html"
    if request.method == 'POST':
        message = request.POST.get('message')
        sid = SentimentIntensityAnalyzer()
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        message_text = '''It seems to me we are in the middle of no man's land with respect to the  following:  Opec production speculation, Mid east crisis and renewed  tensions, US elections and what looks like a slowing economy (?), and no real weather anywhere in the world. I think it would be most prudent to play  the markets from a very flat price position and try to day trade more aggressively. I have no intentions of outguessing Mr. Greenspan, the US. electorate, the Opec ministers and their new important roles, The Israeli and Palestinian leaders, and somewhat importantly, Mother Nature.  Given that, and that we cannot afford to lose any more money, and that Var seems to be a problem, let's be as flat as possible. I'm ok with spread risk  (not front to backs, but commodity spreads). The morning meetings are not inspiring, and I don't have a real feel for  everyone's passion with respect to the markets.  As such, I'd like to ask  John N. to run the morning meetings on Mon. and Wed.  Thanks. Jeff''';
        sentences = tokenizer.tokenize(message)
        for sentence in sentences:
            s.append(sentence)
            scores = sid.polarity_scores(sentence)
            for key in sorted(scores):
                s.append(key)
                s.append(str(int(scores[key]*100))+"%")
    return render(request,templates,{'scores':s});
def sentence(request):
    sentence1=""
    sentence2=""
    dot=0
    final_similarity=0
    templates= "sentence.html"
    if request.method == 'POST':
     sentence1 = str(request.POST.get('sen1', None))
     sentence2 = str(request.POST.get('sen2', None))
     sentenc="A jewel is a precious stone used to decorate valuable things that you wear, such as rings or necklaces."
     sentenc="A gem is a jewel or stone that is used in jewellery."
     class SentenceSimilarity():
    
      def __init__(self):
        self.word_order = False

      def identifyWordsForComparison(self,sentence):
        #Taking out Noun and Verb for comparison word based
        tokens = nltk.word_tokenize(sentence)        
        pos = nltk.pos_tag(tokens)
        pos1 = [p for p in pos if p[1].startswith('N') or p[1].startswith('V')]     
        return pos1

      def wordSenseDisambiguation(self,sentence):
       # removing the disambiguity by getting the context
       pos = self.identifyWordsForComparison(sentence)
       sense = []
       for p in pos:
           sense.append(lesk(sentence, p[0], pos=p[1][0].lower()))
       return set(sense)

      def getSimilarity(self,arr1, arr2, vector_len):
        #cross multilping all domains 
        vector = [0.0] * vector_len
        count = 0
        for i,a1 in enumerate(arr1):
            all_similarityIndex=[]
            for a2 in arr2:
                similarity = a1.wup_similarity(a2)
                if similarity != None:
                    all_similarityIndex.append(similarity)
                else:
                    all_similarityIndex.append(0.0)
            all_similarityIndex = sorted(all_similarityIndex, reverse = True)
            vector[i]=all_similarityIndex[0]
            if vector[i] >= 0.804:
                count +=1
        return vector, count

      def shortestPathDistance(self,sense1, sense2):
        #getting the shortest path to get the similarity
        if len(sense1) >= len(sense2):
            grt_Sense = len(sense1)
            v1, c1 = self.getSimilarity(sense1, sense2, grt_Sense)
            v2, c2 = self.getSimilarity(sense2, sense1, grt_Sense)
        if len(sense2) > len(sense1):
            grt_Sense = len(sense2)
            v1, c1 = self.getSimilarity(sense2, sense1, grt_Sense)
            v2, c2 = self.getSimilarity(sense1, sense2, grt_Sense)
        return np.array(v1),np.array(v2),c1,c2
    
     obj = SentenceSimilarity()
     try: 
       sense1 = obj.wordSenseDisambiguation(sentence1)
       sense2 = obj.wordSenseDisambiguation(sentence2)        
       v1,v2,c1,c2 = obj.shortestPathDistance(sense1,sense2)
       dot = np.dot(v1,v2)
       tow = (c1+c2)/1.8
       final_similarity = dot/tow
     except AttributeError: 
       print("There is no such attribute")
    return render(request,templates,{'dot':dot,'final_similarity':final_similarity*100});