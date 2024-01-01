import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bz2 # To open zipped files
import re # regular expressions
import os
import gc

import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
import nltk
from nltk import pos_tag
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

wnl = WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 
    
def lemmatize_sent(text): 
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(word_tokenize(text))]


def important_features(vectorizer,classifier,n=40):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names_out()

    topn_class1 = sorted(zip(classifier.feature_count_[0], feature_names),reverse=True)[:n]
    topn_class2 = sorted(zip(classifier.feature_count_[1], feature_names),reverse=True)[:n]

    class1_frequency_dict = {}
    class2_frequency_dict = {}
    
    for coef, feat in topn_class1:
        class1_frequency_dict.update( {feat : coef} )

    for coef, feat in topn_class2:
        class2_frequency_dict.update( {feat : coef} )

    return (class1_frequency_dict, class2_frequency_dict)

def process_uploaded_file(file_path):
    if file_path.endswith('.bz2'):
        with bz2.BZ2File(file_path, 'r') as file:
            # Process the contents of the file, for example, using pandas
            train_file = pd.read_csv(file, delimiter='\t', header=None, names=['Label', 'Text'])
            # Add your processing logic here
            print(train_file.head())
            train_file_lines['Text'] = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in train_file['Text']]
            del train_file
            gc.collect()
            print(type(train_file_lines), "\n")
            print("Train Data Volume:", len(train_file_lines),"\n")
            print("Demo: ", "\n")
            for x in train_file_lines[:5]:
                print(x, "\n")
            train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]
            train_labels[0]
            sns.countplot(train_labels)
            plt.title('Train Labels distribution')
            train_sentences = [x.split(' ', 1)[1][:-1] for x in train_file_lines]
            train_sentences[0]
            train_sentences_size = list(map(lambda x: len(x.split()), train_sentences))

            sns.histplot(train_sentences_size, kde=True)
            plt.xlabel("#words in reviews")
            plt.ylabel("Frequency")
            plt.title("Word Frequency Distribution in Reviews")

            train_label_len = pd.DataFrame({"labels": train_labels, "len": train_sentences_size})
            train_label_len.head()

            neg_mean_len = train_label_len.groupby('labels')['len'].mean().values[0]
            pos_mean_len = train_label_len.groupby('labels')['len'].mean().values[1]

            print(f"Negative mean length: {neg_mean_len:.2f}")
            print(f"Positive mean length: {pos_mean_len:.2f}")
            print(f"Mean Difference: {neg_mean_len-pos_mean_len:.2f}")

            sns.catplot(x='labels', y='len', data=train_label_len, kind='box')
            plt.xlabel("labels (0->negative, 1->positive)")
            plt.ylabel("#words in reviews")
            plt.title("Review Size Categorization")

            for i in range(len(train_sentences)):
                if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
                    train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

            list(filter(lambda x: '<url>' in x, train_sentences))[0]

            del train_file_lines
            gc.collect()


            nltk.data.path.append(file_path)
            nltk.download('averaged_perceptron_tagger')
            nltk.download("punkt")
            nltk.download('wordnet')
            nltk.download('stopwords')

            wnl = WordNetLemmatizer()

            lemmatize_sent('He is WALKING walking to school')

            # Stopwords from stopwords-json
            stopwords_json = {"en":["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost",
                        "alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways",
                        "anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be",
                        "became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between",
                        "beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com",
                        "come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d",
                        "definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight",
                        "either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example",
                        "except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get",
                        "gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having",
                        "he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how",
                        "howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar",
                        "instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later",
                        "latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean",
                        "meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither",
                        "never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often",
                        "oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own",
                        "p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re",
                        "really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see",
                        "seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since",
                        "six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still",
                        "sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves",
                        "then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third",
                        "this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try",
                        "trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v",
                        "value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what",
                        "what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither",
                        "who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you",
                        "you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]}
            
            stopwords_json_en = set(stopwords_json['en'])
            stopwords_nltk_en = set(stopwords.words('english'))
            stopwords_punct = set(punctuation)
            # Combine the stopwords. Its a lot longer so I'm not printing it out...
            stoplist_combined = set.union(stopwords_json_en, stopwords_nltk_en, stopwords_punct)
            def preprocess_text(text):
                # Input: str, i.e. document/sentence
                # Output: list(str) , i.e. list of lemmas
                return [word for word in lemmatize_sent(text) 
                        if word not in stoplist_combined
                        and not word.isdigit()]
            
            train_sentences[10]

            preprocess_text(train_sentences[10])

            count_vect = CountVectorizer(analyzer=preprocess_text)

            train_set = count_vect.fit_transform(train_sentences[:10000])

            train_set.toarray().shape

            most_freq_words = pd.DataFrame(count_vect.vocabulary_.items(), columns=['word', 'frequency'])[:100].sort_values(ascending=False, by = "frequency")[:20]
            most_freq_words.plot.bar(x="word", y="frequency", rot=70, title="Most Frequent Words")

            clf = MultinomialNB()
            clf.fit(train_set, train_labels[:10000])
            
            neg_frequency_dict, pos_frequency_dict = important_features(count_vect, clf)

            neg_feature_freq = pd.DataFrame(neg_frequency_dict.items(), columns = ["feature_word", "frequency"])  
            pos_feature_freq = pd.DataFrame(pos_frequency_dict.items(), columns = ["feature_word", "frequency"]) 

            neg_feature_freq.plot.bar(x="feature_word", y="frequency", rot=70, figsize=(15, 5), title="Important Negative Features(words)")
            pos_feature_freq.plot.bar(x="feature_word", y="frequency", rot=70, figsize=(15, 5), title="Important Positive Features(words)")
    else:
        # If not compressed, assume it's a regular CSV file
        df = pd.read_csv(file_path)
        # Add your processing logic here
        print(df.head())

import csv
import os

from collections import defaultdict
from datetime import datetime
from decimal import Decimal

def process_csv(filename):
    product_types = defaultdict(Decimal)

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                hypen_idx = row['product'].replace('"', '').split().index('-')
                product_type =  ' '.join(row['product'].split()[:hypen_idx])
                product_types[product_type] += Decimal(row['price'][1:])
            except ValueError:
                product_types[row['product'].replace('"', '')] += Decimal(row['price'][1:])

    output_file = f'product_types_{str(datetime.now())}.csv'
    with open(os.path.join('output', output_file), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['product_type', 'price'])

        for product_type, price in product_types.items():
            writer.writerow([product_type, price])

    return output_file