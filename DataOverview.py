import string
import csv
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import FeatureExtraction as fe


def plot_by_class(data, col_name, output_path):

    # data[col_name].value_counts().plot(kind='barh', color='lightblue')
    # data[col_name].value_counts().plot.pie(colors = ['pink', 'lightblue'])
    count = dict(sorted(dict(data[col_name].value_counts()).items()))
    y = []
    for key in count.keys():
        y.append(str(bool(int(key))))

    plt.figure(figsize=(10, 5))
    ax = plt.subplot()
    ax.pie(x=list(count.values()), labels=y, autopct='%1.1f%%', shadow=True, startangle=90, pctdistance=0.85, colors=['#ffcc99', 'lightblue'])

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path)
    return


def cleaned_claims_lst(data, col_name):
    claim_lst = [str(data[col_name][idx]) for idx in data.index]
    cleaned_lst = []
    for claim in claim_lst:
        tokens = word_tokenize(claim)
        words = [word.lower() for word in tokens if word.isalpha()]
        cleaned_lst.append(words)
    # print([str(data[col_name][idx]).replace('[^\w\s]', '').lower().split(' ') for idx in data.index])
    return cleaned_lst


def remove_stopWord(textword_list):
    stop_words = set(stopwords.words('english') + ['think', ' ', '', 'partly','true','false'])
    result = []
    for list in textword_list:
        result.append([w for w in list if not w in stop_words])
    return result

def get_corpus(data, col_name):
    claim_lst = cleaned_claims_lst(data, col_name)
    return [word for claim in claim_lst for word in claim]



def plot_text_length_barchart(data, col_name, xlabel, ylabel, legend, output_path):
    plt.figure(figsize=(6, 4))
    ax = plt.subplot()
    bins = np.linspace(0, 200, 20)
    # data[col_name].str.len().hist(bins=bins, label=legend, alpha=0.5, color='lightblue', histtype='stepfilled')
    claim_list = cleaned_claims_lst(data, col_name)
    x = []
    for claim in claim_list:
        x.append(len(claim))
    plt.hist(x=x, bins=bins, label=legend, alpha=0.5, color='lightblue', histtype='stepfilled')

    x = pd.DataFrame(x, columns=['len'])
    ax.axvline(x['len'].mean(), linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for dat, xoff in zip([x['len']], [15, -15]):
        x0 = dat.mean()
        align = 'left' if xoff > 0 else 'right'
        ax.annotate('Mean: {:0.2f}'.format(x0), xy=(x0, 1), xytext=(xoff, 15),
                    xycoords=('data', 'axes fraction'), textcoords='offset points',
                    horizontalalignment=align, verticalalignment='center',
                    arrowprops=dict(arrowstyle='-|>', fc='black', shrinkA=0, shrinkB=0,
                                    connectionstyle='angle,angleA=0,angleB=90,rad=10'),
                    )

    ax.legend(loc='upper right')
    ax.margins(0.05)
    plt.savefig(output_path)
    return


def plot_top_stopwords_barchart(data, col_name, legend, output_path):
    plt.figure(figsize=(10, 5))
    ax = plt.subplot()
    stop = set(nltk.corpus.stopwords.words("english"))
    corpus = get_corpus(data, col_name)
    stopWordCnt = defaultdict(int)
    for word in corpus:
        if word in stop:
            stopWordCnt[word] += 1

    top = sorted(stopWordCnt.items(), key=lambda x: x[1], reverse=True)[:10]
    x, y = zip(*top)
    plt.bar(x, y, label=legend, color='lightblue')
    ax.legend(loc='upper right')
    plt.xticks(rotation=0)
    plt.savefig(output_path)
    return list(x)


def plot_top_non_stopwords_barchart(data, col_name, legend, output_path):
    plt.figure(figsize=(10, 5))
    ax = plt.subplot()
    stop = set(stopwords.words('english')+['think', ' ', ''])
    corpus = get_corpus(data, col_name)
    counter = Counter(corpus)
    most = counter.most_common()
    x, y = [], []
    for word, count in most[:40]:
        if (word not in stop):
            x.append(word)
            y.append(count)

    # plot = sns.barplot(x=x, y=y, palette="Spectral", label=legend)
    # ax.set(xlabel = 'Word', ylabel = 'Count')
    # ax.legend(ncol=1, loc='upper right')
    # plot.set_xticklabels(plot.get_xticklabels(), rotation=30)
    # plt.savefig(output_path)

    plt.bar(x, y, label=legend, color='lightblue')
    ax.legend(loc='upper right')
    plt.xticks(rotation=0)
    plt.savefig(output_path)
    return x

def get_top_N_ngram(data, col_name, n_ngram, n_TopN, output_path):
    claim_list = remove_stopWord(cleaned_claims_lst(data, col_name))

    ngram_dict = {}
    for claim in claim_list:
        dict = pd.Series(nltk.ngrams(claim, n_ngram)).value_counts().to_dict()
        ngram_dict = Counter(ngram_dict) + Counter(dict)
    top_n_bigrams = {k: v for k, v in sorted(ngram_dict.items(), key=lambda item: item[1], reverse=False)[-n_TopN:]}
    top_n_bigrams_de = {k: v for k, v in sorted(ngram_dict.items(), key=lambda item: item[1], reverse=True)[:n_TopN]}
    keys = []
    for k in top_n_bigrams.keys():
        keys.append(str(k))
    vals = list(top_n_bigrams.values())

    plt.figure(figsize=(25, 10))
    ax = plt.subplot()
    plt.barh(keys, vals, label='Count', color='lightblue')
    ax.legend(loc='upper right')
    plt.savefig(output_path)
    return top_n_bigrams_de



def main(data_path):
    data = pd.read_csv(data_path)
    data_Class0 = data.loc[data['NoteTrend'] == 0]
    data_Class1 = data.loc[data['NoteTrend'] == 1]
    # print(data['NoteTrend'].value_counts())
    # print(data.info())
    # print(data_Class0.info())
    # print(data_Class1.info())

    # Plot Class Distribution
    plot_by_class(data, col_name='NoteTrend', output_path='Plot_DataOverview_TrendDist')

    # Claim Length Analysis
    plot_text_length_barchart(data,
                              col_name='Claim',
                              xlabel='Length of Claim',
                              ylabel='Frequency',
                              legend='Claim Length',
                              output_path='Plot_DataOverview_ClaimLength')

    plot_text_length_barchart(data_Class0,
                              col_name='Claim',
                              xlabel='Length of Claim',
                              ylabel='Frequency',
                              legend='Claim Length',
                              output_path='Plot_DataOverview_ClaimLength_NegativeClass')

    plot_text_length_barchart(data_Class1,
                              col_name='Claim',
                              xlabel='Length of Claim',
                              ylabel='Frequency',
                              legend='Claim Length',
                              output_path='Plot_DataOverview_ClaimLength_PositiveClass')


    # Non-Stop Word Count
    plot_top_non_stopwords_barchart(data,
                                col_name='Claim',
                                legend='Non-Stop Words Count',
                                output_path='Plot_DataOverview_NonStopWords')

    # Stop Word Count
    plot_top_stopwords_barchart(data,
                                col_name='Claim',
                                legend='Stop Words Count',
                                output_path='Plot_DataOverview_StopWords')


    # N-gram
    top10bigram = get_top_N_ngram(data,
                    col_name='Claim',
                    n_ngram=2,
                    n_TopN=10,
                    output_path='Plot_DataOverview_Bigram')

    get_top_N_ngram(data,
                    col_name='Claim',
                    n_ngram=3,
                    n_TopN=10,
                    output_path='Plot_DataOverview_Trigram')

    top10bigram_class0 = get_top_N_ngram(data_Class0,
                    col_name='Claim',
                    n_ngram=2,
                    n_TopN=10,
                    output_path='Plot_DataOverview_Bigram_Class0')

    get_top_N_ngram(data_Class0,
                    col_name='Claim',
                    n_ngram=3,
                    n_TopN=10,
                    output_path='Plot_DataOverview_Trigram_Class0')

    top10bigram_class1 = get_top_N_ngram(data_Class1,
                    col_name='Claim',
                    n_ngram=2,
                    n_TopN=10,
                    output_path='Plot_DataOverview_Bigram_Class1')

    get_top_N_ngram(data_Class1,
                    col_name='Claim',
                    n_ngram=3,
                    n_TopN=10,
                    output_path='Plot_DataOverview_Trigram_Class1')

    print(top10bigram)
    print(top10bigram_class0)
    print(top10bigram_class1)

if __name__ == '__main__':
    main(data_path='DataClaimTrend.csv')