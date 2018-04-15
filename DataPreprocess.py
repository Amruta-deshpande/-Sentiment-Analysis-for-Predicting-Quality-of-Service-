import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk import FreqDist
import csv

input_file = "program_output/cleaned_results_computer_with_stars_small.csv"
frequency_matrix_file = 'program_output/freqmatrix_computer_small.csv'
preprocess_file = 'program_output/preprocess_computer_small.csv'

# input_file = "program_output/cleaned_results_business_with_stars.csv"
# frequency_matrix_file = 'program_output/freqmatrix_business.csv'
# preprocess_file = 'program_output/preprocess_business.csv'

# input_file = "program_output/cleaned_results_mobile_with_stars.csv"
# frequency_matrix_file = 'program_output/freqmatrix_mobile.csv'
# preprocess_file = 'program_output/preprocess_mobile.csv'


def read_file(filename):
    """
    This function will read the data from given CSV file and store data in a list
    :param filename: Name of the File
    :return: List of speed, reckless and threshold values
    """
    all_reviews = []
    class_result = []
    try:
        if filename.endswith('.csv'):
            with open(filename, encoding="utf-8") as f:

                lines = csv.reader(f)
                for i, line in enumerate(lines):
                    if i == 0:
                        continue
                    else:
                        all_reviews.append(word_tokenize(line[2]))
                        class_result.append(word_tokenize(line[4]))
                    #break;
                return all_reviews, class_result

    except FileNotFoundError:
        print("Put both CSV file in the same folder as program file.")


def removed_stopwords(input_list):
    """
    This function removes stop_words.
    :param input_list:
    :return:
    """
    stop_words = set(stopwords.words('english'))
    _removed_stopwords = [w.lower() for w in input_list if not w in stop_words]

    return _removed_stopwords


def stemming(input_list):
    """
    This function performs stemming on the input.
    :param input_list: data.
    :return:
    """
    ps = PorterStemmer()
    _stemming = []
    for w in input_list:
        _stemming.append(ps.stem(w))
    return _stemming


def pos_tagging(input_list):
    """
    Perform POS Tagging on the input.
    :param input_list:
    :return:
    """
    _pos_tagging = []
    for one_list in input_list:
        _pos_tagging.append(nltk.pos_tag(one_list))
    return _pos_tagging


def pos_filter(input_list):
    """
    Apply POS Filter on the input data.
    :param input_list:
    :return:
    """
    _pos_filter = []
    _pos_filter_words = []
    for one_item in input_list:
        if one_item[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']:  #
            _pos_filter.append(one_item)
            _pos_filter_words.append(one_item[0].lower())
    return _pos_filter,_pos_filter_words


def Freq_Matrix(words_list):
    """
    Generate frequency matrix from the input.
    :param words_list:
    :return: frequency matrix as a list of list, and unique columns
    """
    all_words = []
    for one_list in words_list:
        for one_item in one_list:
            all_words.append(one_item)

    _freq_matrix = FreqDist(all_words)
    unique_col = []
    for key, val in _freq_matrix.items():
        if val > 10:
            unique_col.append(key)

    return _freq_matrix, unique_col


def main():
    # --------Read the Data-------------------#
    all_reviews,class_result = read_file(input_file)

    # --------Removing the stop words-------------------#
    _all_filtered = []
    for one_list in all_reviews:
        _all_filtered.append(removed_stopwords(one_list))

    # --------Performing stemming on stop words-------------------#
    _stem_words = []
    for one_list in _all_filtered:
        _stem_words.append(stemming(one_list))

    # --------Performing POS tagging-------------------#
    _pos_tagging = pos_tagging(_stem_words)

    # --------Filtering POS tagging containing only Nouns and Adjectives-------------------#
    _pos_filter = []
    _pos_filter_words = []
    for one_list in _pos_tagging:
        tup, words = pos_filter(one_list)
        _pos_filter.append(tup)
        _pos_filter_words.append(words)

    # --------Create the Freqeuency Matrix for each review-------------------#
    _freq_each_review = []
    for one_list in _pos_filter_words:
        tmp = FreqDist(one_list)
        _freq_each_review.append(tmp)

    # --------Create the Freqeuency Matrix for all reviews-------------------#

    _freq_matrix, unique_col = Freq_Matrix(_pos_filter_words)

    # --------Create the Freqeuency Matrix File-------------------#
    fw = open(frequency_matrix_file, 'w', encoding="utf-8", newline='')
    wr = csv.writer(fw)
    unique_col.insert(0, '')
    wr.writerow(unique_col+['Class_Result'])

    rdr = csv.reader(fw)

    result = []
    index = 0
    for word_row,review_row in zip(_pos_filter_words,_freq_each_review):
        row_data =[word_row]+[0]*(len(unique_col)-1)+ class_result[index]
        for one_item in word_row:
            count = review_row[one_item]
            if one_item in unique_col:
                col_no = unique_col.index(one_item)
                row_data[col_no] = count
        result.append(row_data)
        index += 1

    for row_value in result:
        wr.writerow(row_value)

    # --------Create the PreProcess CSV File-------------------#

    fw =  open(preprocess_file, 'w', encoding="utf-8", newline='')
    wr = csv.writer(fw)
    wr.writerow(['Word Tokenized','Removed Stop Words','Stemming','POS Tagging','POS Filter with Noun and Adjectives'])

    for index in range(len(all_reviews)):
        wr.writerow([','.join(all_reviews[index]),','.join(_all_filtered[index]),','.join(_stem_words[index]),
                     ','.join(map(str,_pos_tagging[index])),','.join(map(str,_pos_filter[index])),
                     ','.join(_pos_filter_words[index])]) #,','.join(_freq_each_review[index])])

if __name__ == '__main__':
    main()
