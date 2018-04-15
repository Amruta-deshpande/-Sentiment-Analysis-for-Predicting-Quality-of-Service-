import csv
import nltk
from DataPreprocess import pos_filter, stemming
from DataCleaning import replace_invalid_characters

gt_file_name = "program_output/Ground_Truth.csv"
reviews_file_name = "program_output/review_categorization.csv"
results_file_name = "program_output/p_value_based_model_small.csv"
logistic_file_name = "program_output/p_value_based_model_logistic_small.csv"
frequency_results_file_name = "program_output/Frequency_Based_Data_small.csv"


def read_file(file_name):
    data = []
    with open(file_name) as source:
        rdr = csv.reader(source)
        headers = next(rdr)
        for r in rdr:
            if len(r) > 0:
                r = [replace_invalid_characters(d.strip()) for d in r]
                data.append(r)
    return data


def print_list_of_list(data):
    """
    Print list of list
    :param data: list of list data
    :return:
    """
    for row in data:
        print(row)


def compute_accuracy(predicted, ground_truth):
    """
    Compute Accuracy by dividing (the intersection of ground truth and the predicted values) with the len(ground truth)
    :param predicted: features predicted by the algorithm
    :param ground_truth: features in the ground_truth
    :return: accuracy.
    """
    gt = len(set(ground_truth))
    number_of_extracted_features = 0
    for item in set(predicted):
        if item in ground_truth:
            number_of_extracted_features += 1
    print(number_of_extracted_features, "/", gt)
    return number_of_extracted_features / gt


"""
def each_review_compute_accuracy(predicted, ground_truth):
    ne = []
    numerator = 0
    for item in ground_truth:
        # print(item)
        total = 0
        count = 0
        for keyword in item:
            if keyword:
                if keyword in predicted:
                    count += 1
                total += 1
        numerator += count / (total if total > 0 else 1)
        ne.append(count/(total if total > 0 else 1))

    print(numerator, "/", len(ne))
    print("Algorithm Accuracy:", (numerator / len(ne)))

    return numerator, ne
"""


def pos_tagging(input_list):
    """
    Perform POS Tagging on the input.
    :param input_list:
    :return:
    """
    # print(input_list)
    _pos_tagging = []
    _pos_tagging.append(nltk.pos_tag(input_list))
    return _pos_tagging


def find_top_k_features(data, k):
    """
    Identify the top k features in data based on the number of times a feature occurs.
    :param data: data in which top k features need to be found
    :param k: the number of top features to be identified.
    :return: k features
    """
    k_features = list()

    for row in data:
        feature = list()
        feature.append(row[0])
        feature.append(max(int(row[1]), int(row[2])))

        k_features.append(feature)

    k_features.sort(key=lambda x: int(x[1]))
    k_features.reverse()

    k_features = [feature[0] for feature in k_features[0:k]]

    return k_features


def main():
    """
    This function drives the execution of the program.
    :return:
    """
    data = read_file(gt_file_name)

    manual_dataset = [feature[0] for feature in data]

    manual_dataset_stem = list(set(stemming(manual_dataset)))

    # --------Performing POS tagging-------------------#
    manual_dataset_stem_pos = pos_tagging(manual_dataset_stem)

    # --------Filtering POS tagging containing only Nouns and Adjectives-------------------#
    _pos_filter = []
    _pos_filter_words = []
    for one_list in manual_dataset_stem_pos:
        tup, words = pos_filter(one_list)
        _pos_filter.append(tup)
        _pos_filter_words.append(words)

    manual_dataset_stem_pos = _pos_filter_words[0]

    k_values = [30, 40, 50]
    for k in k_values:
        print("###################################### For k: ", k, "######################################")
        algo_data = read_file(results_file_name)
        algo_dataset = [feature[0:3] for feature in algo_data]
        algo_dataset = find_top_k_features(algo_dataset, k)

        frequency_data = read_file(frequency_results_file_name)
        frequency_dataset = [feature[0:3] for feature in frequency_data]
        frequency_dataset = find_top_k_features(frequency_dataset, k)

        logistic_data = read_file(logistic_file_name)
        logistic_dataset = [feature[0:3] for feature in logistic_data]
        logistic_dataset = find_top_k_features(logistic_dataset, k)

        print("Algorithm Accuracy: ", compute_accuracy(algo_dataset, manual_dataset_stem_pos))
        print("Frequency Based Accuracy: ", compute_accuracy(frequency_dataset, manual_dataset_stem_pos))
        print("Logisitic Based Accuracy: ", compute_accuracy(logistic_dataset, manual_dataset_stem_pos))

        print("\n*******************************************************************\n")

    """
    reviews = read_file(reviews_file_name)
    review_dataset = ([review[4::] for review in reviews])

    review_dataset_stem = []
    for one_list in review_dataset:
        review_dataset_stem.append(stemming(one_list))

    each_review_compute_accuracy(algo_dataset, review_dataset_stem)
    each_review_compute_accuracy(frequency_dataset[1:133], review_dataset_stem)
    """

if __name__ == '__main__':
    main()