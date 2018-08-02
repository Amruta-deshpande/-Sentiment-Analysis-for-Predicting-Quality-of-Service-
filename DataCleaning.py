
import csv
import re


file_name = "dataset/results_computer_with_stars.csv"
cleaned_file_name = "program_output/cleaned_results_computer_with_stars.csv"

# file_name = "dataset/results_business_with_stars.csv"
# cleaned_file_name = "program_output/cleaned_results_business_with_stars.csv"

# file_name = "dataset/results_mobile_with_stars.csv"
# cleaned_file_name = "program_output/cleaned_results_mobile_with_stars.csv"


def main():
    """
    This function drives the execution of the program.
    :return:
    column 14- page function result/website
    column 15- page function result/review description
    column 16- page function result/review title
    column 17 -page function result/rating.
    """
    data = []
    with open(file_name, encoding="utf8") as source:
        rdr = csv.reader(source)
        for r in rdr:
            r = [replace_invalid_characters(d) for d in r]
            number_of_words_in_review = len(r[15].split(" "))
            if (r[14] != "" and r[15]!= "" and r[16] != "" and r[17] != "") and number_of_words_in_review > 5:
                data.append([r[14], r[16], r[15], r[17]])

    data[0] = [d.replace('pageFunctionResult/', '') for d in data[0]]

    for row in data:
        if int(row[3]) > 3:
            row.append('0')
        else:
            row.append('1')
    write_list_of_lists_to_csv(data, cleaned_file_name)


def replace_invalid_characters(input_data):
    """
     This function removes the invalid characters from the input.
    :param input_data: data to be cleaned.
    :return:
    """
    # List of characters to be removed from user reviews.
    for ch in ["&", "#", "\t", "\n", '”', "“", "\'", '"', "'", ".", "!", ",", "%", "$", "&", "*", "/", "(", ")"
               "@", "-", ">", "<", "?", "+", ";", ":","[", "]", "{", "},", "_", "|", "~", "`", "’"]:
        if ch in input_data:
            input_data = input_data.replace(ch, "")
            input_data = input_data.strip()

    # Replace URL's in string
    input_data = re.sub(r'^https?:\/\/.*[\r\n]*', '', input_data, flags=re.MULTILINE)

    return input_data


def print_list_of_list(data):
    """
    Print list of list
    :param data: list of list data
    :return:
    """
    for row in data:
        print(row)


def write_list_of_lists_to_csv(data, file_name):
    """
    This function is used to write a list of list data into a file
    :param data: list of list data
    :param file_name: name of the target file.
    :return:
    """
    with open(file_name, 'w', encoding='utf-8') as f:
        writer = csv.writer(f,lineterminator='\n')
        writer.writerows(data)

if __name__ == '__main__':
    main()
