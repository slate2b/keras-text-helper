"""
Prep Data for Tokenizer

This script accepts a dataset file along with a vocabulary file
and prepares the dataset for tokenization.  It is designed
for use with the Tokenizer from keras.preprocessing.text.

Specifically, the script lemmatizes the text tokens from the
input dataset, then it compares the lemmatized tokens to
the tokens in the given vocabulary.  It then removes any tokens
from the dataset which are not found in the vocabulary.  When
finished, it saves the prepped data as a comma separated value
file (.csv).

This tool accepts dataset files in the form of comma separated
value files (.csv), and it accepts vocabulary files in the form
of text files (.txt).

This file can also be imported as a module and contains the following
functions:

    * prep_data_for_tokenizer - preps the data and returns it as a list
    * save_prepped_data - converts the prepped data and saves it as a .csv file
    * main - the main function of the script
"""


import pandas as pd
from nltk.stem import WordNetLemmatizer
import time
from tqdm import tqdm


def prep_data_for_tokenizer(vocab_fname, dataset_fname, max_tkns):
    """
    Lemmatizes the text tokens from the input data and then removes the
    text tokens from each line which are not in the given vocabulary.

    :param vocab_fname: str - the filename for the vocabulary
    :param dataset_fname: str - the filename for the input data
    :param max_tkns: int - maximum number of tokens per line
    :return: list - the prepped data as a list of strings
    """

    # load the source text
    source_filename = dataset_fname
    source_file = open(source_filename, 'r', encoding='utf-8')
    source_text = source_file.read()  # Read file into a text string
    source_file.close()

    # load the pre-defined vocabulary
    vocab_filename = vocab_fname
    vocab_file = open(vocab_filename, 'r', encoding='utf-8')
    vocab_text = vocab_file.read()
    vocab_file.close()

    # Remove punctuation from the source text string
    table = str.maketrans(',', ' ')
    source_text = source_text.translate(table)

    # Remove double-quotes to avoid python quoting double-quotes and throwing off the vocab
    # Example: the description [tortilla 6"] turns into ["tortilla ""]
    # Which means that the term went from [tortilla] to ["tortilla] which is not present in the vocabulary
    # The following code will replace the double-quotes that should be there with the text [doublequote] and remove any
    # additional double-quotes
    while '""' in source_text:
        source_text = source_text.replace('""', 'doublequote')
    while '"' in source_text:
        source_text = source_text.replace('"', '')
    while 'doublequote' in source_text:
        source_text = source_text.replace('doublequote', '"')

    # Split the source text into lines using \n as delimiter
    source_lines = source_text.split(sep='\n')

    print("First five elements of source_lines: \n")
    for i in range(5):
        print("line # " + str(i + 1) + ":")
        print(source_lines[i + 1])

    # Create a Lemmatizer to convert plural forms of words to singulars to optimize vocabulary
    wnl = WordNetLemmatizer()

    # Create a list to hold the lemmatized lines
    lemmatized_lines = []

    # loop through each line
    for i in range(len(source_lines)):
        line = source_lines[i]
        lemmatized_line = ""
        words_in_line = line.split()

        # loop through each word in the current line
        for j in range(len(words_in_line)):
            word = words_in_line[j]
            word = word.lower()

            if word != "us" and word != "ps" and word != "bs" and word != "as" and word != "es":  # COO's to prevent lemmatizer from modifying
                word = wnl.lemmatize(word)
            if j == 0:
                lemmatized_line = word
            else:
                lemmatized_line = lemmatized_line + " " + str(word)

        # Add the lemmatized line to the lemmatized_lines list
        lemmatized_lines.append(lemmatized_line)

    print("\nFirst five elements of lemmatized_lines: ")
    for i in range(5):
        print("line # " + str(i + 1) + ":")
        print(lemmatized_lines[i + 1])

    # Remove the header line
    lemmatized_lines.pop(0)

    # Split the vocab text
    vocab_tokens = vocab_text.split()

    # Split the lemmatized text
    lemmatized_text = ""

    for i in range(len(lemmatized_lines)):

        if i == 0:
            lemmatized_text = lemmatized_lines[i]
        else:
            lemmatized_text = lemmatized_text + lemmatized_lines[i]

    lemmatized_tokens = lemmatized_text.split()

    # Turn the vocab text into a set
    vocab = set(vocab_tokens)

    print("\nRemoving all tokens from the source tokens which are not in the vocabulary...")

    # Remove all tokens from source tokens which are not in the vocabulary
    all_tokens_in_vocab_list = [w for w in lemmatized_tokens if w in vocab]

    # Converting to a set to increase efficiency from O(n) to O(1) in the for loop below
    all_tokens_in_vocab = set(all_tokens_in_vocab_list)

    # Loop through source text one line at a time, remove all words not in vocabulary, then add them to refined_source_text
    refined_source_text = []

    print("\nRemoving words from the source text which aren't in the vocabulary...")

    time.sleep(1)  # Short delay to manage print messages for tqdm

    for i in tqdm(range(len(lemmatized_lines))):

        refined_line = ""
        line = lemmatized_lines[i]  # single line from the source text
        tokens = line.split()  # split the line into individual terms (by whitespace)
        tokens_in_vocab = []
        for tkn in tokens:
            if len(tokens_in_vocab) > max_tkns:  # Limiting to 20 tokens to narrow the focus for the models
                break
            if tkn in all_tokens_in_vocab:
                tokens_in_vocab.append(tkn)
        for tkn in tokens_in_vocab:
            if tkn not in refined_line:
                refined_line = refined_line + " " + tkn

        # Remove initial space character
        finished_line = ""
        for j in range(len(refined_line)):
            if j != 0:
                finished_line = finished_line + refined_line[j]

        # Append the finished line to the refined source text list
        refined_source_text.append(finished_line)

    print("\nFirst five elements of refined_source_text after removing non-vocab terms: \n")
    for i in range(5):
        print("line # " + str(i + 1) + ":")
        print(refined_source_text[i])

    # This data set ended up with a blank line at the end
    last_line_index = len(refined_source_text) - 1
    refined_source_text.pop(last_line_index)

    return refined_source_text


def save_prepped_data(filepath, data):
    """
    Saves the prepped data to a .csv file

    :param filepath : str - the filepath for the CSV input file
    :param data : list - the prepped data as a list of strings
    :return : None
    """

    # Convert the data to a Pandas Series to make it easy the data to a file
    data_series = pd.Series(data=data)

    # Cleanup the filepath for the prepped file
    if '.csv' in filepath:
        filepath = filepath.replace('.csv', '')

    prepped_data_filename = filepath + '_prepped_for_tokenizer.csv'

    print("\nSaving prepped data as: " + str(prepped_data_filename))

    data_series.to_csv(path_or_buf=prepped_data_filename, sep=',', encoding='utf-8', index=False)


def main():

    print("\n*****************************\n"
          "*  PREP DATA FOR TOKENIZER  *\n"
          "*****************************\n")

    dataset_filepath = input("Please enter filepath to the dataset file (CSV format): ")
    vocabulary_filepath = input("Please enter filepath to the vocabulary file (TXT format): ")
    max_tokens = int(input("\nPlease enter the maximum number of tokens per line of text: "))

    if ".csv" not in dataset_filepath:
        dataset_filepath = dataset_filepath + ".csv"
    if ".txt" not in vocabulary_filepath:
        vocabulary_filepath = vocabulary_filepath + ".txt"

    prepared_data = prep_data_for_tokenizer(dataset_fname=dataset_filepath, vocab_fname=vocabulary_filepath,
                                            max_tkns=max_tokens)

    save_prepped_data(filepath=dataset_filepath, data=prepared_data)

    print("\n*************************\n"
          "*  DATA PREP COMPLETED  *\n"
          "*************************\n")

    exit()


if __name__ == "__main__":
    main()
