"""
Vocab Builder

This script reads an input file and builds a custom vocabulary
that can be used by machine learning models. The script was
designed for use specifically for simple Text Classification ML
models built using Keras, but it could be utilized in other
context as well.

This tool accepts comma separated value files (.csv).

This script requires that `nltk` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * build_vocabulary - returns a list containing:
       - [Index 0]: str - a newline-separated vocabulary string
       - [Index 1]: int - the number of tokens in the vocabulary
    * save_vocabulary - saves a newline-separated string to a text file
    * main - the main function of the script
"""

from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer


def build_vocabulary(filepath, min_occur, max_tkns):
    """
    Builds custom vocabulary from input data.

    :param filepath : str - the filepath for the CSV input file
    :param min_occur : int - the minimum threshold of occurrences to keep a token
    :param max_tkns : int - the maximum number of tokens for the vocabulary
    :return vocab_data : list
        - [Index 0] : str - a newline-separated vocabulary string
        - [Index 1] : int - the number of tokens in the vocabulary
    """

    print("\n* * * * *\n"
          "\nReading input data...\n")

    file = open(filepath + str('.csv'), encoding='utf-8')
    text = file.read()
    file.close()

    # Convert the input file into a list of individual token strings
    tokens = nltk.word_tokenize(text)

    # Create a Lemmatizer to convert plural forms of words to singulars to optimize vocabulary
    wnl = WordNetLemmatizer()

    # Lemmatize the tokens from the input file
    lemmatized_tokens = []
    for token in tokens:
        token = token.lower()
        token = wnl.lemmatize(token)
        lemmatized_tokens.append(token)

    # Create a vocabulary counter
    vocab_counter = Counter()

    # Update the vocab counter with the tokens
    vocab_counter.update(lemmatized_tokens)

    print("Number of tokens identified in the input data: " + str(len(vocab_counter)) + "\n")

    input("Press ENTER to continue...\n")

    print("Removing unwanted tokens...\n")

    # Remove unwanted terms from the vocabulary
    tokens_to_remove_list = ['crude', 'rude', 'offensive', 'etc']
    for item in tokens_to_remove_list:
        if item in vocab_counter:
            vocab_counter.pop(item)

    # Remove single-character tokens from vocabulary
    single_char_to_remove = []
    for token in vocab_counter:
        if len(token) == 1:
            single_char_to_remove.append(token)
    for token in single_char_to_remove:
        vocab_counter.pop(token)

    print("Removing tokens which failed to meet the minimum occurrence threshold...\n")

    # Remove tokens which fail to meet the minimum occurrence threshold
    minimum_occurrence_threshold = min_occur
    low_occurrence_tokens = [token for token in vocab_counter if vocab_counter[token] < minimum_occurrence_threshold]
    for token in low_occurrence_tokens:
        del vocab_counter[token]

    print("Updating vocabulary to the most common tokens...\n")

    # List of tuples for each token (token, count)
    vocab_tuples = vocab_counter.most_common(max_tkns)
    print(vocab_tuples)

    print("\nPausing to allow review of the vocabulary tokens and counts listed above.\n")
    input("Press ENTER to continue...\n")

    final_tokens = []

    # Extract the tokens from the tuples and add them to a final tokens list
    for item in vocab_tuples:
        current_token = item[0]
        final_tokens.append(current_token)

    print("Adding important tokens not captured in the input data... \n")

    # Adding important tokens to the vocab that might have been missing in the input data
    if "critical" not in final_tokens:
        final_tokens.append("critical")
        print(" - Added: critical\n")
    if "important" not in final_tokens:
        final_tokens.append("important")
        print(" - Added: important\n")
    if "necessary" not in final_tokens:
        final_tokens.append("necessary")
        print(" - Added: necessary\n")

    final_num_tokens = len(final_tokens)

    print("Final number of tokens: " + str(final_num_tokens) + "\n")

    # Join the elements in the word list into a newline-separated string
    data = "\n".join(final_tokens)

    # Build list to return
    vocab_data = [data, final_num_tokens]

    return vocab_data


def save_vocabulary(filepath, num_tkns, data):
    """
    Saves the vocabulary to a .txt file

    :param filepath : str - the filepath for the CSV input file
    :param num_tkns : int - the number of tokens in the vocabulary
    :param data : str - newline-separated vocabulary string
    :return : None
    """

    vocab_filename = filepath + '_vocab_' + str(num_tkns) + '_tokens.txt'

    print("Saving vocabulary as: " + str(vocab_filename))

    file = open(vocab_filename, 'w', encoding="utf-8")
    file.write(data)
    file.close()


def main():

    print("\n************************\n"
          "*  VOCABULARY BUILDER  *\n"
          "************************\n")

    input_filepath = input("Please enter filepath to the input file (CSV format): ")
    minimum_occurrences = int(input("\nPlease enter the minimum number of occurrences for valid tokens: "))
    max_tokens = int(input("\nPlease enter the maximum number of tokens to output: "))

    if ".csv" in input_filepath:
        input_filepath = input_filepath.replace(".csv", "")

    vocabulary_data = build_vocabulary(filepath=input_filepath, min_occur=minimum_occurrences, max_tkns=max_tokens)

    vocabulary = vocabulary_data[0]
    num_final_tokens = vocabulary_data[1]

    save_vocabulary(filepath=input_filepath, num_tkns=num_final_tokens, data=vocabulary)

    print("\n**************************\n"
          "*  VOCABULARY COMPLETED  *\n"
          "**************************\n")

    exit()


if __name__ == "__main__":
    main()
