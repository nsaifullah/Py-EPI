import string
from gensim import corpora


def remove_singleton_words(list_of_lines: list[list[str]]):
    freq_d = {}
    for line in list_of_lines:
        for token in line:
            freq_d[token] += 1

    out_lines = [
        [token for token in line if freq_d[token] > 1]
        for line in list_of_lines
    ]

    return out_lines


root_dir = r'C:\Users\nikhi\Dropbox\JaldiKaro\DataScience'

punc_d = {p: '' for p in list(string.punctuation)}
no_punc_tbl = str.maketrans(punc_d)
docu_lines = []
with open(rf'{root_dir}/NLP/src/Why I am Not a Liberal - D Brooks - NYT.txt', 'r', encoding='utf-8') as input_file:
    input_line = input_file.readline()
    while input_line != '':
        docu_lines.append(input_line)
        input_line = input_file.readline()

docu_lines = [w.strip('\n').translate(no_punc_tbl) for w in docu_lines if w != '\n']
processed_lines = [
    [word for word in doc_line.lower().split()]
    for doc_line in docu_lines
]

dictionary = corpora.Dictionary(processed_lines)
dictionary.save(f'{root_dir}/NLP/src/dbrooks_part_one.dict')
