"""Tokenizes the sentences with BertTokenizer as tokenisation costs some time.
"""

from transformers import BertTokenizer

#BERT_VOCAB = "bert_model/scibert_scivocab_uncased/vocab.txt"
BERT_VOCAB = "bert_model/bert_base_uncased/vocab.txt"
#BERT_VOCAB = "bert_model/bert_large_uncased/vocab.txt"
#BERT_NAME = "scibert"
BERT_NAME = "bert_base"
#BERT_NAME = "bert_large"
MAX_SEQ_LENGTH = 128


def tokenize_file(in_file, out_file, tokenizer):
    with open(in_file, encoding="utf-8") as in_f:
        with open(out_file, encoding="utf-8", mode="w") as out_f:
            for line in in_f:
                line = line.replace("\r", "")
                if line.strip() == "" or line.startswith("###"):
                    out_f.write(line + "\n")
                else:
                    ls = line.split("\t")
                    tag, sentence = ls[0], ls[1]
                    tokenized = tokenizer.encode(sentence, add_special_tokens=True, max_length=MAX_SEQ_LENGTH)
                    out_f.write(f'{tag}\t{" ".join([str(t) for t in tokenized])}\n')


def tokenize():
    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)
    print("nicta-piboso")
    tokenize_file("datasets/nicta-piboso/train_clean.txt", f"datasets/nicta-piboso/train_{BERT_NAME}.txt", tokenizer)
    tokenize_file("datasets/nicta-piboso/dev_clean.txt", f"datasets/nicta-piboso/dev_{BERT_NAME}.txt", tokenizer)
    tokenize_file("datasets/nicta-piboso/test_clean.txt", f"datasets/nicta-piboso/test_{BERT_NAME}.txt", tokenizer)

    print("pubmed-20k")
    tokenize_file("datasets/pubmed-20k/train_clean.txt", f"datasets/pubmed-20k/train_{BERT_NAME}.txt", tokenizer)
    tokenize_file("datasets/pubmed-20k/dev_clean.txt", f"datasets/pubmed-20k/dev_{BERT_NAME}.txt", tokenizer)
    tokenize_file("datasets/pubmed-20k/test_clean.txt", f"datasets/pubmed-20k/test_{BERT_NAME}.txt", tokenizer)

    print("DRI")
    tokenize_file("datasets/DRI/full_clean.txt", f"datasets/DRI/full_{BERT_NAME}.txt", tokenizer)

    print("ART")
    tokenize_file("datasets/ART/full_clean.txt", f"datasets/ART/full_{BERT_NAME}.txt", tokenizer)

    #for t in ["ART", "DRI", "NIC", "PMD"]:
    #    print(f"{t}_generic")
    #    for s in ["train", "dev", "test"]:
    #        tokenize_file(f"datasets/{t}_generic/{s}_clean.txt", f"datasets/{t}_generic/{s}_{BERT_NAME}.txt", tokenizer)



tokenize()
