def baseline(train, test):
    """
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    # (III): for each word seen in training, predict its most frequent tag.
    # For words not seen in training, predict the most frequent tag overall.
    words = {}

    for sentence in train:
        for word, tag in sentence:
            if word not in words:
                words[word] = {}
            if tag not in words[word]:
                words[word][tag] = 1
            else:
                words[word][tag] += 1

    max_words = {w: max(t, key=t.get) for w, t in words.items()}
    freq_words = [max(t.values()) for w, t in words.items()]
    overall_max = max(freq_words)

    preds = [
        [
            (word, max_words[word] if word in max_words else overall_max)
            for word in sentence
        ]
        for sentence in test
    ]
    return preds
