# NOTE: math.log(float) is much faster than np.log(float)
import argparse
import math

# smoothing constants for initial/transition probabilities and emissions, respectively
# you do not need to change these
transition_epsilon = 1e-5
emit_epsilon = 1e-5


def viterbi(train_data, test_data):
    """
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
    """
    init_prob, emit_prob, trans_prob = train(train_data)
    tags = sorted(emit_prob.keys())
    predicts = []

    for sentence in test_data:
        log_prob = {}
        predict_tag_seq = {}
        for tag in tags:
            log_prob[tag] = init_prob[tag]
            predict_tag_seq[tag] = []

        for i, word in enumerate(sentence):
            log_prob, predict_tag_seq = stepforward(
                i,
                word,
                log_prob,
                predict_tag_seq,
                emit_prob,
                trans_prob,
            )

        best_tag = max(log_prob, key=log_prob.get)
        predicts.append(
            [(word, predict_tag_seq[best_tag][i]) for i, word in enumerate(sentence)]
        )

    return predicts


class Counts:
    def __init__(self, name: str):
        self.name = name

        self.inits = 0
        self.total = 0
        self.outgoing = 0
        self.transitions: dict[str, int] = {}
        self.emissions: dict[str, int] = {}

    def incr(self, word, prev_tag):
        self.total += 1
        self.emissions[word] = self.emissions.get(word, 0) + 1

        if prev_tag is None:
            self.inits += 1

        if prev_tag is not None:
            self.transitions[prev_tag] = self.transitions.get(prev_tag, 0) + 1

    def incr_outgoing(self):
        self.outgoing += 1

    def hapax(self, words):
        return sum([1 if w in self.emissions else 0 for w in words])

    def __eq__(self, other):
        if not isinstance(other, Counts):
            return NotImplemented
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


def train(sentences):
    # (III): estimate the HMM parameters from the tagged training data.
    # Return three dictionaries:
    # 1. init_prob[tag] = log probability that the first tag in a sentence is tag
    # 2. emit_prob[tag][word] = log probability that tag emits word
    # 3. trans_prob[prev_tag][next_tag] = log probability that next_tag follows prev_tag
    # Apply regular Laplace smoothing to initial probabilities for every observed tag:
    # (count_first(tag) + transition_epsilon) / (len(sentences) + transition_epsilon * number_of_tags)
    # Apply regular Laplace smoothing to every possible pair of tag-to-tag transition:
    # (count(prev_tag, next_tag) + transition_epsilon) / (outgoing_count(prev_tag) + transition_epsilon * number_of_tags)
    # For emission probabilities use hapax-aware smoothing:
    # - count words that appear exactly once in the whole training set
    # - count how many of those hapax words belong to each tag
    # - use epsilon = emit_epsilon * max(1, hapax_for_tag) / max(1, total_hapax)
    # - do laplace smoothing using this epsilon and include an explicit UNSEEN emission entry for every tag
    # Store natural logs of all three smoothed probabilities to avoid repeated logs during stepforward

    counts: dict[str, Counts] = {}
    words: dict[str, int] = {}

    for i in range(len(sentences)):
        sentence = sentences[i]
        prev_tag = None
        for j in range(len(sentence)):
            word, tag = sentence[j]
            if tag not in counts:
                counts[tag] = Counts(name=tag)

            counts[tag].incr(word, prev_tag)

            if prev_tag is not None:
                counts[prev_tag].incr_outgoing()

            if word not in words:
                words[word] = 1
            else:
                words[word] += 1

            prev_tag = tag

    number_of_tags = len(counts)
    init_prob = {
        tag: math.log(
            (count.inits + transition_epsilon)
            / (len(sentences) + transition_epsilon * number_of_tags)
        )
        for tag, count in counts.items()
    }

    hapax_words = [w for w, c in words.items() if c == 1]
    total_hapax = len(hapax_words)
    hapax_T = {
        tag: sum(1 for w in count.emissions if words[w] == 1)
        for tag, count in counts.items()
    }
    alpha_T = {
        tag: emit_epsilon * max(1, hT) / max(1, total_hapax)
        for tag, hT in hapax_T.items()
    }
    emit_prob = {
        tag: {
            word: math.log(
                (count.emissions.get(word, 0) + alpha_T[tag])
                / (count.total + alpha_T[tag] * (len(count.emissions) + 1))
            )
            for word in list(count.emissions.keys()) + ["UNSEEN"]
        }
        for tag, count in counts.items()
    }

    trans_prob = {
        prev_tag: {
            next_tag: math.log(
                (counts[next_tag].transitions.get(prev_tag, 0) + transition_epsilon)
                / (counts[prev_tag].outgoing + transition_epsilon * number_of_tags)
            )
            for next_tag in counts
        }
        for prev_tag in counts
    }

    return init_prob, emit_prob, trans_prob


def stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    # (III): advance the Viterbi trellis by one observed word.
    # Parameters:
    # - i: index of the current word in the sentence
    # - word: the observed word at position i
    # - prev_prob[tag]: best log-probability for each tag in the previous column
    # - prev_predict_tag_seq[tag]: best tag sequence ending in each previous tag
    # - emit_prob, trans_prob: log probabilities returned by train; use them directly
    # Return log_prob and predict_tag_seq dictionaries for every current tag

    predict_tag_seq = {}
    log_prob = {}
    for tag, tag_emit_probs in emit_prob.items():
        if i > 0:
            next_trans_prob = {
                prev_tag: prev_tag_prob + trans_prob[prev_tag][tag]
                for prev_tag, prev_tag_prob in prev_prob.items()
            }
            max_trans_prob = max(next_trans_prob.values())
            max_trans_tag = max(next_trans_prob, key=next_trans_prob.get)
        else:
            max_trans_prob = prev_prob[tag]
            max_trans_tag = tag
        obs_likelihood = tag_emit_probs.get(word, tag_emit_probs["UNSEEN"])
        log_prob[tag] = obs_likelihood + max_trans_prob
        predict_tag_seq[tag] = prev_predict_tag_seq[max_trans_tag] + [tag]

    return log_prob, predict_tag_seq


if __name__ == "__main__":
    import utils
    from baseline import baseline

    parser = argparse.ArgumentParser(description="CS440 MP6 HMM POS Tagging")
    parser.add_argument(
        "--train",
        dest="training_file",
        type=str,
        required=True,
        help="Training data file",
    )
    parser.add_argument(
        "--test", dest="test_file", type=str, required=True, help="Testing data file"
    )
    parser.add_argument(
        "--algorithm",
        choices=("baseline", "viterbi"),
        default="viterbi",
        help="Which tagger to run",
    )
    args = parser.parse_args()

    print("Loading dataset...")
    train_set = utils.load_dataset(args.training_file)
    test_set = utils.load_dataset(args.test_file)
    print("Loaded dataset")
    print()

    algorithm = baseline if args.algorithm == "baseline" else viterbi

    print(f"Running {args.algorithm}...")
    predictions = algorithm(train_set, utils.strip_tags(test_set))
    accuracy, correct_wordtagcounter, wrong_wordtagcounter = utils.evaluate_accuracies(
        predictions, test_set
    )
    multitags_acc, unseen_acc = utils.specialword_accuracies(
        train_set, predictions, test_set
    )

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"\tMultitags Accuracy: {multitags_acc * 100:.2f}%")
    print(f"\tUnseen words Accuracy: {unseen_acc * 100:.2f}%")
    print(
        f"\tTop K Wrong Word-Tag Predictions: {utils.topk_wordtagcounter(wrong_wordtagcounter, k=4)}"
    )
    print(
        f"\tTop K Correct Word-Tag Predictions: {utils.topk_wordtagcounter(correct_wordtagcounter, k=4)}"
    )
