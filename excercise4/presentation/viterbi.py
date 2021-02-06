import pandas

class HiddenMarkovModel:
    transitions = dict()
    emissions = dict()
    prior = dict()

    def __init__(self):
        self.transitions = {'O': {'O': 0, 'B': 0, 'I': 0},
                            'B': {'O': 0, 'B': 0, 'I': 0},
                            'I': {'O': 0, 'B': 0, 'I': 0}}
        self.prior = {'O': 0, 'B': 0, 'I': 0}

    def train_probabilities(self, train_set: pandas.DataFrame):
        trans = {'O': {'O': 0, 'B': 0, 'I': 0},
                 'B': {'O': 0, 'B': 0, 'I': 0},
                 'I': {'O': 0, 'B': 0, 'I': 0}}
        emi = dict()
        prior = {'O': 0, 'B': 0, 'I': 0}
        for index, row in train_set.iterrows():
            # The current sequence consists of (word, tag) pairs
            seq = list(zip(row['tokens'],row['target']))

            for i in range(len(seq)):
                # First tag: Prior probabilities
                if i == 0:
                    prior[seq[i][1]] += 1
                else:
                    trans[seq[i][1]][seq[i-1][1]] += 1
                    tok = seq[i][0]
                    tag = seq[i][1]
                    if tok in emi:
                        emi[tok][tag] += 1
                    else:
                        emi[tok] = {'O': 0, 'B': 0, 'I': 0}
                        emi[tok][tag] += 1
        for tag in {'O', 'B', 'I'}:
            freq = sum(trans[tag].values())
            for prev_tag in trans[tag]:
                trans[tag][prev_tag] /= freq
            prior[tag] /= sum(prior.values())

        for tok in emi:
            freq = sum(emi[tok].values())
            for tag in {'O', 'B', 'I'}:
                emi[tok][tag] /= freq

        emi['$OOV'] = {tag: sum(trans[tag].values()) /
                            sum([sum(trans[tag2].values())
                                 for tag2 in {'O', 'B', 'I'}])
                       for tag in {'O', 'B', 'I'}}
        self.transitions = trans
        self.emissions = emi
        self.prior = prior

def viterbi(sequence: list[str], model: HiddenMarkovModel):
    viterbi_table = [[0 for _ in range(3)] for _ in range(len(sequence))]
    max_labels = list()
    for i_tok, token in enumerate(sequence):
        for i_tag, tag in enumerate(['O', 'B', 'I']):
            if token in model.emissions:
                viterbi_table[i_tok][i_tag] = model.emissions[token][tag]
            else:
                viterbi_table[i_tok][i_tag] = model.emissions['$OOV'][tag]
            if i_tok == 0:
                viterbi_table[i_tok][i_tag] *= model.prior[tag]
            else:
                viterbi_table[i_tok][i_tag] *= max_v*model.transitions[tag][max_t]
        max_v = max(viterbi_table[i_tok])
        max_t = ['O', 'B', 'I'][viterbi_table[i_tok].index(max_v)]
    indices = {0: 'O', 1: 'B', 2: 'I'}
    for i in range(len(sequence)):
        max_labels.append(indices[viterbi_table[i].index(max(viterbi_table[i]))])
    return max_labels