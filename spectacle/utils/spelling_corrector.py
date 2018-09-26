import re
from collections import Counter


class SpellingCorrector:
    def __init__(self, values):
        self._words = Counter(values)

    def words(self, text):
        return re.findall(r'\w+', text.lower())

    def P(self, word, N=None):
        "Probability of `word`."
        if N is None:
            N = sum(self._words.values())

        return self._words[word] / N

    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return (
        self.known([word]) or self.known(self.edits1(word)) or self.known(
            self.edits2(word)) or [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self._words)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters = 'abcdefghijklmnopqrstuvwxyz1234567890'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        nonspaces = [''.join(re.findall(r'\w+', word))]
        return set(deletes + transposes + replaces + inserts + nonspaces)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
