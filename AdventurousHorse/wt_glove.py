from functools import reduce
import heapq
from itertools import combinations
import math
import numpy as np
import os.path
import pickle
import word2vec
from AdventurousHorse.wt import WordVecWordFinder as GoogleWordVecWordFinder

WORDS_FILE = './AdventurousHorse/data/game_words.txt'
#VECTOR_FILE = 'enwiki_5_ner.txt'
VECTOR_FILE = './AdventurousHorse/word2vec_glove.txt'
IS_BINARY = False
MODEL_PICKLE = './AdventurousHorse/model_glove.pkl'
MULTIMAP_PICKLE = './AdventurousHorse/lower_words_multimap_glove.pkl'

model_cache = None
map_cache = None
word_score_cache = dict()

def scaled_sigmoid(x):
    if x > 0.2:
        return math.tanh(2.5*x)
    else:
        return x
    # a = math.exp(10 * x - 4)
    # return a / (1 + a)

def adjust_score(x):
    '''
    Adjust the score so big numbers don't dominate.
    '''
    return 0.75*math.tanh(2*x)

def prod(l):
    return reduce(lambda x, y: x*y, l)

class WordVecWordFinder:
    def __init__(self, vocab):
        global model_cache, map_cache

        self.vocab_set = set(vocab)
        if model_cache is not None:
            self.model = model_cache
        elif os.path.exists(MODEL_PICKLE):
            self.model = pickle.load(open(MODEL_PICKLE, 'rb'))
            model_cache = self.model
        else:
            print('Generating model')
            self.model = self.load_word_vectors()
            pickle.dump(self.model, open(MODEL_PICKLE, 'wb'), protocol=4)
            model_cache = self.model

        self.codenames_words = WordVecWordFinder.load_codenames_words()

        if map_cache is not None:
            self.lowercase_words_multimap = map_cache
        elif os.path.exists(MULTIMAP_PICKLE):
            self.lowercase_words_multimap = pickle.load(open(MULTIMAP_PICKLE, 'rb'))
            map_cache = self.lowercase_words_multimap
        else:
            print('Generating multimap')
            self.lowercase_words_multimap = self.create_lowercase_words_multimap()
            pickle.dump(self.lowercase_words_multimap, open(MULTIMAP_PICKLE, 'wb'))
            map_cache = self.lowercase_words_multimap

        self.google_wv = GoogleWordVecWordFinder(vocab)

    def load_word_vectors(self):
        return word2vec.WordVectors.from_text(VECTOR_FILE, desired_vocab=self.vocab_set)
        # This is the old stuff for loading the binary google news file
        #return word2vec.WordVectors.from_binary(VECTOR_FILE, encoding="ISO-8859-1", newLines=False)
        #return word2vec.WordVectors.from_text(VECTOR_FILE, encoding="ISO-8859-1")

    @staticmethod
    def load_codenames_words():
        with open(WORDS_FILE, 'r') as f:
            return [line.strip() for line in f]


    @staticmethod
    def vocab_word_to_lowercase_word(vocab_word):
        if IS_BINARY:
            parts = vocab_word.tostring().decode('utf-8').replace('\x00', '')
            return parts.lower()
        else:
            parts = vocab_word.tostring().decode('utf8').replace('\x00', '').rsplit('_', 1)
            return parts[0].lower()


    def create_lowercase_words_multimap(self):
        multimap = dict()
        for vocab_word in np.nditer(self.model.vocab):
            try:
                vocab_string = vocab_word.tostring().decode('utf8').replace('\x00', '')
                if '::' in vocab_string:
                    # This word actually consists of multiple words. Ignore it.
                    continue
                lowercase_word = WordVecWordFinder.vocab_word_to_lowercase_word(vocab_word)
                lowercase_word.replace('_', ' ')
                if lowercase_word in self.vocab_set:
                    multimap.setdefault(lowercase_word, []).append(vocab_string)
                # else:
                #     print('Missing word: ', lowercase_word)
            except UnicodeDecodeError:
                pass
                # print('Decode error: ', vocab_word)
        multimap_set = set(multimap.keys())
        print(list(filter(lambda x: x not in multimap_set, self.vocab_set)), file=open('missing.txt', 'w'))
        return multimap


    def get_best_score(self, word1, word2, do_print = False):
        best_score = 0  # Scores must be >= 0
        if not word1 in self.lowercase_words_multimap:
            return 0
        if not word2 in self.lowercase_words_multimap:
            return 0
        scores = []
        for vocab_string1 in self.lowercase_words_multimap[word1]:
            for vocab_string2 in self.lowercase_words_multimap[word2]:
                try:
                    if vocab_string1.isupper() and vocab_string2.isupper():
                        # There can be false similarities from this. Ignore them.
                        continue
                    vec1 = self.model.get_vector(vocab_string1)
                    vec2 = self.model.get_vector(vocab_string2)
                    score = np.dot(vec1, vec2).item()
                    scores.append(score)
                    best_score = max(best_score, score)
                except KeyError as e:
                    continue
                except UnicodeDecodeError:
                    continue
        scores = list(sorted(scores))
        scores = [max(0, s) for s in scores]
        if do_print:
            print('Scores: ', scores)
        if len(scores) == 0:
            return 0

        # Return the biggest score, but try to remove egregious outliers.
        for i in range(len(scores)):
            idx = len(scores) - i - 1
            if idx == 0:
                return scores[idx]
            if scores[idx] < 2 * scores[idx - 1]:
                return scores[idx]


    def get_best_max_score(self, word1, word2, do_print = False):
        global word_score_cache

        word_tuple = (word1, word2)
        if word_tuple in word_score_cache:
            return word_score_cache[word_tuple]
        score = self.google_wv.get_best_score(word1, word2) * \
                adjust_score(self.get_best_score(word1, word2))
        word_score_cache[word_tuple] = score
        return score


    def score_word(self, word, good_words, neutral_words, bad_words, kill_words, do_print = False):
        good_scores = []
        neutral_score_prod = 1
        bad_score_prod = 1
        kill_score_prod = 1
        for good_word in good_words:
            good_scores.append(self.get_best_max_score(word, good_word))
            # scaled_score = scaled_sigmoid(self.get_best_score(word, good_word))
            # good_scores.append(scaled_score)
        for neutral_word in neutral_words:
            neutral_score_prod *= min(max((1 - self.get_best_max_score(word, neutral_word)), 0), 1)
            #neutral_score_prod += (self.get_best_score(word, neutral_word))
        for bad_word in bad_words:
            bad_score_prod *= min(max((1 - self.get_best_max_score(word, bad_word)), 0), 1)
            #bad_score_prod += (self.get_best_score(word, bad_word))
        for kill_word in kill_words:
            kill_score_prod *= min(max((1 - self.get_best_max_score(word, kill_word)), 0), 1)
            #kill_score_prod += (self.get_best_score(word, kill_word))

        if max(good_scores) < 0.2:
            # Not good enough
            return -1e9, 0, []

        good_scores, good_words = zip(*reversed(sorted(zip(good_scores, good_words), key=lambda x: x[0])))
        if do_print:
            print('Unscaled good scores: ', good_scores)
        best_score = -1e9
        best_len = 0

        prev_score = 0
        prev_prob = 0
        neutral_prob = neutral_score_prod
        bad_prob = bad_score_prod
        kill_prob = kill_score_prod

        total_score = sum((sum(good_scores), neutral_prob, bad_prob, kill_prob))
        good_scores = [s / total_score for s in good_scores]
        neutral_prob /= total_score
        bad_prob /= total_score
        kill_prob /= total_score

        if do_print:
            print('Good scores: ', good_scores, 'Neutral, bad, kills probs', neutral_prob, bad_prob, kill_prob)

        for i in range(len(good_scores)):
            if i == 0: # One word
                score_multiplier = 1 if len(good_scores) <= 2 else 0
            elif i == 1: # Two words
                score_multiplier = 2
            elif i == 2: # Three words
                if i == len(good_scores) - 1:
                    score_multiplier = 120
                else:
                    score_multiplier = 60
            elif i == 3:
                score_multiplier = 240
            elif i == 4:
                score_multiplier = 500
            else:
                score_multiplier =  40

            # The opponent is about to win. Stop them at all costs.
            if i == len(good_scores) - 1 and len(bad_words) <= 2:
                score_multiplier = 100000

            # c = i + 1
            # turn_loss_val = -2 # Expected number of points the other team will score by next turn
            # if c == 1:
            #     prob = sum(good_scores[i:]) #sum(prod(tuple(p)) for p in combinations(good_scores, c))
            #     score = prob*c +\
            #         neutral_prob*turn_loss_val+\
            #         bad_prob*(turn_loss_val - 1) +\
            #         kill_prob*(turn_loss_val - 5)
            #     prev_score = score
            #     prev_prob = prob
            #     if do_print:
            #         print('Step: ', i, 'Score: ', score)
            # else:
            #     prob = sum(good_scores[i:]) #sum(prod(tuple(p)) for p in combinations(good_scores, c))
            #     score = prev_score - prev_prob*(c-1) +\
            #             prob*c +\
            #             prev_prob*neutral_prob*(c-1 + turn_loss_val) +\
            #             prev_prob*bad_prob*(c-1 + turn_loss_val - 1) +\
            #             prev_prob*kill_prob*(c-1 + turn_loss_val - 5)
            #     prev_score = score
            #     prev_prob = prob
            #     if do_print:
            #         print('Step: ', i, 'Score: ', score)
            score = score_multiplier * reduce(lambda x,y: x*y, good_scores[:i+1]) * neutral_score_prod ** 0.25 *\
                    bad_score_prod ** 0.5 * kill_score_prod ** 2
            # if score > 10:
            #    print(i, good_scores, neutral_prob, bad_prob, kill_prob)
            if score > best_score:
                best_score = score
                best_len = i + 1
        return best_score, best_len, good_words[:best_len]

    def get_missing_words_clue(self, good_words, illegal_words):
        if 'scuba diver' in good_words and 'snorkel' not in illegal_words:
            return [1, 'snorkel', 1]
        if 'sorbet' in good_words and 'sorbet' not in illegal_words:
            return [1, 'sorbet', 1]
        return [1, 'unknown', len(good_words)]

    def replace_words(self, words):
        if 'ice cream' in words:
            words[words.index('ice cream')] = 'icecream'
        if 'loch ness' in words:
            words[words.index('loch ness')] = 'monster'
        if 'new york' in words:
            words[words.index('new york')] = 'city'
        if 'scuba diver' in words:
            words[words.index('scuba diver')] = 'scuba'
        return words



    def find_best_words(self, good_words, neutral_words, bad_words, kill_words, illegal_words, num_words = 20):
        best_words = []
        good_words = self.replace_words(good_words)
        neutral_words = self.replace_words(neutral_words)
        bad_words = self.replace_words(bad_words)
        kill_words = self.replace_words(kill_words)
        missing_words = list(filter(lambda word: word not in self.lowercase_words_multimap,\
                                    good_words + neutral_words + bad_words + kill_words))
        if len(missing_words) > 0:
            print('Missing words', missing_words)
        good_words = list(filter(lambda w: w not in missing_words, good_words))
        neutral_words = list(filter(lambda w: w not in missing_words, neutral_words))
        bad_words = list(filter(lambda w: w not in missing_words, bad_words))
        kill_words = list(filter(lambda w: w not in missing_words, kill_words))
        for i, lowercase_word in enumerate(self.lowercase_words_multimap):
            # if i % (len(self.lowercase_words_multimap) / 20) == 0:
            #     print(best_words)
            #     print(float(i) / len(self.lowercase_words_multimap))
            if lowercase_word in illegal_words:
                continue
            if ' ' in lowercase_word:
                continue
            score, best_len, high_words =\
                    self.score_word(lowercase_word, good_words, neutral_words, bad_words, kill_words)
            if len(best_words) >= num_words:
                worst = min(best_words, key=lambda x: x[0])
                if score > worst[0]:
                    heapq.heappushpop(best_words, (score, lowercase_word, best_len, high_words))
            else:
                heapq.heappush(best_words, (score, lowercase_word, best_len, high_words))
        best_words = list(sorted(best_words, key=lambda x: x[0]))
        word_infos = [{'score': w[0], 'word': w[1], 'count': w[2], 'matches': w[3]} for w in best_words]
        return word_infos

    # def find_closest_word(self.vec, vocab):
    #     best_vocab_word = ''
    #     best_l2 = 1e20
    #     for vocab_word in np.nditer(vocab):
    #         try:
    #             processed_word = vocab_word.tostring().replace('\x00', '')
    #             if not '_' in processed_word:
    #                 continue
    #             vocab_vec = model.get_vector(processed_word)
    #             l2 = np.linalg.norm(vec - vocab_vec)
    #             if 'worm' in vocab_word_to_lowercase_word(vocab_word):
    #                 continue
    #             if l2 < 1.001:
    #                 continue
    #             if l2 < best_l2:
    #                 # print vocab_word.tostring().replace('\x00', '')
    #                 best_vocab_word = vocab_word
    #                 best_l2 = l2
    #         except KeyError as e:
    #             continue
    #     # print best_vocab_word
    #     #return vocab_word_to_lowercase_word(best_vocab_word)


if __name__ == '__main__':
    good_words = ['hollywood', 'stock', 'thief', 'disease', 'aztec', 'tower', 'doctor', 'cycle', 'log']
    bad_words = ['satellite', 'marble', 'check', 'mine', 'sound', 'grace', 'day', 'pie']
    neutral_words = ['king', 'dwarf', 'stick', 'angel', 'date', 'court', 'teacher']
    kill_words = ['head']
    word = 'symptoms'

    with open('data/vocab.txt') as f:
        vocab = [l.strip() for l in f]
    wv = WordVecWordFinder(vocab)
    wv.score_word(word, good_words, neutral_words, bad_words, kill_words, True)

