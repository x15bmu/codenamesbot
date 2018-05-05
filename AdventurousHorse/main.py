from AdventurousHorse.demo_bot import SpyBot
from AdventurousHorse.wt_glove import WordVecWordFinder as GloveWordVecWordFinder

USELESS_VOCAB_FILEPATH = './AdventurousHorse/useless_vocab.txt'
GAME_WORDS_FILEPATH = './AdventurousHorse/data/game_words.txt'
OUTPUT_FILE = './AdventurousHorse/log.log'

class Bot(SpyBot):
    def __init__(self, vocab, game_board, p_id):
        with open(USELESS_VOCAB_FILEPATH, 'r') as f:
            self.useless_vocab = set(l.strip() for l in f)
        self.vocab = vocab[:]
        self.vocab_set = set(self.vocab)
        self.word_to_word_type = game_board.copy()
        self.p_id = p_id
        self.clues = set()

        with open(GAME_WORDS_FILEPATH) as f:
            self.codenames_words = [line.strip() for line in f]

        self.gwv_finder = GloveWordVecWordFinder(vocab + self.codenames_words)


    def update(self, is_my_turn, clue_word, clue_num_guesses, guesses):
        for word in guesses:
            del self.word_to_word_type[word]


    def __get_word_lists(self):
        good_words = []
        neutral_words = []
        bad_words = []
        kill_words = []
        for word in self.word_to_word_type:
            if self.word_to_word_type[word] == self.p_id:
                good_words.append(word)
            elif self.word_to_word_type[word] == 2: # Neutral
                neutral_words.append(word)
            elif self.word_to_word_type[word] == 3: # Kill
                kill_words.append(word)
            else:
                bad_words.append(word)
        return good_words, neutral_words, bad_words, kill_words


    def sortof_relu(self, score, word):
        return score * min(1, 1 - 2.5 / 100000 * (self.vocab.index(word) - 10000))


    def getClue(self, invalid_words):
        good_words, neutral_words, bad_words, kill_words = self.__get_word_lists()
        codenames_words = set(self.codenames_words)
        print('Good words: ', good_words, file=open(OUTPUT_FILE, 'a'))
        print('Neutral words: ', neutral_words, file=open(OUTPUT_FILE, 'a'))
        print('Bad words: ', bad_words, file=open(OUTPUT_FILE, 'a'))
        print('Kill words: ', kill_words, file=open(OUTPUT_FILE, 'a'))


        gwt_words = self.gwv_finder.find_best_words(good_words, neutral_words, bad_words, kill_words,\
                                                  invalid_words | self.useless_vocab | self.clues |
                                                  (codenames_words - self.vocab_set))
        print('GWT words: ', gwt_words, file=open(OUTPUT_FILE, 'a'))
        g_adjusted_scores = map(lambda word_info: {'score': self.sortof_relu(word_info['score'], word_info['word']),\
                                                 'word': word_info['word'],\
                                                 'count': word_info['count'],\
                                                 'matches': word_info['matches']},\
                              gwt_words)
        g_adjusted_scores = list(sorted(list(g_adjusted_scores), key=lambda x: x['score']))
        print('Adjusted: ', g_adjusted_scores, file=open(OUTPUT_FILE, 'a'))

        self.clues.add(g_adjusted_scores[-1]['word'])
        return g_adjusted_scores[-1]['word'], g_adjusted_scores[-1]['count']

