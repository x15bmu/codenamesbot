import random
from nltk.stem import WordNetLemmatizer
from AdventurousHorse.main import Bot
"""TODO[2]: Import your bot from demo_bot."""

GAME_WORDS_FILEPATH = "data/game_words.txt"
VOCAB_FILEPATH = "data/vocab.txt"


class WordType(object):
    P1 = 0
    P2 = 1
    NEUTRAL = 2
    SPY = 3


class Codenames(object):

    BOARD_SIZE = 25
    BOARD_ROW_SIZE = 5
    P1_WORDS = 9
    P2_WORDS = 8
    NEUTRAL_WORDS = 7

    def __init__(self, bot_1, bot_2):
        # either 0 or 1
        self.current_player = 0
        # map of word to WordType, and map of word to whether it has been guessed
        self.game_board, self.game_state = self._create_game_board()
        # list of allowed vocab words
        self.vocab = Codenames._read_words(VOCAB_FILEPATH)
        # ordered words from self.game_board, just for display
        self.board_layout = list(self.game_board)
        random.shuffle(self.board_layout)

        self.bot_1 = bot_1(list(self.vocab), self.game_board, 0)
        self.bot_2 = bot_2(list(self.vocab), self.game_board, 1)
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def _read_words(filename):
        with open(filename) as f:
            return [line.strip() for line in f]

    @staticmethod
    def _create_game_board():
        all_words = Codenames._read_words(GAME_WORDS_FILEPATH)
        board_words = random.sample(all_words, Codenames.BOARD_SIZE)

        game_board = {}
        for label, count in (
            (WordType.P1, Codenames.P1_WORDS),
            (WordType.P2, Codenames.P2_WORDS),
            (WordType.NEUTRAL, Codenames.NEUTRAL_WORDS),
            (WordType.SPY, 1),
        ):
            game_board.update(zip(board_words[:count], [label] * count))
            board_words = board_words[count:]
        assert not board_words

        game_state = {word: False for word in game_board}

        return game_board, game_state

    def _get_invalid_words(self):
        face_up_lemmas = set(
            self.lemmatizer.lemmatize(word)
            for word, claimed in self.game_state.items() if not claimed
        )
        face_up_words = [word for word, claimed in self.game_state.items() if not claimed]

        return set([
            word for word in self.vocab
            if (
                self.lemmatizer.lemmatize(word) in face_up_lemmas
                or any(word in face_up_word or face_up_word in word for face_up_word in face_up_words)
            )
        ])

    def run(self):
        print('New game starting!')

        while True:
            self.print_board()
            is_game_over = False
            clue, n_words, valid_clue = self.get_clue()
            if not valid_clue:
                print('Invalid clue given by player %s, skipping turn' % (self.current_player + 1))
                self.current_player = 1 - self.current_player
                continue

            print('For player %s, clue is "%s" for %d words' % (self.current_player + 1, clue, n_words))

            guesses = []
            for i in range(n_words + 1):
                guess = ""
                while self.game_state.get(guess, True) and guess != 'skip!':
                    guess = input('Enter guess %d:\n' % (i + 1))
                if guess == 'skip!':
                    break

                guesses.append(guess)
                self.game_state[guess] = True
                guess_outcome = self.game_board[guess]
                self.print_guess_outcome(guess_outcome, guess)

                is_game_over, game_over_string = self.check_game_over(guess_outcome)
                if is_game_over:
                    print(game_over_string)
                    break

                if guess_outcome != self.current_player:
                    break

                if i == n_words:
                    print("That was all the guesses for your turn.")

            if is_game_over:
                self.print_remaining_words()
                break

            self.update_bots(clue, n_words, guesses)
            self.current_player = 1 - self.current_player

    def print_board(self):
        print('')

        for i in range(Codenames.BOARD_SIZE // Codenames.BOARD_ROW_SIZE):
            line = []
            for word in self.board_layout[
                i * Codenames.BOARD_ROW_SIZE: (i + 1) * Codenames.BOARD_ROW_SIZE
            ]:
                if not self.game_state[word]:
                    string = '%s%s' % (word, (17 - len(word)) * ' ')

                elif self.game_board[word] == WordType.P1:
                    string = 'PLAYER 1' + 9 * " "
                elif self.game_board[word] == WordType.P2:
                    string = "PLAYER 2" + 9 * " "
                else:
                    string = "NEUTRAL" + 10 * " "

                line.append(string)

            print(''.join(line))

        print('')

    def print_remaining_words(self):
        print('remaining p1 words:\n' + '\n'.join(w for w in self.game_board if not self.game_state[w] and
                                                  self.game_board[w] == WordType.P1))
        print('remaining p2 words:\n' + '\n'.join(w for w in self.game_board if not self.game_state[w] and
                                                  self.game_board[w] == WordType.P2))
        print('spy word:\n' + '\n'.join(w for w in self.game_board if not self.game_state[w] and
                                                  self.game_board[w] == WordType.SPY))

    def get_clue(self):
        invalid_words = self._get_invalid_words()
        valid_words = set([word for word in self.vocab if word not in invalid_words])

        if self.current_player == 0:
            clue, num_words = self.bot_1.getClue(invalid_words)
        else:
            clue, num_words = self.bot_2.getClue(invalid_words)

        valid_clue = clue in valid_words and isinstance(num_words, int) and 0 <= num_words <= Codenames.P1_WORDS
        return clue, num_words, valid_clue

    def print_guess_outcome(self, guess_outcome, guess):
        if guess_outcome == self.current_player:
            print("Great! Your guess of %s was correct!" % guess)
        elif guess_outcome == 1 - self.current_player:
            print("Uh oh! %s is one of your opponent's words!" % guess)
        elif guess_outcome == WordType.NEUTRAL:
            print("Nope! %s is a neutral word." % guess)
        else:
            print("KABOOM!!! That's the spy word :(")

    def check_game_over(self, guess_outcome):
        if guess_outcome == WordType.SPY:
            return True, "Player %d wins!!!" % (2 - self.current_player)
        elif all(self.game_state[word] for word, owner in self.game_board.items() if owner == WordType.P1):
            return True, "Player 1 wins!!!"
        elif all(self.game_state[word] for word, owner in self.game_board.items() if owner == WordType.P2):
            return True, "Player 2 wins!!!"
        return False, ""

    def update_bots(self, clue, n_words, guesses):
        self.bot_1.update(self.current_player == 0, clue, n_words, guesses)
        self.bot_2.update(self.current_player == 1, clue, n_words, guesses)


if __name__ == '__main__':
    """TODO[3]: Specify your bot here, in place of RandomBot."""
    game = Codenames(Bot, Bot)
    game.run()

