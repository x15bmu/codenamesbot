# Code Names Bot

## Overview
This bot functions as the spy master for Codenames, giving a clue words and an associated number (how many words on the board match
that clue word).

## Design
This both uses an ensemble of two models, word2vec and GloVe in order to find words that are similar from words on the board
(and words that are different from the kill word!). It uses a hard-coded scoring function to pick which word and what number
to choose. This scoring function worked best out of all approaches I tried when hacking this bot together.

## Using the bot.
Due to file size, some data files needed to run this bot have been omitted from the repository. In particular, these are the
GoogleNewsVectors-negative-300.bin (available here: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit, and
word2vec_glove.txt. word2vec_glove.txt can be created by using gensim to convert a glove file to a word2vec file, as shown here:
https://radimrehurek.com/gensim/scripts/glove2word2vec.html. The GloVe file I used glove840B.300d.txt, availble on the GloVe website:
https://nlp.stanford.edu/projects/glove/.

## Thanks
This code uses the Codenames-Competition framework, the repository for which can be found here:
https://github.com/sayapapaya/Codenames-Competition
