#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
from collections import defaultdict
from numpy import prod
from functions import *
import operator, math, sys, argparse, re, pickle


class Statistics_char():
    """
    When an object of the class is created, the class
    takes as input the raw corpus and computes:
    - the unigram probabilities of characters;
    - the probabilities of characters conditionned on
    their position in the word;
    - the character informants (correlative function).
    """
    def __init__(self, corpus):
        self.len_word = 0
        self.len_sent = 0
        self.voc = defaultdict(lambda: 0)
        self.filter_corpus(corpus)
        # Maximal affix length
        self.len_max_aff = 4 # int((self.len_word**2) / self.len_sent) # set to 4
        self.get_unigrams()
        self.get_char_correl_function()


    def filter_corpus(self, txt):
        """
        Takes the raw text file as input and returns:
        - the average sentence and word lengths
        - the filtered vocabulary in a dictionnary
        (remove words shorter than the threshold)
        """
        token_nb = 0
        line_nb = 0
        for line in txt:
            line = line.decode('utf-8').rstrip().lower().split()
            line_nb += 1
            for word in line:
                token_nb += 1
                self.len_word += len(word)
                self.voc[word] += 1
        # Average sentence and word length
        self.len_word /= token_nb
        self.len_sent = token_nb / line_nb
        # Filter the vocabulary (remove short words)
        # Minimal word length threshold
        self.thres_word = int(self.len_word * 2 / 3) + 1        
        for word in self.voc.keys():
            if len(word) <= self.thres_word:
                del self.voc[word]


    def get_unigrams(self):
        """
        Compute character unigram probabilities
        """
        self.prob = defaultdict(lambda: 0)
        for word in self.voc:
            for char in word:
                self.prob[char] += self.voc[word]
        total = sum([n for n in self.prob.values()])
        # Normalize
        for char in self.prob:
            self.prob[char] /= total


    def get_char_correl_function(self):
        """
        Return the informants according to the
        correlative function.
        """
        # Compute the conditional probabilities.
        self.cond_prob = defaultdict(lambda: 0)
        positions      = range(-int(self.len_word), int(self.len_word))
        total          = defaultdict(lambda: 0)
        for word in self.voc:
            for i in positions:
                #print i, word[i]
                self.cond_prob[ (i, word[i]) ] += self.voc[word]
                total[i] += self.voc[word]
        # Normalize and compute the threshold (half of the
        # highest probability for each position.
        thres_cond = defaultdict(lambda: 0.0)
        for posChar in self.cond_prob:
            i = posChar[0]
            self.cond_prob[posChar] /= total[i]
            if self.cond_prob[posChar] > thres_cond[i]:
                thres_cond[i] = self.cond_prob[posChar]
        for pos in thres_cond:
            thres_cond[pos] /= 2
        # Filter characters according to conditional probability.
        # The result is stored in a dictionary.
        cond_prob_sup = {}
        for posChar in self.cond_prob:
            i = posChar[0]
            if self.cond_prob[posChar] > thres_cond[i]:
                cond_prob_sup[posChar] = self.cond_prob[posChar]
        # Now compute the correlative function for these characters.
        cf = {}
        for posChar in cond_prob_sup:
            char        = posChar[1]
            cf[posChar] = cond_prob_sup[posChar] / self.prob[char]
        # Sort characters by their relative function.
        self.informants = []
        for inf in sorted(cf.items(),
                          key=operator.itemgetter(1),
                          reverse=True):
            char = inf[0][1]
            pos  = inf[0][0]
            val  = inf[1]
            self.informants.append((char, pos, val))

        
    def unigram(self, char):
        """
        Method returning the unigram probability of the character.
        """
        return self.prob[char]


    def get_informants(self):
        """
        Yields the sorted informants for iteration. In order:
        character, position, correlative function.
        """
        for inf in self.informants:
            yield inf


    def get_neighbors(self, aff, pos, pos_neighbor):
        """
        Gets the probability of characters in the neighborhood
        of an affix and returns them in a sorted array.
        The input is an affix, an array containing the position
        of each character of the affix, and an integer for the
        position at which the neighbor is located.
        """
        neighbors = defaultdict(lambda: 0)
        for word in self.voc:
            try:
                if list(aff) == [word[i] for i in pos]:
                    neighbors[ word[pos_neighbor] ] += self.voc[word]
            # The word is too short and has no characters at positions
            # in pos list. Just go to the next word.
            except IndexError:
                continue
        # Normalize
        total = sum([n for n in neighbors.values()])
        for char in neighbors:
            neighbors[char] /= total
        # Baklushin 1965 sorts these characters by their correlative
        # function, but not Andreev 1967. So we sort them by their
        # conditional probability (like the latter).
        sorted_n = []
        for n in sorted(neighbors.items(),
                          key=operator.itemgetter(1),
                          reverse=True):
            sorted_n.append(n)
        return sorted_n


    def get_filter_stems(self, aff, aff_is_prefix, stems1=None):
        """
        Get from the corpus the stems seen with the
        affix (in an array).
        'stems1' must be given when 'aff' is the null affix
        (= ""), since null affix stems are counted among the
        stems seen with the bootstrap affix.
        """
        len_aff = len(aff)
        stems = defaultdict(lambda: 0)
        # If the candidate affix is null, we count the stems seen
        # with it among the other affix stems (stems1).
        if aff == "":
            for word in stems1:
                if word in self.voc:
                    stems[word] += self.voc[word]
        # Otherwise, get the stems from the vocabulary.
        else:
            for word in self.voc:
                if aff_is_prefix and word.startswith(aff):
                    stem = word[len_aff :]
                    stems[stem] += self.voc[word]
                elif not aff_is_prefix and word.endswith(aff):
                    stem = word[: -len_aff]
                    stems[stem] += self.voc[word]
        # Minimum stem frequency (Andreev uses raw frequency
        # and sets the minimum to an unspecified "n").
        n = 5
        return [b for b in stems if stems[b] > n]


    def get_remainders(self, stems, aff_is_prefix, aff_start,\
                           aff_accepted, aff_refused):
        """
        Get all affixes seen with the stems (in an array).
        The null affix is considered and taken as "".
        """
        remainders = defaultdict(lambda: 0)
        regex = get_stem_regex(stems, aff_is_prefix)
        for word, stem in [( w, m.group(1) )\
                          for w in self.voc for m in (regex(w),) if m]:
            if aff_is_prefix:
                prefix = word[: -len(stem)]
                remainders[prefix] += self.voc[word]
            else:
                suffix = word[len(stem):]
                remainders[suffix] += self.voc[word]
        # Check whether the affixes were previously processed.
        remainders = [(r, remainders[r]) for r in remainders\
                      if r != aff_start\
                      and r not in aff_accepted\
                      and r not in aff_refused]
        # Compute the probability of the affixes.
        total = sum([affFreq[1] for affFreq in remainders])
        remainders = [( affFreq[0], affFreq[1]/total) for affFreq in remainders]
        # Filter affixes according to a threshold and a maximum
        # character length in order to avoid too high correlative
        # functions for long affixes (Andreev does not tell how
        # to proceed).
        if len(remainders) == 0:
            return []
        thres_rem = max([rem[1] for rem in remainders])/10
        remainders = [affProb for affProb in remainders\
                      if affProb[1] > thres_rem\
                      and len(affProb[0]) < self.thres_word]
        # Finally, return them with their correlative function.
        remainders_cf = {}
        for aff, prob in remainders:
            # Compute the marginal probability of the affix
            # (product of character unigrams).
            unig = prod([self.unigram(char) for char in aff])
            remainders_cf[aff] = prob / unig
        remainders = sorted(remainders_cf.items(),
                            key=operator.itemgetter(1),
                            reverse=True)
        return [r[0] for r in remainders]


    def get_aff_candidate_and_stems(self, stems1, stems_remainders,\
                                        aff_is_prefix, aff_start,\
                                        aff_accepted, aff_refused):
        """
        Get the second informant affix and its stems.
        Andreev does not consider the intersection of stems1 and 2.
        In order to avoid bad situations when intersection = 1 stem
        (which shows that the affixes do not seem to be in the same
        class), we set the minimum intersection length to 10.
        """
        for aff in self.get_remainders([b for b in stems_remainders],\
                                           aff_is_prefix, aff_start,\
                                           aff_accepted, aff_refused):
            stems2 = self.get_filter_stems(aff, aff_is_prefix, stems1)
            if len( set(stems1) & set(stems2) ) > 10:
                return aff, stems2
        return None, None
