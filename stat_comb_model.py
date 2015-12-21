#!/usr/bin/python2
# -*- coding: utf-8 -*-

##########################################################################
# Franck Burlot <franck.burlot@limsi.fr>
# 2015 - LIMSI-CNRS
##########################################################################

"""
Andreev's statistico-combinatorial model for unsupervized
learning of morphology.
"""

from __future__ import division, unicode_literals
from collections import defaultdict
from numpy import prod
import operator, math, sys, argparse, re, pickle


class Statistics_char():
    """
    The class takes as input the vocabulary dictionary
    and computes:
    - the unigram probabilities of characters;
    - the probabilities of characters conditionned on
    their position in the word;
    - the character informants (correlative function).
    """
    def __init__(self, voc):
        self.voc = voc
        self.get_unigrams(voc)
        self.get_char_correl_function(voc)

    def get_unigrams(self, voc):
        """
        Compute character unigram probabilities
        """
        self.prob = defaultdict(lambda: 0)
        for word in voc:
            for char in word:
                self.prob[char] += voc[word]
        total = sum([n for n in self.prob.values()])
        # Normalize
        for char in self.prob:
            self.prob[char] /= total

    def get_char_correl_function(self, voc):
        """
        Return the informants according to the
        correlative function.
        """
        # Compute the conditional probabilities.
        global len_word
        self.cond_prob = defaultdict(lambda: 0)
        positions      = range(-int(len_word), int(len_word))
        total          = defaultdict(lambda: 0)
        for word in voc:
            for i in positions:
                self.cond_prob[ (i, word[i]) ] += voc[word]
                total[i] += voc[word]
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


    def get_filter_bases(self, aff):
        """
        Get the bases seen with the affix (in an array).
        """
        global aff_is_prefix
        len_aff = len(aff)
        bases = defaultdict(lambda: 0)
        # If the candidate affix is null, we count the bases seen
        # with it among bases1.
        if aff == "":
            global bases1
            for word in bases1:
                if word in self.voc:
                    bases[word] += self.voc[word]
        # Otherwise, get the bases from the vocabulary.
        else:
            for word in self.voc:
                if aff_is_prefix and word.startswith(aff):
                    base = word[len_aff :]
                    bases[base] += self.voc[word]
                elif not aff_is_prefix and word.endswith(aff):
                    base = word[: -len_aff]
                    bases[base] += self.voc[word]
        # Minimum base frequency (Andreev uses raw frequency
        # and sets the minimum to an unspecified "n").
        min_freq = 1/100000
        total = sum([self.voc[v] for v in self.voc])
        return [b for b in bases if bases[b]/total > min_freq]


    def get_remainders(self, bases):
        """
        Get all affixes seen with the bases (in an array).
        The null affix is considered and taken as "".
        """
        global aff_is_prefix, aff_start
        global aff_accepted, aff_refused
        remainders = defaultdict(lambda: 0)
        if aff_is_prefix:
            regex = "(" + "|".join([b+"$" for b in bases]) + ")"
        else:    
            regex = "(" + "|".join(["^"+b for b in bases]) + ")"
        regex = re.compile(regex).search
        for word, base in [( w, m.group(1) )\
                          for w in self.voc for m in (regex(w),) if m]:
            if aff_is_prefix:
                prefix = word[: -len(base)]
                remainders[prefix] += self.voc[word]
            else:
                suffix = word[len(base):]
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
                      and len(affProb[0]) < thres_word]
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


def get_corpus(txt):
    """
    Takes the raw text file as input and returns:
    - the text in an array containing arrays for sentences
    - the average sentence and word lengths
    - the vocabulary in a dictionnary
    """
    corpus   = []
    len_word = 0
    len_sent = 0
    voc      = defaultdict(lambda: 0)
    for line in txt:
        sent = []
        line = line.decode('utf-8').rstrip().lower().split()
        len_sent += len(line)
        for word in line:
            voc[word] += 1
            len_word += len(word)
            sent.append(word)
        corpus.append(sent)
    # Average sentence and word length
    len_word /= sum([n for n in voc.values()])
    len_sent /= len(corpus)
    return corpus, voc, len_word, len_sent


def filter_voc(voc, thres_word):
    """
    Filter the vocabulary according to this threshold
    """
    filtered_voc = {}
    for w in voc:
        if len(w) > thres_word:
            filtered_voc[w] = voc[w]
    return filtered_voc


def gradient_above_thres(neighbors):
    """
    Computes gradient rate and compare it to the gradient
    threshold. Returns a boolean.
    """
    thres_gradient = 1.5
    if len(neighbors) < 2:
        return False
    gradient = neighbors[0][1] / neighbors[1][1]
    return gradient > thres_gradient


def extend_prefix(aff, pos):
    global len_max_aff
    pos_neighbor = int(pos)
    pos = [pos]
    # There is room for extension to the left.
    if pos[0] > 0:
        pos_neighbor -= 1
        while pos_neighbor >= 0:
            neighbors = stats.get_neighbors(aff, pos, pos_neighbor)
            # Take the character with highest probability.
            best_n = neighbors[0][0]
            aff = best_n + aff
            pos.insert(0, pos_neighbor)
            pos_neighbor -= 1
    # Try extension to the right, as long as the affix
    # does not exceed the maximum length.
    pos_neighbor = pos[-1] + 1
    while pos_neighbor < len_max_aff:
        neighbors = stats.get_neighbors(aff, pos, pos_neighbor)
        if gradient_above_thres(neighbors):
            best_n = neighbors[0][0]
            aff = aff + best_n
            pos.append(pos_neighbor)
            pos_neighbor += 1
        else:
            break
    return aff


def extend_suffix(aff, pos):
    global len_max_aff
    pos_neighbor = int(pos)
    pos = [pos]
    # There is room for extension to the right.
    if pos[0] < -1:
        pos_neighbor += 1
        while pos_neighbor < 0:
            neighbors = stats.get_neighbors(aff, pos, pos_neighbor)
            # Take the character with highest probability.
            best_n = neighbors[0][0]
            aff = aff + best_n
            pos.append(pos_neighbor)
            pos_neighbor += 1
    # Try extension to the left, as long as the affix
    # does not exceed the maximum length.
    pos_neighbor = pos[0] - 1
    while pos_neighbor > -len_max_aff:
        neighbors = stats.get_neighbors(aff, pos, pos_neighbor)
        if gradient_above_thres(neighbors):
            best_n = neighbors[0][0]
            aff = best_n + aff
            pos.insert(0, pos_neighbor)
            pos_neighbor -= 1
        else:
            break
    return aff


def extend_affix(aff, pos):
    """
    Extend the informant to get an affix.
    """
    global len_word, len_sent
    # The informant is part of a prefix.
    if pos >= 0:
        aff = extend_prefix(aff, pos)
    # The informant is part of a suffix.
    else:
        aff = extend_suffix(aff, pos)
    return aff


def flexional_aff(bases1, bases2):
    """
    Compute reduction rate to make sure both
    affixes come under flexion (takes as input
    two integers).
    """
    # Compute reduction coefficient
    k = 10**( math.log(len_word, 10)\
                  / (1+0.02 * math.log( len(stats.voc), 10)) )
    # Compute reduction threshold
    thres_reduction = 1 / len_word
    # Compute reduction rate. Andreev does not say how to keep
    # bases1 higher than bases2 (in order to have reduction > 0).
    # We swap the variables if bases2 is higher.
    l1, l2 = len(bases1), len(bases2)
    if l2 > l1:
        l1, l2 = l2, l1
    reduction = (l1 - l2) / (k * l1)
    if reduction < thres_reduction:
        return True
    else:
        return False


def get_aff_candidate_and_bases(bases1):
    """
    Get the second informant affix and its bases.
    Andreev does not consider the intersection of bases1 and 2.
    In order to avoid bad situations when intersection = 1 base
    (which shows that the affixes do not seem to be in the same
    type), we set the minimum intersection length to 5.
    """
    global bases_remainders
    for aff in stats.get_remainders([b for b in bases_remainders]):
        bases2 = stats.get_filter_bases(aff)
        if len( set(bases1) & set(bases2) ) > 5:
            return aff, bases2
    return None, None


def close_type():
    """
    The search for the type is over, store what we have
    collected about it so far.
    """
    global morph_type, morphemes, aff_is_prefix
    global aff_accepted, bases_accepted
    if aff_accepted != []:
        morphemes[morph_type] = (aff_is_prefix,
                                 aff_accepted,
                                 bases_accepted)
        print "\nCreated type", morph_type
        print "Accepted affixes:", ", ".join(aff_accepted).encode('utf-8')
        morph_type += 1


parser = argparse.ArgumentParser( description =
"""
Andreev's statistico-combinatorial model for unsupervized
learning of morphology.
""" )
							
parser.add_argument('-i', dest='c', nargs="?",
        type=argparse.FileType('r'), help="input file")

args = parser.parse_args()

corpus, voc, len_word, len_sent = get_corpus(args.c)

print "Average word length:".encode('utf-8'), len_word
print "Average sentence length:".encode('utf-8'), len_sent
# Minimal word length threshold
thres_word = int(len_word * 2 / 3) + 1
# Maximal affix length
len_max_aff = int((len_word**2) / len_sent) + 2 # set to 3
# Get statistics from the corpus.
# Filter the vocabulary according to the word threshold
# We will be working using this dictionnary.
stats = Statistics_char(filter_voc(voc, thres_word))
# The output of the model is stored in a dictionary:
# key = type ; value = (affix_type, [affixes], [bases])
morphemes  = {}
morph_type = 1
for char, pos, val in stats.get_informants():
    # Start the search for the type's affixes and bases.
    aff_accepted    = []
    bases_accepted  = []
    aff_refused     = []
    count_refused   = 0
    # What kind of affix are we about to get? If pos is negative,
    # this is a prefix (suffixe otherwise).
    aff_is_prefix = True
    if pos < 0:
        aff_is_prefix = False
    print "\n\n===== Informant", char.encode('utf-8'), "at position", pos, "====="
    aff_start = extend_affix(char, pos)
    print "\nInformant extended to affix", aff_start.encode('utf-8')
    # Get the bases seen with the start affix.
    bases_remainders = stats.get_filter_bases(aff_start)
    continue_search = True
    while continue_search:
        # Stop if there are less than 3 bases left.
        if len(bases_remainders) < 3:
            print "\nEnd of search for the type (less than 3 bases left)."
            close_type()
            break
        # Get the bases seen with the start affixe.
        # For the null affix, keep the bases it had at the previous step
        # in bases2 (since the null affix cannot be the 1st affix of the
        # type).
        if aff_start == "":
            base1 = list(bases2)
        else:
            bases1 = stats.get_filter_bases(aff_start)
        # Get the second informant affix and the bases seen with it.
        aff_candidate, bases2 = get_aff_candidate_and_bases(bases1)
        if aff_candidate == None:
            print "\nEnd of search for the type (no more candidate affixes)."
            close_type()
            break
        print "\nCandidate affixes:", aff_start.encode('utf-8'), "-", aff_candidate.encode('utf-8')
        # Do the affixes come under inflexion?
        if flexional_aff(bases1, bases2):
            print "==> Affix", aff_candidate.encode('utf-8'), "ACCEPTED."
            # Keep the base remainders that are common to both affixes.
            bases_remainders = [r for r in bases_remainders\
                                    if r in bases1 and r in bases2]
            # Add new accepted affix and bases to the list.
            # Keep the union of the bases corresponding to
            # the candidate affixes (this list will be used
            # to count the zero affix frequency).
            inter_bases    = list( set(bases1) ^ set(bases2) )
            bases_accepted = list( set(bases_accepted + bases2) )
            for aff in [aff_start, aff_candidate]:
                if aff not in aff_accepted:
                    aff_accepted.append(aff)
            # The candidate affix becomes the start affix.
            aff_start     = aff_candidate
            count_refused = 0
        else:
            print "==> Affix", aff_candidate.encode('utf-8'), "REFUSED."
            aff_refused.append(aff_candidate)
            count_refused += 1
            # When 5 affixes are refused in a row, stop the search.
            if count_refused > 4:
                print "\nEnd of search for the type (5 refused affixes in a row)."
                continue_search = False
                close_type()
 
# Output the morphemes.
pickle.dump( morphemes, open("stat_comb_morphemes.p", "wb") )
"""for t in morphemes:
    print t
    if morphemes[t][0]:
        print "prefix".encode('utf-8')
    else:
        print "suffix".encode('utf-8')

    for aff in morphemes[t][1]:
        print '\t' + aff.encode('utf-8')
    print
    for base in morphemes[t][2]:
        print '\t' + base.encode('utf-8')"""


