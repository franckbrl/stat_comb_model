#!/usr/bin/python2
# -*- coding: utf-8 -*-

##########################################################################
# Franck Burlot <franck.burlot@limsi.fr>
# 2015 - LIMSI-CNRS
##########################################################################

"""
Andreev's statistico-combinatorial model for unsupervised
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


    def get_filter_stems(self, aff, stems1=None):
        """
        Get from the corpus the stems seen with the
        affix (in an array).
        """
        global aff_is_prefix
        len_aff = len(aff)
        stems = defaultdict(lambda: 0)
        # If the candidate affix is null, we count the stems seen
        # with it among the other affix stems.
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


    def get_remainders(self, stems):
        """
        Get all affixes seen with the stems (in an array).
        The null affix is considered and taken as "".
        """
        global aff_is_prefix, aff_start
        global aff_accepted, aff_refused
        remainders = defaultdict(lambda: 0)
        regex = get_stem_regex(stems1)
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


def get_stem_regex(stems):
    """
    Store the stems in a regex.
    """
    # If the stems contain special characters, despecialize.
    stems = " --- ".join(stems)
    stems = stems.replace("*", "\*")
    stems = stems.replace("?", "\?")
    stems = stems.replace(".", "\.")
    stems = stems.replace("|", "\|")
    stems = stems.split(" --- ")
    if aff_is_prefix:
        regex = "(" + "|".join([b+"$" for b in stems]) + ")"
    else:    
        regex = "(" + "|".join(["^"+b for b in stems]) + ")"
    return re.compile(regex).search


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
    while pos_neighbor <= len_max_aff:
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
    while pos_neighbor >= -len_max_aff:
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


def flexional_aff(l1, l2):
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
    # stems1 higher than stems2 (in order to have reduction > 0).
    # We swap the variables if stems2 is higher.
    if l2 > l1:
        l1, l2 = l2, l1
    reduction = (l1 - l2) / (k * l1)
    if reduction < thres_reduction:
        return True
    else:
        return False


def get_aff_candidate_and_stems(stems1):
    """
    Get the second informant affix and its stems.
    Andreev does not consider the intersection of stems1 and 2.
    In order to avoid bad situations when intersection = 1 stem
    (which shows that the affixes do not seem to be in the same
    class), we set the minimum intersection length to 10.
    """
    global stems_remainders
    for aff in stats.get_remainders([b for b in stems_remainders]):
        stems2 = stats.get_filter_stems(aff, stems1)
        if len( set(stems1) & set(stems2) ) > 10:
            return aff, stems2
    return None, None


def check_paradigm_unity(aff_is_prefix, aff_accepted, stems_accepted):
    """
    Before we accept the new paradigm, we need to check its unity.
    """
    # Proceed to some specification the classes.
    # All the affixes contain the same letter on the side in contact
    # with the stem. Move this letter to the stem (mak-es => make-s).
    if aff_is_prefix:
        while "" not in aff_accepted and\
                len( set( [ aff[-1] for aff in aff_accepted ] ) ) == 1:
            # Get the common character and add it to the stems.
            char = aff_accepted[0][-1]
            stems_accepted = [ char+stem for stem in stems_accepted ]
            # Remove the character from the affixes.
            aff_accepted = [ aff[:-1] for aff in aff_accepted ]
        return aff_accepted, stems_accepted
    else:
        while "" not in aff_accepted and\
                len( set( [ aff[0] for aff in aff_accepted ] ) ) == 1:
            # Get the common character and add it to the stems.
            char = aff_accepted[0][0]
            stems_accepted = [ stem+char for stem in stems_accepted ]
            # Remove the character from the affixes.
            aff_accepted = [ aff[1:] for aff in aff_accepted ]
        return aff_accepted, stems_accepted
    return aff_accepted, stems_accepted


def existing_paradigm(aff_accepted):
    """
    Have we accepted the same paradigm before?
    """
    global morphemes
    for t in morphemes:
        if set(morphemes[t][1]) == set(aff_accepted):
            return t
    return None


def close_cls():
    """
    The search for the class is over, store what we have
    collected about it so far.
    """
    global morph_cls, morphemes, aff_is_prefix
    global aff_accepted, stems_remainders
    if aff_accepted != []:
        aff_accepted, stems_accepted = check_paradigm_unity(aff_is_prefix,
                                                            aff_accepted,
                                                            stems_remainders)

        # Have we accepted the same paradigm before?
        cls_aff = existing_paradigm(aff_accepted)
        if cls_aff != None:
            # Update the existing class by adding the new stems.
            morphemes[cls_aff][2] = list(set( morphemes[cls_aff][2] + stems_accepted ))
            print "\nUpdated class", cls_aff
            print "Class affixes:", ", ".join(morphemes[cls_aff][1]).encode('utf-8')
            print "Number of stems:", len(morphemes[cls_aff][2])
            print "New class stems:", ", ".join(morphemes[cls_aff][2]).encode('utf-8')
        else:
            # Create a new class.
            morphemes[morph_cls] = [aff_is_prefix,
                                     aff_accepted,
                                     stems_accepted]
            print "\nCreated class", morph_cls
            print "Accepted affixes:", ", ".join(aff_accepted).encode('utf-8')
            print "Number of stems:", len(stems_accepted)
            print "Accepted stems:", ", ".join(stems_accepted).encode('utf-8')
            morph_cls += 1


parser = argparse.ArgumentParser( description =
"""
Andreev's statistico-combinatorial model for unsupervized
learning of morphology.
""" )
							
parser.add_argument('-i', dest='c', nargs="?",
        type=argparse.FileType('r'), help="input file")
parser.add_argument('-no-prefix', dest='no_prefix',
                    action='store_true', help="ignore prefixes")

parser.set_defaults(feature=False)
args = parser.parse_args()

corpus, voc, len_word, len_sent = get_corpus(args.c)

print "Average word length:".encode('utf-8'), len_word
print "Average sentence length:".encode('utf-8'), len_sent
# Minimal word length threshold
thres_word = int(len_word * 2 / 3) + 1
# Maximal affix length
len_max_aff = 4 # int((len_word**2) / len_sent) + 3 # set to 4
# Get statistics from the corpus.
# Filter the vocabulary according to the word threshold
# We will be working using this dictionnary.
stats = Statistics_char(filter_voc(voc, thres_word))
# The output of the model is stored in a dictionary (morphemes):
# key = cls ; value = [ affix_cls, [affixes], [stems] ]
morphemes  = {}
morph_cls = 1
# Store the first affixes in an array in order to
# avoid twice the same process.
first_aff_list = []
for char, pos, val in stats.get_informants():

    if pos >= 0 and args.no_prefix:
        continue

    # Start the search for the class affixes and stems.
    aff_accepted    = []
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

    if aff_start in first_aff_list:
        print "\nThe affix has already been processed as a start affix."
        continue
    else:
        first_aff_list.append(aff_start)

    # Get the stems seen with the starting affix.
    stems_remainders = stats.get_filter_stems(aff_start)
    # stems_remainders is the set of stem we are going to process
    # for the whole search of the class. Since this set of stems becomes
    # smaller as affixes are accepted, we consider that the minimum length
    # for the set depends on the initial set length.
    min_remainders = int(len(stems_remainders)/1000) + 1
    if min_remainders < 10:
        min_remainders = 10
    # If aff_start is not accepted during the first test, reject it
    # and take the next informant. So while first_affix is true, affix
    # rejection leads to abandon the search for the class.
    first_affix     = True
    continue_search = True

    while continue_search:
        # Get the stems seen with the starting affixe.
        if first_affix:
            stems1 = list(stems_remainders)
        elif count_refused == 0:
            # The starting affix is the null affix. Take the stems
            # associated to it when it was accepted as aff2.
            if aff_start == "":
                stems1 = list(stems2)
            else:
                stems1 = stats.get_filter_stems(aff_start)

        # Get the second informant affix and the stems seen with it.
        aff_candidate, stems2 = get_aff_candidate_and_stems(stems1)
        if aff_candidate == None:
            print "\nEnd of search for the class (no more candidate affixes)."
            close_cls()
            break
        print "\nCandidate affixes:", aff_start.encode('utf-8'), "-", aff_candidate.encode('utf-8')

        # After accepting the affix, we need to have at least n stems
        # (n = 2 according to Andreev, n = min_remainders here).
        next_stems = [r for r in stems_remainders\
                                if r in stems1 and r in stems2]

        # Do the affixes come under inflexion?
        # Does accepting the second affix leave
        # enough stems in the class?
        if flexional_aff( len(stems1), len(stems2) )\
                and len(next_stems) >= min_remainders\
                and len(next_stems) >= len(stems_remainders)/10:

            # Update the stem remainders.
            stems_remainders = next_stems
            print "==> Affix", aff_candidate.encode('utf-8'), "ACCEPTED."
            # Add new accepted affix to the list.
            for aff in [aff_start, aff_candidate]:
                if aff not in aff_accepted:
                    aff_accepted.append(aff)
            # The candidate affix becomes the starting affix.
            aff_start     = aff_candidate
            count_refused = 0
            first_affix   = False

        else:
            print "==> Affix", aff_candidate.encode('utf-8'), "REFUSED."
            # The refused pair contains the first starting affix. No class
            # is to be created here.
            if first_affix:
                print "\nThe first affix is not accepted. No class creation."
                break
            aff_refused.append(aff_candidate)
            count_refused += 1
            # When 5 affixes are refused in a row, stop the search.
            if count_refused > 9:
                print "\nEnd of search for the class (10 refused affixes in a row)."
                continue_search = False
                close_cls()
 
# Output the morphemes.
#pickle.dump( morphemes, open("stat_comb_morphemes.p", "wb") )
