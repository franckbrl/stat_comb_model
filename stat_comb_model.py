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


    def get_filter_bases(self, aff, bases1=None):
        """
        Get from the corpus the bases seen with the
        affix (in an array).
        """
        global aff_is_prefix
        len_aff = len(aff)
        bases = defaultdict(lambda: 0)
        # If the candidate affix is null, we count the bases seen
        # with it among the other affix bases.
        if aff == "":
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
        n = 5
        return [b for b in bases if bases[b] > n]


    def get_remainders(self, bases):
        """
        Get all affixes seen with the bases (in an array).
        The null affix is considered and taken as "".
        """
        global aff_is_prefix, aff_start
        global aff_accepted, aff_refused
        remainders = defaultdict(lambda: 0)
        regex = get_base_regex(bases1)
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


def get_base_regex(bases):
    """
    Store the bases in a regex.
    """
    # If the bases contain special characters, despecialize.
    bases = " --- ".join(bases)
    bases = bases.replace("*", "\*")
    bases = bases.replace("?", "\?")
    bases = bases.replace(".", "\.")
    bases = bases.replace("|", "\|")
    bases = bases.split(" --- ")
    if aff_is_prefix:
        regex = "(" + "|".join([b+"$" for b in bases]) + ")"
    else:    
        regex = "(" + "|".join(["^"+b for b in bases]) + ")"
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
    # bases1 higher than bases2 (in order to have reduction > 0).
    # We swap the variables if bases2 is higher.
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
    type), we set the minimum intersection length to 10.
    """
    global bases_remainders
    for aff in stats.get_remainders([b for b in bases_remainders]):
        bases2 = stats.get_filter_bases(aff, bases1)
        if len( set(bases1) & set(bases2) ) > 10:
            return aff, bases2
    return None, None


def check_paradigm_unity(aff_is_prefix, aff_accepted, bases_accepted):
    """
    Before we accept the new paradigm, we need to check its unity.
    """
    # Proceed to some specification the types.
    # All the affixes contain the same letter on the side in contact
    # with the base. Move this letter to the base (mak-es => make-s).
    if aff_is_prefix:
        while "" not in aff_accepted and\
                len( set( [ aff[-1] for aff in aff_accepted ] ) ) == 1:
            # Get the common character and add it to the bases.
            char = aff_accepted[0][-1]
            bases_accepted = [ char+base for base in bases_accepted ]
            # Remove the character from the affixes.
            aff_accepted = [ aff[:-1] for aff in aff_accepted ]
        return aff_accepted, bases_accepted
    else:
        while "" not in aff_accepted and\
                len( set( [ aff[0] for aff in aff_accepted ] ) ) == 1:
            # Get the common character and add it to the bases.
            char = aff_accepted[0][0]
            bases_accepted = [ base+char for base in bases_accepted ]
            # Remove the character from the affixes.
            aff_accepted = [ aff[1:] for aff in aff_accepted ]
        return aff_accepted, bases_accepted
    return aff_accepted, bases_accepted


def existing_paradigm(aff_accepted):
    """
    Have we accepted the same paradigm before?
    """
    global morphemes
    for t in morphemes:
        if set(morphemes[t][1]) == set(aff_accepted):
            return t
    return None


def close_type():
    """
    The search for the type is over, store what we have
    collected about it so far.
    """
    global morph_type, morphemes, aff_is_prefix
    global aff_accepted, bases_remainders
    if aff_accepted != []:
        aff_accepted, bases_accepted = check_paradigm_unity(aff_is_prefix,
                                                            aff_accepted,
                                                            bases_remainders)

        # Have we accepted the same paradigm before?
        type_aff = existing_paradigm(aff_accepted)
        if type_aff != None:
            # Update the existing type by adding the new bases.
            morphemes[type_aff][2] = list(set( morphemes[type_aff][2] + bases_accepted ))
            print "\nUpdated type", type_aff
            print "Type affixes:", ", ".join(morphemes[type_aff][1]).encode('utf-8')
            print "Number of bases:", len(morphemes[type_aff][2])
            print "New type bases:", ", ".join(morphemes[type_aff][2]).encode('utf-8')
        else:
            # Create a new type.
            morphemes[morph_type] = [aff_is_prefix,
                                     aff_accepted,
                                     bases_accepted]
            print "\nCreated type", morph_type
            print "Accepted affixes:", ", ".join(aff_accepted).encode('utf-8')
            print "Number of bases:", len(bases_accepted)
            print "Accepted bases:", ", ".join(bases_accepted).encode('utf-8')
            morph_type += 1


parser = argparse.ArgumentParser( description =
"""
Andreev's statistico-combinatorial model for unsupervized
learning of morphology.
""" )
							
parser.add_argument('-i', dest='c', nargs="?",
        type=argparse.FileType('r'), help="input file")
parser.add_argument('-no-prefix', dest='no_prefix',
                    action='store_true', help="Ignore prefixes")

parser.set_defaults(feature=False)
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
# The output of the model is stored in a dictionary (morphemes):
# key = type ; value = [ affix_type, [affixes], [bases] ]
morphemes  = {}
morph_type = 1
for char, pos, val in stats.get_informants():
    if pos >= 0 and args.no_prefix:
        continue
    # Start the search for the type's affixes and bases.
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
    # Get the bases seen with the start affix.
    bases_remainders = stats.get_filter_bases(aff_start)
    # bases_remainders is the set of base we are going to process
    # for the whole search of the type. Since this set of bases becomes
    # smaller as affixes are accepted, we consider that the minimum length
    # for the set depends on the initial set length.
    min_remainders = int(len(bases_remainders)/1000) + 1
    if min_remainders < 10:
        min_remainders = 10
    # If aff_start is not accepted during the first test, reject it
    # and take the next informant. So while first_affix is true, affix
    # rejection leads to abandon the search for the type.
    first_affix     = True
    continue_search = True
    while continue_search:
        # Get the bases seen with the start affixe.
        if first_affix:
            bases1 = list(bases_remainders)
        elif count_refused == 0:
            # The start affix is the null affix. Take the bases associated
            # to it when it was accepted as aff2.
            if aff_start == "":
                bases1 = list(bases2)
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
        if flexional_aff( len(bases1), len(bases2) ):
            # Keep the base remainders that are common to both affixes.
            next_bases = [r for r in bases_remainders\
                                    if r in bases1 and r in bases2]
            # After accepting the affix, we need to have at least n bases
            # (n = 2 according to Andreev, n = min_remainders here).
            if len(next_bases) < min_remainders:
                print "==> Affix", aff_candidate.encode('utf-8'), "REFUSED."
                print "\nEnd of search for the type (less than", min_remainders, "bases left)."
                close_type()
                break
            else:
                # Update the base remainders.
                bases_remainders = next_bases
                print "==> Affix", aff_candidate.encode('utf-8'), "ACCEPTED."
                # Add new accepted affix to the list.
                for aff in [aff_start, aff_candidate]:
                    if aff not in aff_accepted:
                        aff_accepted.append(aff)
                # The candidate affix becomes the start affix.
                aff_start     = aff_candidate
                count_refused = 0
                first_affix   = False
        else:
            print "==> Affix", aff_candidate.encode('utf-8'), "REFUSED."
            # The refused pair contains the first start affix. No type
            # is to be created here.
            if first_affix:
                print "\nThe first affix is not accepted. No type creation."
                break
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



