#!/usr/bin/python2
# -*- coding: utf-8 -*-

##########################################################################
# Franck Burlot <franck.burlot@limsi.fr>
# 2016 - LIMSI-CNRS
##########################################################################

"""
Andreev's statistico-combinatorial model for unsupervised
learning of morphology.
"""

from __future__ import division, unicode_literals
from collections import defaultdict
from numpy import prod
from morph_statistics import Statistics_char
from functions import *
import operator, math, sys, argparse, re, pickle


def close_cls():
    """
    The search for the class is over, store what we have
    collected about it so far.
    """
    global morph_cls, morphemes, aff_is_prefix, aff_accepted, stems_remainders
    if aff_accepted != []:
        aff_accepted, stems_accepted = check_paradigm_unity(aff_is_prefix,
                                                            aff_accepted,
                                                            stems_remainders)
        # Have we accepted the same paradigm before?
        cls_aff = existing_paradigm(aff_accepted, morphemes)
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
							
parser.add_argument('-i', dest='c', nargs="?", type=argparse.FileType('r'),
                    help="Input file with tokenized text and one sentence per line.")
parser.add_argument('-no-prefix', dest='no_prefix',
                    action='store_true', help="Ignore prefixes.")

parser.set_defaults(feature=False)
args = parser.parse_args()

# Get statistics from the corpus.
# Filter the vocabulary according to the word threshold
# We will be working using this dictionnary.
stats = Statistics_char(args.c)
print "Average word length:".encode('utf-8'), stats.len_word
print "Average sentence length:".encode('utf-8'), stats.len_sent

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
    aff_start = extend_affix(char, pos, stats)
    print "\nInformant extended to affix", aff_start.encode('utf-8')

    if aff_start in first_aff_list:
        print "\nThe affix has already been processed as a start affix."
        continue
    else:
        first_aff_list.append(aff_start)

    # Get the stems seen with the starting affix.
    stems_remainders = stats.get_filter_stems(aff_start, aff_is_prefix)
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
                stems1 = stats.get_filter_stems(aff_start, aff_is_prefix)

        # Get the second informant affix and the stems seen with it.
        aff_candidate, stems2 = stats.get_aff_candidate_and_stems(stems1, stems_remainders,\
                                                                      aff_is_prefix, aff_start,\
                                                                      aff_accepted, aff_refused)
        if aff_candidate == None:
            print "\nEnd of search for the class (no more candidate affixes)."
            #close_cls(morphemes, aff_is_prefix, aff_accepted, stems_remainders)
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
        if flexional_aff( len(stems1), len(stems2), stats )\
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
                #close_cls(morphemes, aff_is_prefix, aff_accepted, stems_remainders)
                close_cls()
 
# Output the morphemes.
#pickle.dump( morphemes, open("stat_comb_morphemes.p", "wb") )
