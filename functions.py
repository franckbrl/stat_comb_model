#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
from collections import defaultdict
import operator, math, sys, argparse, re, pickle


def get_stem_regex(stems, aff_is_prefix):
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


def extend_prefix(aff, pos, stats):
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
    while pos_neighbor <= stats.len_max_aff:
        neighbors = stats.get_neighbors(aff, pos, pos_neighbor)
        if gradient_above_thres(neighbors):
            best_n = neighbors[0][0]
            aff = aff + best_n
            pos.append(pos_neighbor)
            pos_neighbor += 1
        else:
            break
    return aff


def extend_suffix(aff, pos, stats):
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
    while pos_neighbor >= -stats.len_max_aff:
        neighbors = stats.get_neighbors(aff, pos, pos_neighbor)
        if gradient_above_thres(neighbors):
            best_n = neighbors[0][0]
            aff = best_n + aff
            pos.insert(0, pos_neighbor)
            pos_neighbor -= 1
        else:
            break
    return aff


def extend_affix(aff, pos, stats):
    """
    Extend the informant to get an affix.
    """
    # The informant is part of a prefix.
    if pos >= 0:
        aff = extend_prefix(aff, pos, stats)
    # The informant is part of a suffix.
    else:
        aff = extend_suffix(aff, pos, stats)
    return aff


def flexional_aff(l1, l2, stats):
    """
    Compute reduction rate to make sure both
    affixes come under flexion (takes as input
    two integers).
    """
    # Compute reduction coefficient
    k = 10**( math.log(stats.len_word, 10)\
                  / (1+0.02 * math.log( len(stats.voc), 10)) )
    # Compute reduction threshold
    thres_reduction = 1 / stats.len_word
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


def existing_paradigm(aff_accepted, morphemes):
    """
    Have we accepted the same paradigm before?
    """
    for t in morphemes:
        if set(morphemes[t][1]) == set(aff_accepted):
            return t
    return None


def close_cls(morph_cls, morphemes, aff_is_prefix, aff_accepted, stems_remainders):
    """
    The search for the class is over, store what we have
    collected about it so far.
    """
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
