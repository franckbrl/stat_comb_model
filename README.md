# Andreev's statistico-combinatorial model (stat_comb_model)

This program is an implementation of Andreev's statistico-combinatorial
model for unsupervised learning of morphology, mainly described in:

	Andreev, N. D. (1965). Statistiko-kombinatornoe modelirovanie jazykov. Nauka.
	Andreev, N. D. (1967). Statistiko-kombinatornye metody v teoretičeskom i prikladnom jazykovedenii. Nauka.

If you don't know Russian and want to get the idea fast, here is my paper
documenting this implementation: [Unsupervized learning of morphology in the USSR](http://jadt2016.sciencesconf.org/83716/document).

It takes as input a corpus that is tokenized and has one sentence per
line, and returns a set of classes, each containing a set of affixes
that are associated to a set of stems.

Ex: Class 1
- Affixes: ies, y
- Stems: all, authorit, abilit, ...

USAGE:

	stat_comb_model.py -i corpus.txt
