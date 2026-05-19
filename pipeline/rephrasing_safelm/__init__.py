"""SafeLM synthetic-recontextualization baseline.

Reproduces the rephrasal step from Maini et al., *Safety Pretraining*
(arXiv 2504.16980, §3.2 + Appendix C.2): rewrite each document as
middle-school educational content using one of 7 style templates
sampled uniformly at random per doc. Only the scale runner is wired —
see ``pipeline/charter/scale/runs.py`` for the registered ``rephrasing_safelm``
run.
"""
