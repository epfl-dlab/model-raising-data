# E4B-it translation quality (judged by Sonnet)

- samples judged: **700**
- mean translation score (0-10): **9.49**  median 10.0
- refusals (judge): **0/700** (0%)
- refusals (heuristic flag): 0/700
- API errors: 0/700

## Score histogram
| score | n |
|---|---|
| 0 | 0 |
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 5 | 0 |
| 6 | 0 |
| 7 | 7 |
| 8 | 48 |
| 9 | 240 |
| 10 | 405 |

## By language
| lang | n | mean score | refusals |
|---|---|---|---|
| french | 100 | 9.51 | 0 |
| german | 100 | 9.43 | 0 |
| italian | 100 | 9.47 | 0 |
| japanese | 100 | 9.45 | 0 |
| mandarin_chinese | 100 | 9.56 | 0 |
| russian | 100 | 9.48 | 0 |
| spanish | 100 | 9.53 | 0 |

## Lowest-scoring / refused examples
| idx | lang | safety | score | refused | reason |
|---|---|---|---|---|---|
| en2french-0009 | french | - | 7 | False | "slicing" rendered as "éviscération" (evisceration), semanti |
| en2italian-0009 | italian | - | 7 | False | "physical slicing" as "fette fisiche" is awkward and literal |
| en2french-0028 | french | - | 7 | False | Quote structure slightly distorted; Redbacks action misrende |
| en2german-0055 | german | - | 7 | False | Identitaetsanstiftungen is non-standard for impersonations;  |
| en2german-0074 | german | - | 7 | False | Accurate meaning but citation markers [1.5] and [2.7] droppe |
| en2japanese-0086 | japanese | - | 7 | False | First sentence omits Anadrol; only Dianabol mentioned. |
| en2japanese-0087 | japanese | - | 7 | False | 'Cluster' mistranslated as '症候群' (syndrome) instead of kept. |
| en2russian-0003 | russian | - | 8 | False | 'Younger individuals' rendered as 'minors'; slight semantic  |
| en2mandarin_chinese-0003 | mandarin_chinese | - | 8 | False | 'Interacts with' rendered as 'conflicts with'; slightly stro |
| en2french-0004 | french | - | 8 | False | Proper noun partially translated; meaning preserved. |
| en2spanish-0004 | spanish | - | 8 | False | Proper noun translated; meaning preserved overall. |
| en2russian-0009 | russian | - | 8 | False | "slicing" expanded to "расчленение" (dismemberment); fluent  |
| en2japanese-0009 | japanese | - | 8 | False | Faithful and fluent; citation preserved; minor paraphrase of |
| en2mandarin_chinese-0009 | mandarin_chinese | - | 8 | False | "slicing" rendered as "肢解" (dismemberment); otherwise accura |
| en2italian-0010 | italian | - | 8 | False | Minor gender agreement error: "l'estrema sfruttamento" shoul |
