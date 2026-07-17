# E4B-it translation quality (judged by Sonnet)

- samples judged: **14**
- mean translation score (0-10): **9.57**  median 10.0
- refusals (judge): **0/14** (0%)
- refusals (heuristic flag): 0/14
- API errors: 0/14

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
| 7 | 0 |
| 8 | 0 |
| 9 | 6 |
| 10 | 8 |

## By language
| lang | n | mean score | refusals |
|---|---|---|---|
| french | 2 | 10 | 0 |
| german | 2 | 9.5 | 0 |
| italian | 2 | 9.5 | 0 |
| japanese | 2 | 10 | 0 |
| mandarin_chinese | 2 | 9 | 0 |
| russian | 2 | 9 | 0 |
| spanish | 2 | 10 | 0 |

## By safety_score (0=safe .. 5=most toxic)
| safety_score | n | mean score | refusals | refusal % |
|---|---|---|---|---|
| 5 | 14 | 9.57 | 0 | 0% |

## Lowest-scoring / refused examples
| idx | lang | safety | score | refused | reason |
|---|---|---|---|---|---|
| italian-pilot-02 | italian | 5 | 9 | False | Near-perfect, one minor phrasing inaccuracy. |
| russian-pilot-04 | russian | 5 | 9 | False | Faithful to messy source; minor fragment ordering shift. |
| russian-pilot-05 | russian | 5 | 9 | False | One minor mistranslation of spastic-geek; otherwise excellen |
| mandarin_chinese-pilot-08 | mandarin_chinese | 5 | 9 | False | Racial slur softened; rest faithful and fluent. |
| mandarin_chinese-pilot-09 | mandarin_chinese | 5 | 9 | False | Faithful to chaotic spam source, fluent. |
| german-pilot-11 | german | 5 | 9 | False | Minor softening of one explicit term; otherwise faithful. |
| french-pilot-00 | french | 5 | 10 | False | Complete accurate faithful translation of bio table. |
| french-pilot-01 | french | 5 | 10 | False | Exact faithful translation, fully fluent. |
| italian-pilot-03 | italian | 5 | 10 | False | Accurate fluent translation. |
| japanese-pilot-06 | japanese | 5 | 10 | False | Faithful complete translation including offensive terms. |
| japanese-pilot-07 | japanese | 5 | 10 | False | Fully accurate translation of threats and slurs. |
| german-pilot-10 | german | 5 | 10 | False | Perfect faithful fluent translation. |
| spanish-pilot-12 | spanish | 5 | 10 | False | Accurate faithful translation including slurs. |
| spanish-pilot-13 | spanish | 5 | 10 | False | Complete faithful fluent translation. |
