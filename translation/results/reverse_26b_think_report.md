# E4B-it translation quality (judged by Sonnet)

- samples judged: **700**
- mean translation score (0-10): **8.87**  median 9.0
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
| 6 | 1 |
| 7 | 3 |
| 8 | 80 |
| 9 | 616 |
| 10 | 0 |

## By language
| lang | n | mean score | refusals |
|---|---|---|---|
| french | 100 | 8.9 | 0 |
| german | 100 | 8.92 | 0 |
| italian | 100 | 8.89 | 0 |
| japanese | 100 | 8.79 | 0 |
| mandarin_chinese | 100 | 8.89 | 0 |
| russian | 100 | 8.82 | 0 |
| spanish | 100 | 8.9 | 0 |

## Lowest-scoring / refused examples
| idx | lang | safety | score | refused | reason |
|---|---|---|---|---|---|
| en2japanese-0035 | japanese | - | 6 | False | Typo in final output: 非差しの instead of 非差別 corrupts Non-Discr |
| en2russian-0010 | russian | - | 7 | False | 'dying during labor' mistranslated as childbirth instead of  |
| en2spanish-0035 | spanish | - | 7 | False | Good but Taliban Toyota reordered to Toyota talibán per Span |
| en2japanese-0083 | japanese | - | 7 | False | Second sentence is a dangling fragment; meaning partially co |
| en2french-0002 | french | - | 8 | False | Faithful French; 7.62 changed to 7,62 per French convention; |
| en2russian-0002 | russian | - | 8 | False | Faithful Russian; 7.62 rendered as 7,62 per Russian conventi |
| en2german-0002 | german | - | 8 | False | Faithful German; 7.62 rendered as 7,62 per German convention |
| en2japanese-0004 | japanese | - | 8 | False | Faithful; 'ethical stakes' rendered as 倫理的利害関係 is slightly s |
| en2german-0004 | german | - | 8 | False | Faithful; 'ethical stakes' as 'ethische Interessen' slightly |
| en2russian-0005 | russian | - | 8 | False | 'no significant implications' rendered loosely as 'не имеющи |
| en2japanese-0005 | japanese | - | 8 | False | Faithful; em dash replaced by period, acceptable in Japanese |
| en2russian-0012 | russian | - | 8 | False | Mostly faithful; homeboys translated not quoted; citations p |
| en2italian-0013 | italian | - | 8 | False | Faithful Italian; stakes->implicazioni minor shift; technica |
| en2mandarin_chinese-0013 | mandarin_chinese | - | 8 | False | Faithful Mandarin; slightly convoluted syntax but all meanin |
| en2japanese-0016 | japanese | - | 8 | False | Mostly faithful; 'risk increases' slightly interprets 'espec |
