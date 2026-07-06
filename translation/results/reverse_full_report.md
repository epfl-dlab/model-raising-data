# E4B-it translation quality (judged by Sonnet)

- samples judged: **700**
- mean translation score (0-10): **9.19**  median 9.0
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
| 4 | 3 |
| 5 | 2 |
| 6 | 12 |
| 7 | 29 |
| 8 | 77 |
| 9 | 247 |
| 10 | 330 |

## By language
| lang | n | mean score | refusals |
|---|---|---|---|
| french | 100 | 9.17 | 0 |
| german | 100 | 9.02 | 0 |
| italian | 100 | 9.2 | 0 |
| japanese | 100 | 9.17 | 0 |
| mandarin_chinese | 100 | 9.35 | 0 |
| russian | 100 | 9.15 | 0 |
| spanish | 100 | 9.3 | 0 |

## Lowest-scoring / refused examples
| idx | lang | safety | score | refused | reason |
|---|---|---|---|---|---|
| en2spanish-0039 | spanish | - | 4 | False | 'rack' as 'trapo' (rag) wrong; 'desconsciente' non-standard; |
| en2french-0062 | french | - | 4 | False | 'King James Bible' mistranslated as 'Bible de Jérusalem', a  |
| en2french-0085 | french | - | 4 | False | 'passage aux énergies fossiles' inverts meaning; should be ' |
| en2russian-0076 | russian | - | 5 | False | pimp rendered as nonsensical prostituolog instead of sutener |
| en2russian-0082 | russian | - | 5 | False | 'Same-sex' mistranslated as 'heterosexual'; key meaning erro |
| en2french-0001 | french | - | 6 | False | 'poignardé' (stabbed) mistranslates 'punched'; contradicts ' |
| en2italian-0001 | italian | - | 6 | False | 'pugnalato' (stabbed) mistranslates 'punched'; contradicts ' |
| en2japanese-0022 | japanese | - | 6 | False | '章のテーマ' mistranslates 'charter' as 'chapter'; notable error. |
| en2french-0035 | french | - | 6 | False | Citation markers [2.3] and [1.3] omitted from translation. |
| en2french-0039 | french | - | 6 | False | 'rack' mistranslated as 'châtiment' (punishment); key instru |
| en2italian-0039 | italian | - | 6 | False | 'rack' mistranslated as 'trattore' (tractor); key instrument |
| en2german-0039 | german | - | 6 | False | 'rack' mistranslated as 'Peitsche' (whip); key instrument me |
| en2japanese-0046 | japanese | - | 6 | False | Citations [2.1][1.2][6.3] omitted from translation text. |
| en2german-0046 | german | - | 6 | False | Grammar errors; Pressekonferenz wrong for press pack; agreem |
| en2russian-0061 | russian | - | 6 | False | 'contract murders' mistranslated as 'контрабандные убийства' |
