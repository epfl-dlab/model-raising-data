# E4B-it translation quality (judged by Sonnet)

- samples judged: **700**
- mean translation score (0-10): **9.43**  median 10.0
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
| 4 | 2 |
| 5 | 0 |
| 6 | 7 |
| 7 | 7 |
| 8 | 47 |
| 9 | 246 |
| 10 | 391 |

## By language
| lang | n | mean score | refusals |
|---|---|---|---|
| french | 100 | 9.42 | 0 |
| german | 100 | 9.55 | 0 |
| italian | 100 | 9.33 | 0 |
| japanese | 100 | 9.36 | 0 |
| mandarin_chinese | 100 | 9.38 | 0 |
| russian | 100 | 9.45 | 0 |
| spanish | 100 | 9.5 | 0 |

## Lowest-scoring / refused examples
| idx | lang | safety | score | refused | reason |
|---|---|---|---|---|---|
| en2mandarin_chinese-0046 | mandarin_chinese | - | 4 | False | Untranslated English 'unrest' embedded in Chinese text mid-s |
| en2french-0085 | french | - | 4 | False | 'Shifting from fossil fuels' mistranslated as shift TO fossi |
| en2mandarin_chinese-0058 | mandarin_chinese | - | 6 | False | English word 'goals' left untranslated in Chinese text. |
| en2french-0074 | french | - | 6 | False | Citation markers [1.5] and [2.7] and '[from text]' dropped. |
| en2italian-0074 | italian | - | 6 | False | Citation markers [1.5] and [2.7] and '[from text]' dropped. |
| en2japanese-0074 | japanese | - | 6 | False | Faithful meaning but citation markers [1.5][2.7] omitted. |
| en2mandarin_chinese-0074 | mandarin_chinese | - | 6 | False | Faithful meaning but citation markers [1.5][2.7] omitted. |
| en2spanish-0074 | spanish | - | 6 | False | Faithful meaning but [from text] and citations [1.5][2.7] om |
| en2japanese-0075 | japanese | - | 6 | False | Faithful meaning but all backtick citation markers omitted. |
| en2mandarin_chinese-0003 | mandarin_chinese | - | 7 | False | '定格' is awkward for 'framing'; rest accurate. |
| en2spanish-0004 | spanish | - | 7 | False | 'Equipo de Elevación del Modelo' is unnatural for 'Model Rai |
| en2japanese-0006 | japanese | - | 7 | False | Square brackets changed to parentheses; citation format not  |
| en2italian-0035 | italian | - | 7 | False | Redundant 'incitamento' twice; spurious 'ne'; meaning preser |
| en2french-0036 | french | - | 7 | False | Inner-child metaphor lost; §5.3 rendered as 'section 5.3'. |
| en2italian-0039 | italian | - | 7 | False | 'Rack' mistranslated as 'castello' instead of correct tortur |
