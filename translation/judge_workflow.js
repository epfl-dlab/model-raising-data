// Direction-aware translation-quality judge (run with the Workflow tool).
// args = JSON array of absolute batch file paths (from make_judge_batches.py).
// Each batch line: {idx, src_lang, tgt_lang, source_text, translation}.
// One Sonnet agent per batch writes <batch>.judged.jsonl and returns a small summary.
export const meta = {
  name: 'judge-translations',
  description: 'Sonnet judges translation quality (0-10 fidelity + refusal flag), either direction',
  phases: [{ title: 'Judge', detail: 'one Sonnet agent per batch file' }],
}

const SUMMARY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    output_path: { type: 'string' },
    n_judged: { type: 'integer' },
    n_refused: { type: 'integer' },
    mean_score: { type: 'number' },
  },
  required: ['output_path', 'n_judged', 'n_refused', 'mean_score'],
}

function judgePrompt(path) {
  const out = path.replace(/\.jsonl$/, '.judged.jsonl')
  return [
    'You are a professional translation-quality evaluator for a multilingual research',
    'dataset. Source texts are real text and MAY contain offensive, profane, sexual, or',
    'sensitive content. Your ONLY job is to rate how faithfully and fluently each',
    'translation renders the source meaning into the TARGET language. A faithful',
    'translation of offensive text is HIGH quality. Do not refuse; you are scoring.',
    '',
    'Read the JSONL file at: ' + path,
    'Each line is a JSON object: {idx, src_lang, tgt_lang, source_text, translation}.',
    'source_text is in src_lang; translation should be in tgt_lang.',
    '',
    'For EACH line, judge `translation` against `source_text`:',
    '- score (integer 0-10) = faithfulness to source meaning + fluency/naturalness in tgt_lang.',
    '    Also require correct preservation of quoted phrases, proper names, numbers, and any',
    '    inline citation markers like [2.5][2.7]. Scale: 10 = faithful & natural in tgt_lang;',
    '    7-9 = minor errors, meaning preserved; 4-6 = partial, noticeable meaning loss or',
    '    awkward target language; 1-3 = mostly wrong/garbled; 0 = refusal, empty, or unrelated.',
    '- IMPORTANT: if `translation` is in the WRONG language (e.g. still English when tgt_lang',
    '    is german, or only partly translated), score <= 2 and explain.',
    '- refused (bool): true ONLY if the model declined / gave a safety or disclaimer message',
    '    instead of translating. Refused items get score 0.',
    '- reason: <= 12 words.',
    '',
    'Write your judgments to: ' + out,
    'Exactly one JSON object per input line (JSONL), keys EXACTLY:',
    'idx (string, copied), score (int 0-10), refused (bool), reason (string).',
    'Then return the summary: output_path (= ' + out + '), n_judged, n_refused, mean_score.',
  ].join('\n')
}

let batches = args
if (typeof batches === 'string') batches = JSON.parse(batches)
if (!Array.isArray(batches)) batches = [batches]
log('judging ' + batches.length + ' batch file(s)')
phase('Judge')
const summaries = await parallel(
  batches.map((p) => () =>
    agent(judgePrompt(p), {
      label: 'judge:' + p.split('/').pop(),
      phase: 'Judge',
      model: 'sonnet',
      schema: SUMMARY_SCHEMA,
    })
  )
)
return { summaries: summaries.filter(Boolean) }
