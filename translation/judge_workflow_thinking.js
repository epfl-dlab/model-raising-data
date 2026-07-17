// Thinking-aware translation judge. Same as judge_workflow.js, but the `translation`
// field contains the model's UN-separated reasoning followed by its final translation
// (gemma4 thinking mode, no vLLM reasoning parser). The judge must extract the FINAL
// translation and score only that.
export const meta = {
  name: 'judge-translations-thinking',
  description: 'Sonnet judges thinking-mode translations (extract final answer from the reasoning, then score)',
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
    'dataset. Source texts are real text and MAY contain sensitive content; rate only',
    'translation quality, do not refuse.',
    '',
    'IMPORTANT — these outputs were produced in THINKING mode and the reasoning was NOT',
    'separated from the answer. So the `translation` field contains the model\'s',
    'step-by-step reasoning (often starting with "thought" and drafting the translation',
    'several times) FOLLOWED by its final translation. For each line you MUST first',
    'identify the model\'s FINAL, complete translation (the last full rendering of the',
    'whole source into tgt_lang) and evaluate ONLY that final translation — ignore the',
    'planning/draft text. If no complete final translation is present, score low and say so.',
    '',
    'Read the JSONL file at: ' + path,
    'Each line: {idx, src_lang, tgt_lang, source_text, translation}. source_text is in',
    'src_lang; the FINAL translation should be in tgt_lang.',
    '',
    'For EACH line, judge the FINAL translation against source_text:',
    '- score (int 0-10) = faithfulness to source meaning + fluency/naturalness in tgt_lang,',
    '    preserving quoted phrases, names, numbers, and inline [x.y] citation markers.',
    '    10 faithful+natural; 7-9 minor errors; 4-6 partial/meaning loss; 1-3 mostly wrong;',
    '    0 refusal/empty/no usable final translation. Wrong-language final output scores <=2.',
    '- refused (bool): true only if the model declined / gave a disclaimer instead of',
    '    translating (score 0).',
    '- reason: <= 12 words.',
    '',
    'Write judgments to: ' + out,
    'One JSON object per input line (JSONL), keys EXACTLY: idx (string), score (int 0-10),',
    'refused (bool), reason (string). Then return the summary: output_path (= ' + out + '),',
    'n_judged, n_refused, mean_score.',
  ].join('\n')
}

let batches = args
if (typeof batches === 'string') batches = JSON.parse(batches)
if (!Array.isArray(batches)) batches = [batches]
log('judging ' + batches.length + ' thinking batch file(s)')
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
