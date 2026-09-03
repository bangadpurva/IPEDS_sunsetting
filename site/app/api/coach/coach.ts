export type CoachEvidence = {
  field: string;
  credential?: string;
  score?: number;
  jobGrowth?: number | null;
  annualOpenings?: number | null;
  programChange?: number | null;
};

export type CoachRequest = {
  message: string;
  profile?: { interest?: string; workStyle?: string; priority?: string };
  evidence?: CoachEvidence[];
};

const pct = (value?: number | null) => value == null ? 'not reported' : `${value > 0 ? '+' : ''}${value.toFixed(1)}%`;

export function normalizeCoachRequest(value: unknown): CoachRequest {
  if (!value || typeof value !== 'object') throw new Error('Send a question to the coach.');
  const raw = value as CoachRequest;
  const message = String(raw.message || '').trim().slice(0, 800);
  if (!message) throw new Error('Send a question to the coach.');
  return {
    message,
    profile: {
      interest: String(raw.profile?.interest || '').slice(0, 120),
      workStyle: String(raw.profile?.workStyle || '').slice(0, 80),
      priority: String(raw.profile?.priority || '').slice(0, 80),
    },
    evidence: Array.isArray(raw.evidence) ? raw.evidence.slice(0, 5).map(item => ({
      field: String(item.field || '').slice(0, 140), credential: String(item.credential || '').slice(0, 100),
      score: Number.isFinite(item.score) ? item.score : undefined,
      jobGrowth: Number.isFinite(item.jobGrowth) ? item.jobGrowth : null,
      annualOpenings: Number.isFinite(item.annualOpenings) ? item.annualOpenings : null,
      programChange: Number.isFinite(item.programChange) ? item.programChange : null,
    })).filter(item => item.field) : [],
  };
}

export function rulesResponse(request: CoachRequest) {
  const evidence = request.evidence || [];
  if (!evidence.length) return {
    answer: 'Give Viascope an interest or choose a work style first. I can then explain the strongest matching fields using the education and career evidence on this page.',
    nextQuestion: 'What subject, problem, or kind of work are you curious about?',
  };
  const [first, second] = evidence;
  const intro = `Start with ${first.field}${first.credential ? ` (${first.credential})` : ''}. Its current evidence shows ${pct(first.jobGrowth)} projected job growth and ${pct(first.programChange)} change in program completions.`;
  const comparison = second ? ` Compare it with ${second.field}; a different evidence pattern can reveal whether you value labor-market demand, student momentum, or the credential itself most.` : '';
  const caution = ' These are broad national signals, not a prediction of admission, cost, earnings, or personal fit.';
  return { answer: intro + comparison + caution, nextQuestion: 'Which matters most for your next step: cost, time to complete, nearby options, or career outlook?' };
}

export function modelPrompt(request: CoachRequest, fallback: ReturnType<typeof rulesResponse>, structuredProfile?:unknown) {
  return `You are the Viascope education decision coach. Answer the learner's question using only the supplied evidence. Be warm, concise, practical, and candid about missing data. Never promise admission, employment, salary, or personal outcomes. Distinguish national field-level evidence from institution-level facts. End with one useful follow-up question.\n\nLearner profile: ${JSON.stringify(structuredProfile||request.profile)}\nEvidence: ${JSON.stringify(request.evidence)}\nDeterministic fallback: ${fallback.answer}\nLearner question: ${request.message}`;
}
