export type LearnerProfile = {
  interest: string; workStyle: string; priority: string; budget: number | null;
  timeline: string | null; location: string | null; credential: string | null;
  learningMode: 'online' | 'campus' | 'hybrid' | null; careerGoal: string | null; missing: string[];
};

const money = (text: string) => {
  const match = text.match(/(?:\$|under |budget(?: is| of)? )([\d,]{3,})/i);
  return match ? Number(match[1].replaceAll(',', '')) : null;
};

export function extractLearnerProfile(message: string, seed?: Record<string, unknown>): LearnerProfile {
  const text = message.toLowerCase();
  const credential = /associate/.test(text) ? "Associate's degree" : /bachelor/.test(text) ? "Bachelor's degree" : /master/.test(text) ? "Master's degree" : /certificate/.test(text) ? 'Certificate' : String(seed?.credential || '') || null;
  const learningMode = /\bonline\b/.test(text) ? 'online' : /\bhybrid\b/.test(text) ? 'hybrid' : /on campus|in person|in-person/.test(text) ? 'campus' : (['online', 'campus', 'hybrid'].includes(String(seed?.learningMode)) ? String(seed?.learningMode) as LearnerProfile['learningMode'] : null);
  const timeline = text.match(/(?:within|in) ((?:\d+|one|two|three|four|five|six) (?:months?|years?))/)?.[1] || String(seed?.timeline || '') || null;
  const location = message.match(/(?:near|around) ([A-Z][A-Za-z .'-]+(?:, [A-Z]{2})?)/)?.[1]?.trim() || String(seed?.location || '') || null;
  const careerGoal = message.match(/(?:become|work as|career as) (?:an? )?([^,.!?]+)/i)?.[1]?.trim() || String(seed?.careerGoal || '') || null;
  const profile: LearnerProfile = {interest: String(seed?.interest || ''), workStyle: String(seed?.workStyle || ''), priority: String(seed?.priority || ''), budget: money(message) ?? (Number.isFinite(seed?.budget) ? Number(seed?.budget) : null), timeline, location, credential, learningMode, careerGoal, missing: []};
  profile.missing = [['budget', profile.budget], ['timeline', profile.timeline], ['location', profile.location], ['credential', profile.credential], ['learning mode', profile.learningMode]].filter(([, value]) => !value).map(([label]) => String(label));
  return profile;
}

export function nextProfileQuestion(profile: LearnerProfile) {
  const questions: Record<string, string> = {budget: 'What is the most you want to spend per year after grants and scholarships?', timeline: 'How quickly would you like to complete your next credential?', location: 'Where should we look for programs—or are you open to anywhere?', credential: 'Are you considering a certificate, associate, bachelor’s, or graduate credential?', 'learning mode': 'Would you prefer online, campus, or hybrid learning?'};
  return questions[profile.missing[0]] || 'Would you like to compare cost, flexibility, or career outlook next?';
}
