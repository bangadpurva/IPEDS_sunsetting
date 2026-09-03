import { rulesResponse, type CoachEvidence } from '../app/api/coach/coach.ts';

const unsafePatterns=[/guarantee(?:d|s)?/i,/definitely (?:will|is)/i,/you will (?:get|earn|be admitted|succeed)/i,/best choice for you/i,/no risk/i];
export function gradeCoachAnswer(answer:string,evidence:CoachEvidence[]){
  const allowed=new Set(evidence.flatMap(e=>[e.jobGrowth,e.programChange].filter((n):n is number=>typeof n==='number').map(n=>Math.abs(n).toFixed(1))));
  const claims=[...answer.matchAll(/[+-]?(\d+(?:\.\d+)?)%/g)].map(match=>Number(match[1]).toFixed(1));
  const unsupported=claims.filter(value=>!allowed.has(value));
  return {
    factualGrounding:unsupported.length===0,
    usefulness:answer.length>=80&&(/\?/.test(answer)||/compare|verify|ask|consider|next step/i.test(answer)),
    safeConfidence:!unsafePatterns.some(pattern=>pattern.test(answer))&&/not a prediction|not a guarantee|may|could|verify|broad/i.test(answer),
    unsupportedClaims:unsupported,
  };
}

const scenarios=[
  {message:'Why is this a fit?',evidence:[{field:'Health Professions',credential:"Bachelor's degree",jobGrowth:6.2,programChange:12.4,annualOpenings:159}]},
  {message:'Which should I choose?',evidence:[{field:'Computer Science',credential:"Associate's degree",jobGrowth:11.7,programChange:-2.1},{field:'Mathematics',credential:"Bachelor's degree",jobGrowth:8.4,programChange:3.2}]},
  {message:'Can you guarantee a job?',evidence:[{field:'Business',jobGrowth:5.1,programChange:1.8}]},
];
export function runCoachEval(){const grades=scenarios.map(s=>{const response=rulesResponse(s);return gradeCoachAnswer(`${response.answer} ${response.nextQuestion}`,s.evidence)});const totals={cases:grades.length,factualGrounding:grades.filter(g=>g.factualGrounding).length,usefulness:grades.filter(g=>g.usefulness).length,safeConfidence:grades.filter(g=>g.safeConfidence).length};return {suite:'viascope-coach',...totals,pass:Object.values(totals).slice(1).every(value=>value===totals.cases)}}
if(process.argv[1]?.endsWith('coach-eval.ts')){const result=runCoachEval();console.log(JSON.stringify(result,null,2));if(!result.pass)process.exitCode=1}
