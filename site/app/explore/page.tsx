'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';

type Program = { cip2:number; cip2_name:string; awlevel_name:string; program_net_pct_change:number|null; bls_occupational_growth:number|null; bls_annual_openings_mapped:number|null; alignment:string; sunset_label:string; baseline_avg_2019_2021:number|null };
type Institution = { unitid:string; institution_name:string; city:string; state:string; cip2_name:string; awlevel_name:string; pct_change_2019_2024:number|null; trend_direction:string };

const themes: Record<string,string[]> = {
  technology:['computer','data','software','cyber','information','math','statistics','technology'],
  health:['health','nursing','medical','therapy','biology'],
  business:['business','finance','accounting','marketing','management','economics'],
  design:['design','visual','arts','communication','media','architecture'],
  education:['education','teaching','library'],
  public:['public','legal','law','social','psychology','security'],
  engineering:['engineering','construction','mechanic','transportation'],
};

function relevance(name:string, goal:string) {
  const words = goal.toLowerCase().split(/\W+/).filter(w => w.length > 2);
  const expanded = new Set(words);
  for (const group of Object.values(themes)) if (group.some(w => words.includes(w))) group.forEach(w => expanded.add(w));
  const lower = name.toLowerCase();
  return [...expanded].reduce((score, word) => score + (lower.includes(word) ? 1 : 0), 0);
}
function pct(value:number|null) { return value == null ? 'Not available' : `${value > 0 ? '+' : ''}${value.toFixed(1)}%`; }
function safeScore(p:Program) {
  const demand = Math.max(-10, Math.min(20, p.bls_occupational_growth ?? 0));
  const momentum = Math.max(-50, Math.min(80, p.program_net_pct_change ?? 0));
  return Math.round(Math.max(1, Math.min(99, 58 + demand * 1.6 + momentum * .18)));
}

export default function Explore() {
  const [programs,setPrograms] = useState<Program[]>([]);
  const [institutions,setInstitutions] = useState<Institution[]>([]);
  const [goal,setGoal] = useState('');
  const [credential,setCredential] = useState('All credentials');
  const [selected,setSelected] = useState<number[]>([]);
  const [tab,setTab] = useState<'paths'|'colleges'>('paths');

  useEffect(() => {
    setGoal(new URLSearchParams(location.search).get('goal') || '');
    Promise.all([fetch('/data/programs.json').then(r=>r.json()),fetch('/data/dimensions.json').then(r=>r.json())]).then(([p,d])=>{setPrograms(p.programs);setInstitutions(d.institution_trends);});
  },[]);

  const ranked = useMemo(() => programs.filter(p => credential === 'All credentials' || p.awlevel_name === credential).map(p=>({p,match:relevance(p.cip2_name,goal),score:safeScore(p)})).filter(x=>!goal || x.match>0).sort((a,b)=>b.match-a.match || b.score-a.score).slice(0,30),[programs,goal,credential]);
  const chosen = ranked.filter(x=>selected.includes(x.p.cip2));
  const collegeRows = useMemo(() => institutions.filter(i=>ranked.slice(0,8).some(x=>x.p.cip2_name===i.cip2_name)).slice(0,24),[institutions,ranked]);
  const credentials = [...new Set(programs.map(p=>p.awlevel_name).filter(Boolean))].sort();
  const toggle=(id:number)=>setSelected(s=>s.includes(id)?s.filter(x=>x!==id):s.length<3?[...s,id]:s);

  return <main className="explore-shell">
    <header className="site-header"><Link className="brand" href="/"><span className="brand-mark">P</span><span>Pathwise</span></Link><nav><Link href="/methodology">Methodology</Link><Link href="/privacy">Privacy</Link></nav></header>
    <section className="explore-intro"><p className="eyebrow">Pathway explorer</p><h1>Turn an interest into options you can compare.</h1><p>Search across federal program-completion and occupational outlook signals. Strong signals are clues—not guarantees.</p></section>
    <section className="workspace">
      <aside className="filters"><h2>Your priorities</h2><label>Interest or goal<input value={goal} onChange={e=>setGoal(e.target.value)} placeholder="Data science, nursing…" /></label><label>Credential<select value={credential} onChange={e=>setCredential(e.target.value)}><option>All credentials</option>{credentials.map(c=><option key={c}>{c}</option>)}</select></label><div className="coach-note"><b>Coach tip</b><p>Start with a broad interest. Compare demand, student momentum, and real institutions before narrowing.</p></div><Link className="text-link" href="/methodology">How recommendations work →</Link></aside>
      <div className="results">
        <div className="results-head"><div><p>{ranked.length} relevant pathways</p><h2>{goal ? `Matches for “${goal}”` : 'Popular pathways to explore'}</h2></div><div className="tabs"><button className={tab==='paths'?'active':''} onClick={()=>setTab('paths')}>Pathways</button><button className={tab==='colleges'?'active':''} onClick={()=>setTab('colleges')}>Colleges</button></div></div>
        {tab==='paths' ? <div className="result-grid">{ranked.map(({p,score})=><article className="result-card" key={`${p.cip2}-${p.awlevel_name}`}><div className="card-top"><span className="match-score">{score} signal</span><button className={selected.includes(p.cip2)?'save active':'save'} onClick={()=>toggle(p.cip2)} disabled={!selected.includes(p.cip2)&&selected.length>=3}>{selected.includes(p.cip2)?'Added':'Compare'}</button></div><h3>{p.cip2_name}</h3><p className="credential">{p.awlevel_name}</p><dl><div><dt>Job outlook</dt><dd>{pct(p.bls_occupational_growth)}</dd></div><div><dt>Program momentum</dt><dd>{pct(p.program_net_pct_change)}</dd></div><div><dt>Annual openings</dt><dd>{p.bls_annual_openings_mapped?.toLocaleString() ?? 'Not available'}</dd></div></dl><p className="evidence-line">{p.alignment || 'Evidence is still developing'} · {p.sunset_label}</p></article>)}</div> : <div className="college-list">{collegeRows.map(i=><article key={`${i.unitid}-${i.cip2_name}`}><div><h3>{i.institution_name}</h3><p>{i.city}, {i.state} · {i.cip2_name}</p></div><div><b>{pct(i.pct_change_2019_2024)}</b><span>2019–24 completions</span></div></article>)}</div>}
        {!programs.length && <p className="loading">Loading federal pathway data…</p>}
        {programs.length>0 && ranked.length===0 && <div className="empty"><h3>Try a broader interest</h3><p>Use terms such as technology, healthcare, business, design, education, or engineering.</p></div>}
      </div>
    </section>
    {chosen.length>0 && <section className="compare-tray"><div><b>Compare pathways</b><span>{chosen.length}/3 selected</span></div>{chosen.map(x=><span key={x.p.cip2}>{x.p.cip2_name}<button onClick={()=>toggle(x.p.cip2)} aria-label={`Remove ${x.p.cip2_name}`}>×</button></span>)}<a href="#comparison">Review</a></section>}
    {chosen.length>1 && <section className="comparison" id="comparison"><p className="eyebrow">Side by side</p><h2>Your shortlist</h2><div className="comparison-grid">{chosen.map(({p,score})=><article key={p.cip2}><h3>{p.cip2_name}</h3><strong>{score}<small>/100 signal</small></strong><p>{p.awlevel_name}</p><ul><li>{pct(p.bls_occupational_growth)} occupational outlook</li><li>{pct(p.program_net_pct_change)} program momentum</li><li>{p.bls_annual_openings_mapped?.toLocaleString() ?? 'Unknown'} annual openings</li></ul></article>)}</div><p className="caveat">Use this comparison to build questions for an advisor or institution. Pathwise does not predict admissions, earnings, or individual outcomes.</p></section>}
    <footer><Link className="brand" href="/"><span className="brand-mark">P</span><span>Pathwise</span></Link><p>Evidence to explore. Judgment stays with you.</p><div><Link href="/methodology">Methodology</Link><Link href="/privacy">Privacy</Link></div></footer>
  </main>;
}
