'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';

type Program = { cip2:number; cip2_name:string; awlevel_name:string; program_net_pct_change:number|null; bls_occupational_growth:number|null; bls_annual_openings_mapped:number|null; alignment:string; sunset_label:string };
type Institution = { unitid:string; institution_name:string; city:string; state:string; cip2_name:string; awlevel_name:string; pct_change_2019_2024:number|null; trend_direction:string };
type Occupation = { soc:string; title:string; growth:number|null; annual_openings:number|null; median_wage:number|null; education:string|null; experience:string|null; training:string|null };
type OccupationData = { source:string; by_cip2:Record<string,Occupation[]> };
type Tab = 'paths'|'careers'|'colleges';

const themes: Record<string,string[]> = {
  technology:['computer','data','software','cyber','information','math','statistics','technology'], health:['health','nursing','medical','therapy','biology'],
  business:['business','finance','accounting','marketing','management','economics'], design:['design','visual','arts','communication','media','architecture'],
  education:['education','teaching','library'], public:['public','legal','law','social','psychology','security'], engineering:['engineering','construction','mechanic','transportation'],
};
const workStyles = ['Build things','Help people','Solve puzzles','Create & communicate'];
const priorities = [{id:'balanced',label:'Balanced view'},{id:'growth',label:'Future growth'},{id:'openings',label:'More openings'},{id:'momentum',label:'Student momentum'}];

function relevance(name:string, goal:string, style:string) {
  const words = goal.toLowerCase().split(/\W+/).filter(w => w.length > 2);
  const styleWords:Record<string,string[]>={'Build things':['engineering','computer','construction','mechanic'],'Help people':['health','education','social','public'],'Solve puzzles':['computer','math','statistics','science'],'Create & communicate':['arts','design','communication','language','media']};
  const expanded = new Set([...words,...(styleWords[style]||[])]);
  for (const group of Object.values(themes)) if (group.some(w => words.includes(w))) group.forEach(w => expanded.add(w));
  const lower = name.toLowerCase(); return [...expanded].reduce((score, word) => score + (lower.includes(word) ? 1 : 0), 0);
}
function pct(value:number|null|undefined) { return value == null ? '—' : `${value > 0 ? '+' : ''}${value.toFixed(1)}%`; }
function money(value:number|null|undefined){return value==null?'—':new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(value)}
function scoreFor(p:Program,priority:string){const growth=Math.max(-10,Math.min(20,p.bls_occupational_growth??0));const momentum=Math.max(-50,Math.min(80,p.program_net_pct_change??0));const openings=Math.log10(Math.max(1,p.bls_annual_openings_mapped??1));const raw=priority==='growth'?55+growth*2.2+momentum*.08:priority==='openings'?45+openings*9+growth*.6:priority==='momentum'?55+momentum*.38+growth*.5:55+growth*1.3+momentum*.16+openings*3;return Math.round(Math.max(1,Math.min(99,raw)))}

export default function Explore(){
  const [programs,setPrograms]=useState<Program[]>([]); const [institutions,setInstitutions]=useState<Institution[]>([]); const [occupations,setOccupations]=useState<OccupationData>({source:'',by_cip2:{}});
  const [goal,setGoal]=useState(''); const [credential,setCredential]=useState('All credentials'); const [state,setState]=useState('All states'); const [style,setStyle]=useState('Solve puzzles'); const [priority,setPriority]=useState('balanced'); const [tab,setTab]=useState<Tab>('paths'); const [selected,setSelected]=useState<string[]>([]); const [guideOpen,setGuideOpen]=useState(true);
  useEffect(()=>{setGoal(new URLSearchParams(location.search).get('goal')||'');Promise.all([fetch('/data/programs.json').then(r=>r.json()),fetch('/data/dimensions.json').then(r=>r.json()),fetch('/data/occupations.json').then(r=>r.json())]).then(([p,d,o])=>{setPrograms(p.programs);setInstitutions(d.institution_trends);setOccupations(o)}).catch(()=>setPrograms([]))},[]);
  const ranked=useMemo(()=>programs.filter(p=>credential==='All credentials'||p.awlevel_name===credential).map(p=>({p,match:relevance(p.cip2_name,goal,style),score:scoreFor(p,priority),key:`${p.cip2}-${p.awlevel_name}`})).filter(x=>!goal||x.match>0).sort((a,b)=>b.match-a.match||b.score-a.score).slice(0,36),[programs,goal,credential,style,priority]);
  const careerRows=useMemo(()=>ranked.slice(0,10).flatMap(x=>(occupations.by_cip2[String(x.p.cip2)]||[]).slice(0,5).map(o=>({...o,field:x.p.cip2_name,cip2:x.p.cip2}))).filter((o,i,a)=>a.findIndex(x=>x.soc===o.soc)===i).sort((a,b)=>(b.annual_openings||0)-(a.annual_openings||0)).slice(0,30),[ranked,occupations]);
  const collegeRows=useMemo(()=>institutions.filter(i=>(state==='All states'||i.state===state)&&ranked.slice(0,10).some(x=>x.p.cip2_name===i.cip2_name)).slice(0,40),[institutions,ranked,state]);
  const chosen=ranked.filter(x=>selected.includes(x.key)); const credentials=[...new Set(programs.map(p=>p.awlevel_name).filter(Boolean))].sort(); const states=[...new Set(institutions.map(i=>i.state).filter(Boolean))].sort();
  const toggle=(key:string)=>setSelected(s=>s.includes(key)?s.filter(x=>x!==key):s.length<3?[...s,key]:s);
  return <main className="app-v2">
    <header className="site-header app-header"><Link className="brand v2-brand" href="/"><span className="brand-mark">V</span><span>viascope</span></Link><div className="app-nav"><Link href="/methodology">How data works</Link><span className="data-live"><i/> Federal data loaded</span></div></header>
    <div className="app-frame">
      <aside className="app-sidebar">
        <div className="sidebar-title"><span>Your scope</span><button onClick={()=>setGuideOpen(v=>!v)} aria-expanded={guideOpen}>{guideOpen?'Hide guide':'Show guide'}</button></div>
        {guideOpen&&<section className="guide-flow"><div className="guide-progress"><i/><i/><i/><span>3 quick questions</span></div><label><b>What are you curious about?</b><input value={goal} onChange={e=>setGoal(e.target.value)} placeholder="Healthcare, data, design…"/></label><fieldset><legend>How do you like to work?</legend><div className="choice-stack">{workStyles.map(x=><button className={style===x?'selected':''} onClick={()=>setStyle(x)} key={x}><span>{style===x?'✓':'○'}</span>{x}</button>)}</div></fieldset><fieldset><legend>What matters most right now?</legend><div className="choice-stack priorities">{priorities.map(x=><button className={priority===x.id?'selected':''} onClick={()=>setPriority(x.id)} key={x.id}><span>{priority===x.id?'●':'○'}</span>{x.label}</button>)}</div></fieldset></section>}
        <div className="filter-block"><label>Credential<select value={credential} onChange={e=>setCredential(e.target.value)}><option>All credentials</option>{credentials.map(x=><option key={x}>{x}</option>)}</select></label><label>College state<select value={state} onChange={e=>setState(e.target.value)}><option>All states</option>{states.map(x=><option key={x}>{x}</option>)}</select></label></div>
        <p className="privacy-note">Your answers stay in this browser. No account required.</p>
      </aside>
      <section className="app-main">
        <div className="app-heading"><div><p>Possibility map</p><h1>{goal?<>Routes for <span>{goal}</span></>:'Start broad. Find a route that fits.'}</h1><p>Built around “{style.toLowerCase()}” with a {priorities.find(x=>x.id===priority)?.label.toLowerCase()}.</p></div><button className="reset-button" onClick={()=>{setGoal('');setCredential('All credentials');setState('All states');setPriority('balanced')}}>Reset scope</button></div>
        <div className="metric-rail"><div><span>Matched pathways</span><b>{ranked.length}</b></div><div><span>Related careers</span><b>{careerRows.length}</b></div><div><span>College program signals</span><b>{collegeRows.length}</b></div><div className="rail-note"><span>Evidence coverage</span><b>{programs.length?'IPEDS + BLS':'Loading…'}</b></div></div>
        <div className="app-tabs" role="tablist" aria-label="Explore results">{([['paths','Study paths'],['careers','Careers & wages'],['colleges','Colleges']] as [Tab,string][]).map(([id,label])=><button role="tab" aria-selected={tab===id} className={tab===id?'active':''} onClick={()=>setTab(id)} key={id}>{label}{id==='paths'&&<span>{ranked.length}</span>}</button>)}</div>
        {!programs.length?<div className="skeleton-list" aria-label="Loading data"><i/><i/><i/></div>:tab==='paths'?<div className="path-table"><div className="table-head"><span>Pathway</span><span>Signal</span><span>Job outlook</span><span>Annual openings</span><span/></div>{ranked.map(({p,score,key})=><article key={key}><div className="path-name"><i style={{'--signal':`${score}%`} as React.CSSProperties}/><span><b>{p.cip2_name}</b><small>{p.awlevel_name}</small></span></div><div className="score-cell"><b>{score}</b><small>/100</small></div><div className={(p.bls_occupational_growth||0)>=0?'positive':'negative'}>{pct(p.bls_occupational_growth)}</div><div>{p.bls_annual_openings_mapped?.toLocaleString()??'—'}</div><button className={selected.includes(key)?'compare-add selected':'compare-add'} disabled={!selected.includes(key)&&selected.length>=3} onClick={()=>toggle(key)}>{selected.includes(key)?'✓ Added':'+ Compare'}</button></article>)}</div>:tab==='careers'?<div className="career-table"><div className="career-head"><span>Occupation</span><span>Median wage</span><span>2024–34 growth</span><span>Annual openings</span><span>Typical entry education</span></div>{careerRows.map(o=><article key={o.soc}><div><b>{o.title}</b><small>{o.field}</small></div><strong>{money(o.median_wage)}</strong><span className={(o.growth||0)>=0?'positive':'negative'}>{pct(o.growth)}</span><span>{o.annual_openings?.toLocaleString()??'—'}</span><span>{o.education||'Varies'}</span></article>)}</div>:<div className="college-table"><div className="college-summary"><p>Showing institutions with strong recent completion volume for your matched fields{state==='All states'?'.':` in ${state}.`}</p><span>Cost and earnings enrichment is next</span></div>{collegeRows.map(i=><article key={`${i.unitid}-${i.cip2_name}`}><div><b>{i.institution_name}</b><small>{i.city}, {i.state}</small></div><div><span>{i.cip2_name}</span><small>{i.awlevel_name}</small></div><div className={(i.pct_change_2019_2024||0)>=0?'positive':'negative'}><b>{pct(i.pct_change_2019_2024)}</b><small>completions since 2019</small></div></article>)}</div>}
        {programs.length>0&&ranked.length===0&&<div className="empty"><h3>Widen your scope</h3><p>Try technology, health, business, design, education, science, or engineering.</p></div>}
      </section>
    </div>
    {chosen.length>0&&<section className="compare-tray v2-tray"><div><b>Your shortlist</b><span>{chosen.length}/3 paths</span></div>{chosen.map(x=><span key={x.key}>{x.p.cip2_name}<button onClick={()=>toggle(x.key)} aria-label={`Remove ${x.p.cip2_name}`}>×</button></span>)}<a href="#comparison">Compare now</a></section>}
    {chosen.length>1&&<section className="comparison v2-comparison" id="comparison"><div className="compare-heading"><div><span>Decision workspace</span><h2>Compare your shortlist</h2></div><p>Use the evidence to ask sharper questions—not to outsource your choice.</p></div><div className="comparison-grid">{chosen.map(({p,score,key})=><article key={key}><span className="compare-field">{p.awlevel_name}</span><h3>{p.cip2_name}</h3><strong>{score}<small>/100</small></strong><dl><div><dt>Job outlook</dt><dd>{pct(p.bls_occupational_growth)}</dd></div><div><dt>Program momentum</dt><dd>{pct(p.program_net_pct_change)}</dd></div><div><dt>Annual openings</dt><dd>{p.bls_annual_openings_mapped?.toLocaleString()??'—'}</dd></div></dl><button onClick={()=>toggle(key)}>Remove</button></article>)}</div></section>}
  </main>
}
