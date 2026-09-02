import Link from 'next/link';

const signals = [
  { label: 'Work you might enjoy', value: 'Analyze · Build · Explain', color: 'violet' },
  { label: 'Study routes', value: '18 matched programs', color: 'lime' },
  { label: 'Career evidence', value: '7.7% projected growth', color: 'coral' },
];

export default function Home() {
  return <main className="landing-v2" id="top">
    <header className="site-header v2-header">
      <Link className="brand v2-brand" href="/" aria-label="Viascope home"><span className="brand-mark">V</span><span>viascope</span></Link>
      <nav aria-label="Main navigation"><a href="#approach">How it works</a><Link href="/methodology">The evidence</Link><Link className="nav-cta" href="/explore">Open explorer</Link></nav>
    </header>
    <section className="v2-hero">
      <div className="v2-copy">
        <p className="intro-label"><span>New</span> A student decision workspace</p>
        <h1>See where your interests could take you.</h1>
        <p className="hero-lede">Viascope connects the things you care about to programs, careers, and colleges—then shows you the evidence without making the decision for you.</p>
        <form className="v2-starter" action="/explore" method="get">
          <label htmlFor="goal">Start with anything</label>
          <div><input id="goal" name="goal" required placeholder="I like solving problems with data…"/><button type="submit">Map my options <span aria-hidden="true">↗</span></button></div>
          <p>Free to explore · No account · Built from federal education and labor data</p>
        </form>
      </div>
      <div className="route-board" aria-label="Example Viascope journey">
        <div className="route-head"><span>Your possibility map</span><b>Data + technology</b></div>
        <div className="route-origin"><span>YOU</span><p>Curious about patterns<br/>and solving problems</p></div>
        <div className="route-line" aria-hidden="true"><i/><i/><i/></div>
        <div className="route-destinations">
          {signals.map((signal,index)=><article className={signal.color} key={signal.label}><span>0{index+1}</span><div><p>{signal.label}</p><b>{signal.value}</b></div></article>)}
        </div>
        <div className="route-foot"><span><i/> IPEDS</span><span><i/> BLS</span><p>Signals updated from public sources</p></div>
      </div>
    </section>
    <section className="proof-strip"><p>One place to connect</p><div><b>Interests</b><span>→</span><b>Programs</b><span>→</span><b>Careers</b><span>→</span><b>Colleges</b></div></section>
    <section className="v2-approach" id="approach">
      <div><h2>A clearer decision starts with better questions.</h2><p>Not another personality quiz. Viascope helps you investigate the routes between who you are now and what you might do next.</p></div>
      <ol><li><span>1</span><div><b>Tell us what matters</b><p>Interests, preferred credential, location, and priorities.</p></div></li><li><span>2</span><div><b>Read the signals</b><p>Compare program momentum, job outlook, openings, and institutions.</p></div></li><li><span>3</span><div><b>Leave with a shortlist</b><p>Build questions to take to families, advisors, and colleges.</p></div></li></ol>
    </section>
    <footer><Link className="brand v2-brand" href="/"><span className="brand-mark">V</span><span>viascope</span></Link><p>Explore widely. Decide thoughtfully.</p><div><Link href="/methodology">Methodology</Link><Link href="/privacy">Privacy</Link></div></footer>
  </main>;
}
