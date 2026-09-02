const pathways = [
  { field: 'Computer & Information Sciences', degree: "Bachelor's degree", growth: '+14.8%', tone: 'mint' },
  { field: 'Health Professions', degree: "Master's degree", growth: '+15.3%', tone: 'blue' },
  { field: 'Mathematics & Statistics', degree: "Bachelor's degree", growth: '+11.2%', tone: 'amber' },
];

export default function Home() {
  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Pathwise home"><span className="brand-mark">P</span><span>Pathwise</span></a>
        <nav aria-label="Main navigation"><a href="#how">How it works</a><a href="/methodology">Our data</a><a className="nav-cta" href="#start">Find your path</a></nav>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">A clearer way through college and career choices</p>
          <h1>Choose a path with evidence, not guesswork.</h1>
          <p className="hero-lede">Connect what you enjoy with degrees, careers, and colleges—then see the labor-market signals and program trends behind every recommendation.</p>
          <form className="path-starter" id="start" action="/explore" method="get">
            <label htmlFor="goal">What are you interested in?</label>
            <div className="starter-row"><input id="goal" name="goal" placeholder="Try data science, healthcare, design…" /><button type="submit">Explore paths <span aria-hidden="true">→</span></button></div>
            <p>No account required. Start broad—you can refine location, budget, and degree later.</p>
          </form>
          <div className="trust-row" id="evidence"><span><b>1.2M+</b> completion records</span><span><b>6,050</b> institutions</span><span><b>Federal</b> IPEDS + BLS data</span></div>
        </div>

        <div className="signal-panel" aria-label="Example recommended pathways">
          <div className="panel-top"><div><p className="panel-kicker">A sample shortlist</p><h2>Paths worth exploring</h2></div><span className="evidence-badge">Evidence-backed</span></div>
          <div className="path-list">
            {pathways.map((path, index) => (
              <article className="path-card" key={path.field}><span className={`path-number ${path.tone}`}>{index + 1}</span><div><h3>{path.field}</h3><p>{path.degree}</p></div><div className="growth"><b>{path.growth}</b><span>projected signal</span></div></article>
            ))}
          </div>
          <div className="panel-note"><span className="note-icon">i</span><p>Recommendations explain program momentum, job demand, and where the data is uncertain.</p></div>
        </div>
      </section>

      <section className="how" id="how">
        <p className="eyebrow">Built around the student decision</p>
        <h2>From “I’m interested in…” to a path you can act on.</h2>
        <div className="steps">
          {[
            ['01', 'Tell us what matters', 'Interests, preferred credential, location, budget, and career priorities.'],
            ['02', 'Compare real pathways', 'See related degrees, occupations, institutions, and the signals behind each match.'],
            ['03', 'Build your next-step plan', 'Shortlist options and know exactly what to verify before you apply.'],
          ].map(([number, title, copy]) => <article key={number}><span>{number}</span><h3>{title}</h3><p>{copy}</p></article>)}
        </div>
      </section>
      <footer><a className="brand" href="#top"><span className="brand-mark">P</span><span>Pathwise</span></a><p>Evidence to explore. Judgment stays with you.</p><div><a href="/methodology">Methodology</a><a href="/privacy">Privacy</a></div></footer>
    </main>
  );
}
