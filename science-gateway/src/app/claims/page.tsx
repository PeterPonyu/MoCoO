import PageShell from '@/components/PageShell';
import { SITE } from '@/lib/site';

export default function ClaimsPage() {
  return (
    <PageShell title="Claims" kicker="Falsifiable statements">
      <section className="rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Claim 1 — batch axis only</h2>
        <p className="mt-3 text-slate-700">{SITE.primaryClaim}</p>
        <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Would refute
        </h3>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
          <li>Fig. 7 batch-day correlation absent under reproduced latent exports</li>
          <li>Embedding trajectories presented as validated cell-state transitions without scope box</li>
        </ul>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Claim 2 — honest exclusions</h2>
        <p className="mt-3 text-slate-700">
          GPU literature baselines are not run (DROP). The package produces reproducible embedding
          exports suitable for batch diagnostics, not a cell-type atlas.
        </p>
        <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Out of scope
        </h3>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
          <li>Cell-state or trajectory proof from embedding plots alone</li>
          <li>GPU baseline leaderboard (scVI, Harmony, scVelo, veloVI, OPT-GPU)</li>
          <li>Journal venue packaging or invented article DOI</li>
        </ul>
      </section>
    </PageShell>
  );
}
