import PageShell from '@/components/PageShell';
import StatTile from '@/components/StatTile';

export default function EvidencePage() {
  return (
    <PageShell title="Evidence" kicker="Metrics and controls">
      <p>
        Verifier-gated batch metrics and ablation evidence. GPU literature scoreboards are explicitly
        excluded (DROP).
      </p>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile value="0.117–0.167" label="iLISI range" note="Batches not mixed" />
        <StatTile value="Batch" label="Physical axis" note="Fig. 7 only" />
        <StatTile value="DROP" label="GPU baselines" note="Not run" />
        <StatTile value="5-seed" label="PCA floor" note="Table IV freeze" />
      </div>

      <section className="rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Batch integration metrics (Table II)</h2>
        <p className="mt-2 text-slate-700">
          scIB-style batch metrics on saved latents. Low iLISI confirms batches remain separated —
          MoCoO reports batch interference rather than claiming batch correction.
        </p>
      </section>

      <section className="rounded-2xl border border-amber-200 bg-amber-50/80 p-6">
        <h2 className="text-lg font-semibold text-slate-900">GPU DROP attestation</h2>
        <p className="mt-2 text-slate-700">
          Literature GPU baselines were not executed. No OPT-GPU leaderboard appears on this Site.
          Evidence is limited to on-disk CSV freeze and shipped figures.
        </p>
      </section>
    </PageShell>
  );
}
