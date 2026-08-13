import { ClaimBlock } from '@/components/PageShell';
import RouteCards from '@/components/RouteCards';
import FigurePanel from '@/components/FigurePanel';
import StatTile from '@/components/StatTile';
import { SITE } from '@/lib/site';

export default function HomePage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-teal-700">
        {SITE.kicker}
      </p>
      <h1 className="mt-2 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
        {SITE.title}
      </h1>
      <p className="mt-4 max-w-3xl text-lg text-slate-700">{SITE.lead}</p>

      <section className="mt-8 rounded-2xl border border-amber-200 bg-amber-50/80 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Not cell-state proof</h2>
        <p className="mt-2 text-slate-700">
          Embedding movement is not a cell changing type or trajectory. Figs 3 and 6 are not
          independent biological validation. GPU literature baselines (scVI, Harmony, scVelo, veloVI,
          OPT-GPU) were not run and are not shown.
        </p>
      </section>

      <section className="mt-10 rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Physical object
        </h2>
        <p className="mt-2 text-slate-800">{SITE.physicalObject}</p>
      </section>

      <div className="mt-8">
        <ClaimBlock />
      </div>

      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile value="Batch" label="Physical axis" note="Fig. 7 only" />
        <StatTile value="IRALL" label="Frozen exports" note="Not new biology" />
        <StatTile value="DROP" label="GPU baselines" note="Not run · not shown" />
        <StatTile value="0.0.3" label="Package version" note="pip install mocoo" />
      </div>

      <section className="mt-10">
        <FigurePanel
          src="/figures/F07_temporal_audit.png"
          alt="Four-panel temporal audit: Spearman of PC1 versus batch day across VAE configurations"
          kicker="Fig. 7 · batch, not cell state"
          caption="PC1 of a saved latent versus batch day (d0…d30). Panel D states this proxy is not cell-level ground truth. Low iLISI (0.117–0.167) means batches are not mixed; MoCoO is not a batch-correction method."
        />
      </section>

      <section className="mt-10">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Explore
        </h2>
        <RouteCards />
      </section>
    </div>
  );
}
