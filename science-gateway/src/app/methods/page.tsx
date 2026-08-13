import PageShell from '@/components/PageShell';
import { SITE } from '@/lib/site';

export default function MethodsPage() {
  return (
    <PageShell title="Methods" kicker="Protocol and definitions">
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-900">Model</h2>
        <p>
          Momentum Contrast (MoCo) regularization on a VAE latent space with optional Neural ODE
          dynamics. Produces latent embeddings, optional velocity fields, and pseudotime estimates.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-900">Data and exports</h2>
        <ul className="list-disc space-y-2 pl-5">
          <li>IRALL frozen embedding exports for cross-dataset comparison</li>
          <li>Batch day labels as the only physical axis on this Site (Fig. 7)</li>
          <li>Tables I–IV from on-disk CSV freeze (see manuscript NUMBER-LOCK)</li>
        </ul>
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-900">Exclusions</h2>
        <ul className="list-disc space-y-2 pl-5">
          <li>GPU DROP: scVI, Harmony, scVelo, veloVI, OPT-GPU baselines not run</li>
          <li>No cell-state or trajectory proof claims from embedding plots alone</li>
          <li>No in-browser training or GPU jobs from this Site</li>
        </ul>
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-900">Reproducibility</h2>
        <p>
          Public code:{' '}
          <a href={SITE.github} className="text-brand hover:underline" target="_blank" rel="noopener noreferrer">
            github.com/PeterPonyu/MoCoO
          </a>
          . Package:{' '}
          <a href={SITE.pypi} className="text-brand hover:underline" target="_blank" rel="noopener noreferrer">
            pypi.org/project/mocoo
          </a>
          .
        </p>
      </section>
    </PageShell>
  );
}
