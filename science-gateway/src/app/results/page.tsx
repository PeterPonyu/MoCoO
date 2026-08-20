import Link from 'next/link';
import PageShell from '@/components/PageShell';

export default function ResultsPage() {
  return (
    <PageShell title="API" kicker="Package helpers" pageId="mocoo.pkg.api">
      <p>
        This path is not a results gallery. It lists helpers exported by the installable package.
      </p>

      <section>
        <h2 className="font-display text-xl text-ink">Training</h2>
        <pre className="mt-3 overflow-x-auto border border-stone-300 bg-white px-4 py-3 text-sm text-ink">
          <code>model.fit(epochs=400, patience=25, val_every=5)</code>
        </pre>
      </section>

      <section>
        <h2 className="font-display text-xl text-ink">Exports</h2>
        <ul className="mt-3 list-disc space-y-1 pl-5">
          <li>
            <code className="font-mono">get_latent()</code> — latent embeddings
          </li>
          <li>
            <code className="font-mono">get_bottleneck()</code> — bottleneck features
          </li>
          <li>
            <code className="font-mono">get_time()</code> / <code className="font-mono">get_velocity()</code> /{' '}
            <code className="font-mono">get_transition()</code> — ODE-head helpers when that flag is on
          </li>
          <li>
            <code className="font-mono">get_loss_history()</code>, <code className="font-mono">get_metrics_history()</code>,{' '}
            <code className="font-mono">get_resource_metrics()</code>
          </li>
        </ul>
      </section>

      <p>
        ODE helpers are optional API surfaces. They are not presented here as a validated
        trajectory method.{' '}
        <Link href="/" className="text-rust underline decoration-stone-300 underline-offset-4">
          Package index
        </Link>
        .
      </p>
    </PageShell>
  );
}
