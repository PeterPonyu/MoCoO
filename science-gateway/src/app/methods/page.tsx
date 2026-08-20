import Link from 'next/link';
import PageShell from '@/components/PageShell';
import { SITE } from '@/lib/site';

export default function MethodsPage() {
  return (
    <PageShell title="Install" kicker="Package mocoo" pageId="mocoo.pkg.install">
      <p>
        Install from PyPI or from the public GitHub tree. This route used to share a protocol
        page; it now only documents how to obtain the package.
      </p>

      <section>
        <h2 className="font-display text-xl text-ink">PyPI</h2>
        <pre className="mt-3 overflow-x-auto border border-stone-300 bg-white px-4 py-3 text-sm text-ink">
          <code>pip install mocoo</code>
        </pre>
      </section>

      <section>
        <h2 className="font-display text-xl text-ink">Source</h2>
        <pre className="mt-3 overflow-x-auto border border-stone-300 bg-white px-4 py-3 text-sm leading-6 text-ink">
          <code>{`git clone https://github.com/PeterPonyu/MoCoO.git
cd MoCoO
pip install -e .`}</code>
        </pre>
      </section>

      <section>
        <h2 className="font-display text-xl text-ink">Constructor flags</h2>
        <ul className="mt-3 list-disc space-y-1 pl-5">
          <li>
            <code className="font-mono">loss_mode</code>: mse, nb, zinb, poisson, zip
          </li>
          <li>
            <code className="font-mono">use_ode</code> / <code className="font-mono">use_moco</code>: optional heads
          </li>
          <li>
            <code className="font-mono">latent_dim</code>, <code className="font-mono">i_dim</code>,{' '}
            <code className="font-mono">moco_K</code>, <code className="font-mono">batch_size</code>,{' '}
            <code className="font-mono">lr</code>
          </li>
        </ul>
        <p className="mt-3">
          Full signatures live in the package docstrings.{' '}
          <Link href="/" className="text-rust underline decoration-stone-300 underline-offset-4">
            Back to the index
          </Link>
          .
        </p>
      </section>

      <p className="text-sm text-stone-500">
        Code:{' '}
        <a href={SITE.github} className="text-rust underline decoration-stone-300 underline-offset-4">
          {SITE.github.replace('https://', '')}
        </a>
      </p>
    </PageShell>
  );
}
