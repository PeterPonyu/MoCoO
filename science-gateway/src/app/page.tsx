import Link from 'next/link';
import { SITE } from '@/lib/site';

export default function HomePage() {
  return (
    <div className="mx-auto max-w-3xl px-5 py-12 sm:px-6" data-page-id="mocoo.pkg.home">
      <p className="text-[12px] font-medium tracking-[0.14em] text-rust uppercase">{SITE.kicker}</p>
      <h1 className="font-display mt-2 text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
        {SITE.title}
      </h1>
      <p className="mt-5 text-[17px] leading-7 text-stone-700">{SITE.lead}</p>

      <p className="mt-4 text-[15px] leading-6 text-stone-600">
        PyPI name <code className="font-mono text-ink">{SITE.packageName}</code>, version{' '}
        {SITE.packageVersion}. License MIT. Optional ODE heads are an API, not a trajectory proof.
      </p>

      <p className="mt-6 flex flex-wrap gap-x-5 gap-y-2 text-sm font-medium">
        <a className="text-rust underline decoration-stone-300 underline-offset-4" href={SITE.github}>
          github.com/PeterPonyu/MoCoO
        </a>
        <a className="text-rust underline decoration-stone-300 underline-offset-4" href={SITE.pypi}>
          pypi.org/project/mocoo
        </a>
      </p>

      <section className="mt-12">
        <h2 className="font-display text-2xl text-ink">What the package provides</h2>
        <ul className="mt-4 list-disc space-y-2 pl-5 text-[16px] leading-7 text-stone-700">
          <li>Count VAE with MSE, NB, ZINB, Poisson, and ZIP likelihoods</li>
          <li>Optional Neural ODE head (API only)</li>
          <li>Optional Momentum Contrast on augmented views</li>
          <li>Information bottleneck (<code className="font-mono">latent_dim</code> → <code className="font-mono">i_dim</code>)</li>
          <li>Optional disentanglement losses (DIP-VAE, β-TC-VAE, InfoVAE)</li>
          <li>Vector-field export helpers for velocity-style plots</li>
        </ul>
      </section>

      <section className="mt-12">
        <h2 className="font-display text-2xl text-ink">Install</h2>
        <pre className="mt-4 overflow-x-auto border border-stone-300 bg-white px-4 py-3 text-sm text-ink">
          <code>pip install mocoo</code>
        </pre>
        <p className="mt-3 text-[15px] text-stone-600">
          Source checkout: clone the repository and run <code className="font-mono">pip install -e .</code>.
          Development extras: <code className="font-mono">pip install -e &quot;.[dev]&quot;</code>.
        </p>
        <p className="mt-2 text-[15px]">
          <Link href="/methods" className="text-rust underline decoration-stone-300 underline-offset-4">
            Install notes
          </Link>
        </p>
      </section>

      <section className="mt-12">
        <h2 className="font-display text-2xl text-ink">Minimal fit</h2>
        <pre className="mt-4 overflow-x-auto border border-stone-300 bg-white px-4 py-3 text-sm leading-6 text-ink">
          <code>{`from mocoo import MoCoO

model = MoCoO(adata, layer='counts', loss_mode='nb', batch_size=128)
model.fit(epochs=100)
adata.obsm['X_mocoo'] = model.get_latent()`}</code>
        </pre>
        <p className="mt-3 text-[15px] text-stone-600">
          <code className="font-mono">use_ode=True</code> and <code className="font-mono">use_moco=True</code> are
          optional constructor flags. See{' '}
          <Link href="/results" className="text-rust underline decoration-stone-300 underline-offset-4">
            API
          </Link>
          .
        </p>
      </section>

      <section className="mt-12 border-t border-stone-300 pt-8">
        <h2 className="font-display text-2xl text-ink">This page</h2>
        <p className="mt-4 text-[16px] leading-7 text-stone-700">
          GitHub Pages for this repository is a code index. It does not host a results gallery,
          metric tables, or manuscript figures. Old paths stay as short notes so bookmarks do not
          break.
        </p>
      </section>
    </div>
  );
}
