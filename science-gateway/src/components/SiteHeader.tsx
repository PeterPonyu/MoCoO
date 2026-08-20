import Link from 'next/link';
import { SITE } from '@/lib/site';

export default function SiteHeader() {
  return (
    <header
      className="border-b border-stone-300 bg-paper/95"
      data-chrome="mocoo-rail"
    >
      <div className="mx-auto flex max-w-3xl items-baseline justify-between gap-4 px-5 py-4 sm:px-6">
        <Link href="/" className="font-display text-xl font-semibold tracking-tight text-ink">
          {SITE.navTitle}
          <span className="ml-2 font-sans text-xs font-medium tracking-wide text-stone-500">
            {SITE.packageName}
          </span>
        </Link>
        <nav className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm font-medium text-stone-600" aria-label="Package">
          <Link href="/methods" className="hover:text-rust">
            Install
          </Link>
          <Link href="/results" className="hover:text-rust">
            API
          </Link>
          <a href={SITE.github} className="hover:text-rust">
            GitHub
          </a>
          <a href={SITE.pypi} className="hover:text-rust">
            PyPI
          </a>
        </nav>
      </div>
    </header>
  );
}
