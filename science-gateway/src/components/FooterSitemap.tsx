import Link from 'next/link';
import { SITE } from '@/lib/site';

export default function FooterSitemap() {
  return (
    <footer className="mt-auto border-t border-stone-300 bg-paper" data-chrome="mocoo-colophon">
      <div className="mx-auto flex max-w-3xl flex-col gap-2 px-5 py-5 text-sm text-stone-600 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <span className="font-display text-ink">{SITE.navTitle} package index</span>
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          <Link href="/" className="hover:text-rust">
            Home
          </Link>
          <Link href="/methods" className="hover:text-rust">
            Install
          </Link>
          <Link href="/results" className="hover:text-rust">
            API
          </Link>
          <Link href="/evidence" className="hover:text-rust">
            Scope
          </Link>
          <Link href="/claims" className="hover:text-rust">
            Limits
          </Link>
          <a href={SITE.github} className="hover:text-rust">
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}
