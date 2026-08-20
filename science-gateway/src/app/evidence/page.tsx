import Link from 'next/link';
import PageShell from '@/components/PageShell';

export default function EvidencePage() {
  return (
    <PageShell title="Scope" kicker="No public scoreboard" pageId="mocoo.pkg.scope">
      <p>
        This repository&apos;s GitHub Pages site does not publish metric tables, evidence panels, or
        figure plates.
      </p>
      <p>
        If you need the installable code, use the{' '}
        <Link href="/" className="text-rust underline decoration-stone-300 underline-offset-4">
          package index
        </Link>{' '}
        or the GitHub repository.
      </p>
    </PageShell>
  );
}
