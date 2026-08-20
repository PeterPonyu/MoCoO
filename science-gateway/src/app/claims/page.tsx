import Link from 'next/link';
import PageShell from '@/components/PageShell';

export default function ClaimsPage() {
  return (
    <PageShell title="Limits" kicker="Not an article page" pageId="mocoo.pkg.limits">
      <p>
        No article claims, venue packaging, or DOI are offered on this page. The public object is
        the mocoo package.
      </p>
      <p>
        <Link href="/" className="text-rust underline decoration-stone-300 underline-offset-4">
          Return to the package index
        </Link>
        .
      </p>
    </PageShell>
  );
}
