export default function PageShell({
  title,
  kicker,
  pageId,
  children,
}: {
  title: string;
  kicker?: string;
  pageId: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mx-auto max-w-3xl px-5 py-12 sm:px-6" data-page-id={pageId}>
      {kicker ? (
        <p className="text-[12px] font-medium tracking-[0.14em] text-rust uppercase">{kicker}</p>
      ) : null}
      <h1 className="font-display mt-2 text-3xl font-semibold tracking-tight text-ink">{title}</h1>
      <div className="mt-8 space-y-6 text-[17px] leading-7 text-stone-700">{children}</div>
    </div>
  );
}
