/**
 * MoCoO science gateway site config.
 */
export const SITE = {
  slug: 'MoCoO',
  navTitle: 'MoCoO',
  title:
    'Momentum-contrast VAE embeddings report batch interference, not cell-state proof',
  kicker: 'ZF Lab · scRNA-seq representation · batch axis',
  lead:
    'The public object is a latent representation (embedding, optional velocity/pseudotime). The only physical axis drawn on this leaf is technical batch (Fig. 7) — not a validated cell-state atlas.',
  physicalObject:
    'Fig. 7 temporal audit: PC1 of a saved latent versus batch day. Not cell-level ground truth.',
  primaryClaim:
    'MoCoO produces frozen embedding exports suitable for batch-axis diagnostics; embedding movement is not evidence of cell type change.',
  homepage: 'https://peterponyu.github.io/',
  scportal: 'https://peterponyu.github.io/scportal/',
  github: 'https://github.com/PeterPonyu/MoCoO',
  pypi: 'https://pypi.org/project/mocoo/',
} as const;

export type BadgeConfig = {
  label: string;
  href?: string;
  enabled: boolean;
  disabledReason?: string;
};

export const BADGES = {
  code: {
    label: 'Code',
    href: SITE.github,
    enabled: true,
  } satisfies BadgeConfig,
  site: {
    label: 'Site',
    href: 'https://peterponyu.github.io/MoCoO/',
    enabled: true,
  } satisfies BadgeConfig,
  archive: {
    label: 'Archive',
    enabled: false,
    disabledReason: 'No Zenodo record yet',
  } satisfies BadgeConfig,
  articleDoi: {
    label: 'Article DOI',
    enabled: false,
    disabledReason: 'On acceptance',
  } satisfies BadgeConfig,
} as const;

export const ROUTES = [
  { href: '/results', label: 'Results', number: '01', blurb: 'Fig. 7 batch axis first; embedding floats follow.' },
  { href: '/methods', label: 'Methods', number: '02', blurb: 'MoCoO module, IRALL exports, exclusions.' },
  { href: '/evidence', label: 'Evidence', number: '03', blurb: 'Batch metrics, ablations, GPU DROP.' },
  { href: '/claims', label: 'Claims', number: '04', blurb: 'Scope limits and refutation hooks.' },
] as const;
