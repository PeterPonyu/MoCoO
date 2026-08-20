/**
 * MoCoO public package-index bindings (not a results site).
 */
export const SITE = {
  slug: 'MoCoO',
  navTitle: 'MoCoO',
  title: 'MoCoO — PyTorch count VAE package',
  kicker: 'Package mocoo · public code index',
  lead:
    'Installable PyTorch package: a count VAE with optional Momentum Contrast and optional Neural ODE heads for single-cell embeddings. This page describes the code. It is not a journal article and does not publish results.',
  physicalObject:
    'The public object is the mocoo package (PyPI and this GitHub repository).',
  primaryClaim:
    'This page is a package index only. It does not state article claims or report scores.',
  homepage: 'https://peterponyu.github.io/',
  scportal: 'https://peterponyu.github.io/scportal/',
  github: 'https://github.com/PeterPonyu/MoCoO',
  pypi: 'https://pypi.org/project/mocoo/',
  packageName: 'mocoo',
  packageVersion: '0.0.3',
  binding: 'mocoo-pypi-index',
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
    label: 'Index',
    href: 'https://peterponyu.github.io/MoCoO/',
    enabled: true,
  } satisfies BadgeConfig,
  archive: {
    label: 'Archive',
    enabled: false,
    disabledReason: 'No archive record on this page',
  } satisfies BadgeConfig,
  articleDoi: {
    label: 'Article DOI',
    enabled: false,
    disabledReason: 'No article DOI on this page',
  } satisfies BadgeConfig,
} as const;

/** Kept so old /methods /results /evidence /claims URLs stay 200. */
export const ROUTES = [
  { href: '/methods', label: 'Install', number: '01', blurb: 'pip install mocoo or a source checkout.' },
  { href: '/results', label: 'API', number: '02', blurb: 'Fit and export helpers on the package.' },
  { href: '/evidence', label: 'Scope', number: '03', blurb: 'This index does not publish scores.' },
  { href: '/claims', label: 'Limits', number: '04', blurb: 'No article claims or venue packaging.' },
] as const;
