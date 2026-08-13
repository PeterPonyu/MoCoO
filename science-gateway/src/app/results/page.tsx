import PageShell from '@/components/PageShell';
import FigurePanel from '@/components/FigurePanel';

const FIGURES = [
  { file: 'F07_temporal_audit.png', kicker: 'Fig. 7 · batch axis', caption: 'Temporal audit: PC1 vs batch day. Physical axis on this leaf — not cell-state proof.' },
  { file: 'F01_architecture.png', kicker: 'Fig. 1 · architecture', caption: 'MoCoO module architecture and training pipeline overview.' },
  { file: 'F02_canonical_irall.png', kicker: 'Fig. 2 · IRALL', caption: 'Canonical IRALL embedding exports — frozen representations, not new biology.' },
  { file: 'F03_seed_stability.png', kicker: 'Fig. 3 · seed stability', caption: 'Multi-seed stability diagnostics. Not independent biological validation.' },
  { file: 'F04_training_dynamics.png', kicker: 'Fig. 4 · training', caption: 'Training dynamics and loss curves.' },
  { file: 'F05_component_ablation.png', kicker: 'Fig. 5 · ablation', caption: 'Component ablation summary.' },
  { file: 'F06_trajectory_diagnostics.png', kicker: 'Fig. 6 · trajectory', caption: 'Trajectory diagnostics — embedding movement, not cell-type change.' },
  { file: 'F08_pca_floor.png', kicker: 'Fig. 8 · PCA floor', caption: 'PCA k-means baseline floor comparison.' },
] as const;

export default function ResultsPage() {
  return (
    <PageShell title="Results" kicker="Outcome figures">
      <p>
        Fig. 7 (batch axis) is listed first among physical results. Remaining floats are tagged
        embedding or metrics — not cell-state atlas proof.
      </p>

      <div className="grid gap-6">
        {FIGURES.map((fig) => (
          <FigurePanel
            key={fig.file}
            src={`/figures/${fig.file}`}
            alt={fig.kicker}
            kicker={fig.kicker}
            caption={fig.caption}
          />
        ))}
      </div>
    </PageShell>
  );
}
