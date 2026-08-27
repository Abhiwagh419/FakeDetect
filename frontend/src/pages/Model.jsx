import { useEffect, useState } from "react";
import { Cpu, Database, Target, Layers3 } from "lucide-react";
import Card from "../components/common/Card";
import KeyValueRow from "../components/common/KeyValueRow";
import Spinner from "../components/common/Spinner";
import { getModelInfo } from "../services/api";

export default function Model() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModelInfo().then(setInfo).catch((err) => setError(err.message));
  }, []);

  if (error) {
    return <p className="text-sm text-[var(--fake)] text-center py-16">{error}</p>;
  }

  if (!info) {
    return (
      <div className="flex justify-center py-16">
        <Spinner size={22} className="text-[var(--accent)]" />
      </div>
    );
  }

  const cm = info.confusion_matrix;
  const cr = info.classification_report;

  return (
    <div>
      <header className="mb-6">
        <h1 className="font-display font-bold text-2xl">Model Overview</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Architecture, training procedure, and evaluation metrics.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
        <Card title="Architecture" icon={Cpu}>
          <KeyValueRow label="Model" value={info.architecture} />
          <KeyValueRow label="Task" value={info.task} />
          <KeyValueRow label="Classes" value={info.classes.join(" / ")} />
          <KeyValueRow label="Inference Device" value={info.device} mono />
        </Card>

        <Card title="Dataset" icon={Database}>
          <KeyValueRow label="Dataset" value={info.dataset} />
          <KeyValueRow label="Training Samples" value={`${info.train_samples.toLocaleString()} (${info.train_real.toLocaleString()} REAL / ${info.train_fake.toLocaleString()} FAKE)`} />
          <KeyValueRow label="Test Samples" value={`${info.test_samples.toLocaleString()} (${info.test_real.toLocaleString()} REAL / ${info.test_fake.toLocaleString()} FAKE)`} />
          <KeyValueRow label="Epochs" value={info.epochs} />
          <KeyValueRow label="Training Hardware" value={info.training_hardware} />
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-5">
        <Card title="Best Validation Accuracy" icon={Target} className="text-center">
          <p className="font-display font-bold text-4xl text-[var(--accent-strong)] text-center py-4">
            {(info.best_val_accuracy * 100).toFixed(2)}%
          </p>
        </Card>
        <Card title="Test Accuracy" icon={Target} className="text-center">
          <p className="font-display font-bold text-4xl text-[var(--real)] text-center py-4">
            {(info.test_accuracy * 100).toFixed(2)}%
          </p>
        </Card>
        <Card title="Test Set Size" icon={Layers3} className="text-center">
          <p className="font-display font-bold text-4xl text-[var(--text)] text-center py-4">
            {info.test_samples.toLocaleString()}
          </p>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
        <Card title="Classification Report">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left border-b border-black/[0.07]">
                <th className="py-2 text-xs text-[var(--text-muted)] font-medium">Class</th>
                <th className="py-2 text-xs text-[var(--text-muted)] font-medium">Precision</th>
                <th className="py-2 text-xs text-[var(--text-muted)] font-medium">Recall</th>
                <th className="py-2 text-xs text-[var(--text-muted)] font-medium">F1</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(cr).map(([cls, m]) => (
                <tr key={cls} className="border-b border-black/[0.05] last:border-0">
                  <td className="py-2.5 font-medium">{cls}</td>
                  <td className="py-2.5 font-mono text-xs">{m.precision.toFixed(2)}</td>
                  <td className="py-2.5 font-mono text-xs">{m.recall.toFixed(2)}</td>
                  <td className="py-2.5 font-mono text-xs">{m.f1.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Confusion Matrix (Test Set)">
          <table className="w-full text-sm text-center">
            <thead>
              <tr>
                <th className="py-2 text-xs text-[var(--text-muted)]"></th>
                {cm.labels.map((l) => (
                  <th key={l} className="py-2 text-xs text-[var(--text-muted)] font-medium">Pred. {l}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cm.matrix.map((row, i) => (
                <tr key={cm.labels[i]}>
                  <td className="py-2.5 text-xs text-[var(--text-muted)] font-medium text-left">Actual {cm.labels[i]}</td>
                  {row.map((v, j) => (
                    <td
                      key={j}
                      className={`py-2.5 font-mono text-sm rounded-lg ${i === j ? "bg-[var(--real-dim)] text-[var(--real)]" : "bg-[var(--fake-dim)] text-[var(--fake)]"}`}
                    >
                      {v.toLocaleString()}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>

      <Card title="Transfer Learning Approach">
        <p className="text-sm text-[var(--text-muted)] leading-relaxed">
          The model starts from ImageNet-pretrained ResNet18 weights. Early convolutional layers are
          frozen to retain general low-level visual features; only <code className="font-mono text-[var(--accent-strong)]">layer4</code> and
          the final fully-connected classifier are fine-tuned on CIFAKE. This reduces overfitting risk
          on a moderately-sized dataset and trains substantially faster than fine-tuning the full network.
        </p>
      </Card>

      <div className="glass-panel p-5 mt-5">
        <p className="text-xs text-[var(--text-faint)] leading-relaxed">
          FakeDetect uses a machine-learning classifier trained and evaluated on the CIFAKE dataset.
          Its predictions are probabilistic and may not generalize perfectly to images generated by AI
          systems or image-processing pipelines outside the training distribution. Metadata and forensic
          indicators are supplementary signals and should not be treated as definitive proof of image origin.
        </p>
      </div>
    </div>
  );
}
