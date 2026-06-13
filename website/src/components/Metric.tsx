interface MetricsProps {
  accuracy?: number;
  cvScore?: number;
}

export default function Metrics({
  accuracy = 0.9667,
  cvScore = 0.9524,
}: MetricsProps) {
  const accuracyPercent = (accuracy * 100).toFixed(2);
  const cvPercent = (cvScore * 100).toFixed(2);

  return (
    <section className="col-span-12 lg:col-span-4 flex flex-col gap-gutter">
      <div className="bg-surface-container rounded-xl p-lg border border-outline-variant flex flex-col gap-xs">
        <span className="font-label-mono text-label-mono text-on-surface-variant uppercase">
          Accuracy Score
        </span>
        <div className="flex items-baseline gap-xs">
          <span className="font-headline-xl text-headline-xl text-primary font-bold">
            {accuracy.toFixed(4)}
          </span>
        </div>
        <div className="mt-base h-1 w-full bg-surface-variant rounded-full overflow-hidden">
          <div
            className="h-full bg-primary"
            style={{ width: `${accuracyPercent}%` }}
          ></div>
        </div>
      </div>
      <div className="bg-surface-container rounded-xl p-lg border border-outline-variant flex flex-col gap-xs">
        <span className="font-label-mono text-label-mono text-on-surface-variant uppercase">
          Cross-Validation
        </span>
        <div className="flex items-baseline gap-xs">
          <span className="font-headline-xl text-headline-xl text-secondary font-bold">
            {cvScore.toFixed(4)}
          </span>
        </div>
        <div className="mt-base h-1 w-full bg-surface-variant rounded-full overflow-hidden">
          <div
            className="h-full bg-secondary"
            style={{ width: `${cvPercent}%` }}
          ></div>
        </div>
      </div>
    </section>
  );
}
