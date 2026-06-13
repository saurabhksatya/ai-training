interface ClassificationReportProps {
  report?: Record<string, any>;
}

export default function ClassificationReport({
  report,
}: ClassificationReportProps) {
  if (!report) {
    return (
      <section className="col-span-12 lg:col-span-7 bg-surface-container rounded-xl border border-outline-variant overflow-hidden">
        <div className="p-md border-b border-outline-variant bg-surface-container-high">
          <h3 className="font-headline-md text-headline-md">
            Classification Report
          </h3>
        </div>
        <div className="p-md text-on-surface-variant">No data available</div>
      </section>
    );
  }

  // Filter out non-class metrics (accuracy, macro avg, weighted avg)
  const classMetrics = Object.entries(report)
    .filter(([key]) => !["accuracy", "macro avg", "weighted avg"].includes(key))
    .slice(0, -3); // Remove the last 3 summary rows

  return (
    <section className="col-span-12 lg:col-span-7 bg-surface-container rounded-xl border border-outline-variant overflow-hidden">
      <div className="p-md border-b border-outline-variant bg-surface-container-high">
        <h3 className="font-headline-md text-headline-md">
          Classification Report
        </h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-right border-collapse">
          <thead>
            <tr className="bg-surface-container-highest">
              <th className="p-md text-left font-label-mono text-label-mono text-on-surface-variant border-b border-outline-variant">
                CLASS
              </th>
              <th className="p-md font-label-mono text-label-mono text-on-surface-variant border-b border-outline-variant">
                PRECISION
              </th>
              <th className="p-md font-label-mono text-label-mono text-on-surface-variant border-b border-outline-variant">
                RECALL
              </th>
              <th className="p-md font-label-mono text-label-mono text-on-surface-variant border-b border-outline-variant">
                F1-SCORE
              </th>
              <th className="p-md font-label-mono text-label-mono text-on-surface-variant border-b border-outline-variant">
                SUPPORT
              </th>
            </tr>
          </thead>
          <tbody className="font-data-tabular text-data-tabular">
            {classMetrics.map(([className, metrics]) => (
              <tr
                key={className}
                className="border-b border-outline-variant/30"
              >
                <td className="p-md text-left text-on-surface-variant font-bold">
                  {className}
                </td>
                <td className="p-md">
                  {typeof metrics === "object" && metrics.precision
                    ? metrics.precision.toFixed(2)
                    : "0"}
                </td>
                <td className="p-md">
                  {typeof metrics === "object" && metrics.recall
                    ? metrics.recall.toFixed(2)
                    : "0"}
                </td>
                <td className="p-md">
                  {typeof metrics === "object" && metrics["f1-score"]
                    ? metrics["f1-score"].toFixed(2)
                    : "0"}
                </td>
                <td className="p-md">
                  {typeof metrics === "object" && metrics.support
                    ? metrics.support.toFixed(0)
                    : "0"}
                </td>
              </tr>
            ))}
            {report["weighted avg"] && (
              <tr className="bg-surface-variant/10">
                <td className="p-md text-left text-primary font-bold">
                  Weighted Avg
                </td>
                <td className="p-md">
                  {report["weighted avg"].precision.toFixed(2)}
                </td>
                <td className="p-md">
                  {report["weighted avg"].recall.toFixed(2)}
                </td>
                <td className="p-md">
                  {report["weighted avg"]["f1-score"].toFixed(2)}
                </td>
                <td className="p-md">
                  {report["weighted avg"].support.toFixed(0)}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
