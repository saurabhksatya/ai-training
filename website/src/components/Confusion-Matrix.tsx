interface ConfusionMatrixProps {
  matrix?: number[][];
}

export default function ConfusionMatrix({ matrix }: ConfusionMatrixProps) {
  if (!matrix || matrix.length === 0) {
    return (
      <section className="col-span-12 lg:col-span-5 bg-surface-container rounded-xl border border-outline-variant overflow-hidden flex flex-col">
        <div className="p-md border-b border-outline-variant bg-surface-container-high flex justify-between items-center">
          <h3 className="font-headline-md text-headline-md">
            Confusion Matrix
          </h3>
        </div>
        <div className="p-md text-on-surface-variant">No data available</div>
      </section>
    );
  }

  const maxValue = Math.max(...matrix.flat());
  const size = matrix.length;

  const getCellColor = (value: number) => {
    if (value === 0)
      return "bg-surface-container-highest text-on-surface-variant";
    const intensity = value / maxValue;
    if (intensity > 0.5) return "bg-secondary text-on-secondary shadow-inner";
    if (intensity > 0.25) return "bg-error/40 text-error";
    return "bg-surface-container-highest text-on-surface-variant";
  };

  return (
    <section className="col-span-12 lg:col-span-5 bg-surface-container rounded-xl border border-outline-variant overflow-hidden flex flex-col">
      <div className="p-md border-b border-outline-variant bg-surface-container-high flex justify-between items-center">
        <h3 className="font-headline-md text-headline-md">Confusion Matrix</h3>
      </div>
      <div className="flex-grow p-xl flex items-center justify-center overflow-auto">
        <div className="flex flex-col gap-xs">
          {/* Column headers */}
          <div className="flex gap-xs">
            <div className="w-8 flex-shrink-0"></div>
            {Array.from({ length: size }).map((_, i) => (
              <div
                key={`col-header-${i}`}
                className="aspect-square w-8 flex-shrink-0 flex items-center justify-center text-xs font-bold text-on-surface-variant"
              >
                {i}
              </div>
            ))}
          </div>
          {/* Matrix rows with row indices */}
          {matrix.map((row, rowIndex) => (
            <div key={`row-${rowIndex}`} className="flex gap-xs">
              <div className="w-8 flex-shrink-0 flex items-center justify-center text-xs font-bold text-on-surface-variant">
                {rowIndex}
              </div>
              {row.map((value, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className={`w-8 h-8 flex-shrink-0 font-headline-md flex items-center justify-center rounded-lg matrix-cell transition-all text-xs ${getCellColor(value)}`}
                >
                  {value}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
