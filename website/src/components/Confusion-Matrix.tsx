export default function ConfusionMatrix() {
  return (
    <section className="col-span-12 lg:col-span-5 bg-surface-container rounded-xl border border-outline-variant overflow-hidden flex flex-col">
      <div className="p-md border-b border-outline-variant bg-surface-container-high flex justify-between items-center">
        <h3 className="font-headline-md text-headline-md">Confusion Matrix</h3>
        <span className="material-symbols-outlined text-on-surface-variant cursor-pointer hover:text-primary">
          info
        </span>
      </div>
      <div className="flex-grow p-xl flex items-center justify-center">
        <div className="grid grid-cols-3 gap-xs w-full max-w-[320px]">
          <div className="aspect-square bg-secondary text-on-secondary font-headline-md flex items-center justify-center rounded-lg shadow-inner matrix-cell transition-all">
            10
          </div>
          <div className="aspect-square bg-surface-container-highest text-on-surface-variant font-headline-md flex items-center justify-center rounded-lg matrix-cell transition-all">
            0
          </div>
          <div className="aspect-square bg-surface-container-highest text-on-surface-variant font-headline-md flex items-center justify-center rounded-lg matrix-cell transition-all">
            0
          </div>

          <div className="aspect-square bg-surface-container-highest text-on-surface-variant font-headline-md flex items-center justify-center rounded-lg matrix-cell transition-all">
            0
          </div>
          <div className="aspect-square bg-secondary text-on-secondary font-headline-md flex items-center justify-center rounded-lg shadow-inner matrix-cell transition-all">
            12
          </div>
          <div className="aspect-square bg-surface-container-highest text-on-surface-variant font-headline-md flex items-center justify-center rounded-lg matrix-cell transition-all">
            0
          </div>

          <div className="aspect-square bg-surface-container-highest text-on-surface-variant font-headline-md flex items-center justify-center rounded-lg matrix-cell transition-all">
            0
          </div>
          <div className="aspect-square bg-error/40 text-error font-headline-md flex items-center justify-center rounded-lg matrix-cell transition-all">
            1
          </div>
          <div className="aspect-square bg-secondary text-on-secondary font-headline-md flex items-center justify-center rounded-lg shadow-inner matrix-cell transition-all">
            7
          </div>
        </div>
      </div>
      <div className="p-md bg-surface-container-low border-t border-outline-variant flex justify-around font-label-mono text-[10px] uppercase text-on-surface-variant">
        <span>Setosa</span>
        <span>Versicolor</span>
        <span>Virginica</span>
      </div>
    </section>
  );
}
