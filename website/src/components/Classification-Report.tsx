export default function ClassificationReport() {
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
            <tr className="border-b border-outline-variant/30">
              <td className="p-md text-left text-on-surface-variant font-bold">
                Setosa
              </td>
              <td className="p-md">1.00</td>
              <td className="p-md">1.00</td>
              <td className="p-md">1.00</td>
              <td className="p-md">10</td>
            </tr>
            <tr className="border-b border-outline-variant/30">
              <td className="p-md text-left text-on-surface-variant font-bold">
                Versicolor
              </td>
              <td className="p-md">0.92</td>
              <td className="p-md">1.00</td>
              <td className="p-md">0.96</td>
              <td className="p-md">12</td>
            </tr>
            <tr className="border-b border-outline-variant/30">
              <td className="p-md text-left text-on-surface-variant font-bold">
                Virginica
              </td>
              <td className="p-md">1.00</td>
              <td className="p-md">0.88</td>
              <td className="p-md">0.93</td>
              <td className="p-md">8</td>
            </tr>
            <tr className="bg-surface-variant/10">
              <td className="p-md text-left text-primary font-bold">
                Weighted Avg
              </td>
              <td className="p-md">0.97</td>
              <td className="p-md">0.97</td>
              <td className="p-md">0.97</td>
              <td className="p-md">30</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}
