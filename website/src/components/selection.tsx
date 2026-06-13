"use client";

import { useEffect, useState } from "react";
import { datasets, modelOptions } from "@/schema/models1";

interface SelectionProps {
  selectedDataset: string;
  onDatasetChange: (value: string) => void;
  selectedTarget: string;
  onTargetChange: (value: string) => void;
  selectedModels: string[];
  onModelChange: (value: string) => void;
  rfTrees: number;
  onRfTreesChange: (value: number) => void;
  maxDepth: number;
  onMaxDepthChange: (value: number) => void;
  knnK: number;
  onKnnKChange: (value: number) => void;
}

export default function Selection({
  selectedDataset,
  onDatasetChange,
  selectedTarget,
  onTargetChange,
  selectedModels,
  onModelChange,
  rfTrees,
  onRfTreesChange,
  maxDepth,
  onMaxDepthChange,
  knnK,
  onKnnKChange,
}: SelectionProps) {
  const [targetOptions, setTargetOptions] = useState<string[]>([]);

  useEffect(() => {
    const loadHeaders = async () => {
      const dataset = datasets.find((item) => item.id === selectedDataset);
      if (!dataset) return;

      try {
        const response = await fetch(dataset.file);
        if (!response.ok) {
          setTargetOptions([]);
          return;
        }

        const text = await response.text();
        const headerLine = text.split("\n")[0];
        const headers = headerLine
          .split(/[,;]/)
          .map((column) => column.trim().replace(/^['"]|['"]$/g, ""))
          .filter(Boolean);

        setTargetOptions(headers);
        if (!headers.includes(selectedTarget) && headers[0]) {
          onTargetChange(headers[0]);
        }
      } catch {
        setTargetOptions([]);
      }
    };

    loadHeaders();
  }, [selectedDataset]);

  const handleModelChange = (value: string) => {
    onModelChange(value);
  };

  return (
    <section className="glass-panel p-lg rounded-xl gap-xl space-y-6 items-end">
      <div className="flex flex-row gap-xs ">
        <div className="flex flex-col gap-xs grow">
          <label className="font-label-mono text-label-mono text-on-surface-variant">
            SELECT DATASET
          </label>
          <select
            value={selectedDataset}
            onChange={(event) => onDatasetChange(event.target.value)}
            className="bg-surface-container-low border border-outline-variant rounded-lg p-sm focus:border-primary focus:ring-0 text-on-surface"
          >
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.label}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-xs grow">
          <label className="font-label-mono text-label-mono text-on-surface-variant">
            TARGET VARIABLE
          </label>
          <select
            value={selectedTarget}
            onChange={(event) => onTargetChange(event.target.value)}
            className="bg-surface-container-low border border-outline-variant rounded-lg p-sm focus:border-primary focus:ring-0 text-on-surface"
          >
            {targetOptions.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="col-span-1 md:col-span-2 grid grid-cols-1 gap-md">
        <div className="flex flex-row gap-xs">
          {modelOptions.map((model) => (
            <label
              key={model.value}
              className="flex items-center gap-xs rounded-md px-sm py-2 hover:bg-surface-hover cursor-pointer grow"
            >
              <input
                type="checkbox"
                checked={selectedModels.includes(model.value)}
                onChange={() => handleModelChange(model.value)}
                className="h-4 w-4 rounded border appearance-none checked:appearance-auto not-checked:bg-white text-primary focus:ring-primary"
              />
              <span>{model.label}</span>
            </label>
          ))}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-md">
          <div className="flex flex-col gap-xs">
            <div className="flex justify-between items-center">
              <label className="font-label-mono text-label-mono text-on-surface-variant">
                RF TREES
              </label>
              <span className="font-label-mono text-primary" id="rf-trees-val">
                {rfTrees}
              </span>
            </div>
            <input
              className="custom-slider"
              max="200"
              min="1"
              type="range"
              value={rfTrees}
              onChange={(event) => onRfTreesChange(Number(event.target.value))}
            />
          </div>
          <div className="flex flex-col gap-xs">
            <div className="flex justify-between items-center">
              <label className="font-label-mono text-label-mono text-on-surface-variant">
                MAX DEPTH
              </label>
              <span className="font-label-mono text-primary" id="max-depth-val">
                {maxDepth}
              </span>
            </div>
            <input
              className="custom-slider"
              max="20"
              min="1"
              type="range"
              value={maxDepth}
              onChange={(event) => onMaxDepthChange(Number(event.target.value))}
            />
          </div>
          <div className="flex flex-col gap-xs">
            <div className="flex justify-between items-center">
              <label className="font-label-mono text-label-mono text-on-surface-variant">
                KNN K
              </label>
              <span className="font-label-mono text-primary" id="knn-val">
                {knnK}
              </span>
            </div>
            <input
              className="custom-slider"
              max="15"
              min="1"
              type="range"
              value={knnK}
              onChange={(event) => onKnnKChange(Number(event.target.value))}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
