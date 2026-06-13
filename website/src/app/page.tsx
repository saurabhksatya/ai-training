"use client";

import ModelAnswer from "@/components/model-answer";
import Navbar from "@/components/Navbar";
import Selection from "@/components/selection";
import { modelOptions } from "@/schema/models1";
import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface ModelResultSummary {
  accuracy: number;
  cv_score: number;
}

export default function Home() {
  const [selectedDataset, setSelectedDataset] = useState("iris");
  const [selectedTarget, setSelectedTarget] = useState("");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [rfTrees, setRfTrees] = useState(100);
  const [maxDepth, setMaxDepth] = useState(10);
  const [knnK, setKnnK] = useState(5);
  const [modelResults, setModelResults] = useState<
    Record<string, ModelResultSummary>
  >({});
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    setModelResults((current) =>
      Object.fromEntries(
        Object.entries(current).filter(([modelName]) =>
          selectedModels.includes(modelName),
        ),
      ),
    );
  }, [selectedModels]);

  const chartData = selectedModels.map((modelName) => {
    const result = modelResults[modelName];
    const label =
      modelOptions.find((model) => model.value === modelName)?.label ??
      modelName;
    return {
      name: label,
      Accuracy: result ? Math.round(result.accuracy * 1000) / 10 : 0,
      "Cross-validation": result ? Math.round(result.cv_score * 1000) / 10 : 0,
    };
  });

  const maxVal = chartData.length > 0
    ? Math.max(...chartData.flatMap((d) => [d.Accuracy, d["Cross-validation"]]))
    : 100;
  const yAxisMax = maxVal > 0 ? maxVal : 100;

  const handleResultUpdate = (
    modelName: string,
    result: ModelResultSummary | null,
  ) => {
    setModelResults((current) => {
      const next = { ...current };
      if (result) {
        next[modelName] = result;
      } else {
        delete next[modelName];
      }
      return next;
    });
  };

  return (
    <>
      <Navbar />
      <main className="max-w-[1440px] flex flex-col gap-xl p-xl w-screen mx-auto">
        <Selection
          selectedDataset={selectedDataset}
          onDatasetChange={(value) => {
            setSelectedTarget("");
            setSelectedModels([]);
            setRfTrees(100);
            setMaxDepth(10);
            setKnnK(5);
            setModelResults({});
            setSelectedDataset(value);
          }}
          selectedTarget={selectedTarget}
          onTargetChange={(value) => {
            setSelectedModels([]);
            setRfTrees(100);
            setMaxDepth(10);
            setKnnK(5);
            setModelResults({});
            setSelectedTarget(value);
          }}
          selectedModels={selectedModels}
          onModelChange={(value) =>
            setSelectedModels((current) =>
              current.includes(value)
                ? current.filter((item) => item !== value)
                : [...current, value],
            )
          }
          rfTrees={rfTrees}
          onRfTreesChange={setRfTrees}
          maxDepth={maxDepth}
          onMaxDepthChange={setMaxDepth}
          knnK={knnK}
          onKnnKChange={setKnnK}
        />
        {selectedModels.length > 0 && (
          <section className="glass-panel rounded-xl p-lg space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="font-heading-lg text-on-surface">
                Model performance
              </h2>
              <span className="font-label-mono text-label-mono text-on-surface-variant">
                Accuracy & CV histogram
              </span>
            </div>
            <div className="rounded-xl border border-outline-variant p-lg bg-surface-container">
              {mounted ? (
                <div className="h-80 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={chartData}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#464554" opacity={0.3} />
                      <XAxis
                        dataKey="name"
                        stroke="#dae2fd"
                        tick={{ fill: "#c7c4d7", fontSize: 12 }}
                      />
                      <YAxis
                        stroke="#dae2fd"
                        tick={{ fill: "#c7c4d7", fontSize: 12 }}
                        tickFormatter={(value) => `${value}%`}
                        domain={[0, yAxisMax]}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#171f33",
                          borderColor: "#464554",
                          borderRadius: "0.5rem",
                          color: "#dae2fd",
                        }}
                        itemStyle={{ color: "#dae2fd" }}
                        labelStyle={{ color: "#dae2fd", fontWeight: "bold" }}
                        formatter={(value: any) => [`${value}%`]}
                      />
                      <Legend
                        verticalAlign="top"
                        height={36}
                        iconType="square"
                        formatter={(value) => (
                          <span className="text-sm text-on-surface-variant ml-1 mr-4">{value}</span>
                        )}
                      />
                      <Bar dataKey="Accuracy" fill="var(--color-primary, #c0c1ff)" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Cross-validation" fill="var(--color-secondary, #4fdbc8)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-80 w-full flex items-center justify-center text-on-surface-variant">
                  Loading chart...
                </div>
              )}
            </div>
          </section>
        )}
        {selectedModels.map((modelName) => (
          <ModelAnswer
            key={modelName}
            modelName={modelName}
            selectedDataset={selectedDataset}
            selectedTarget={selectedTarget}
            rfTrees={rfTrees}
            maxDepth={maxDepth}
            knnK={knnK}
            onResultUpdate={handleResultUpdate}
          />
        ))}
      </main>
    </>
  );
}
