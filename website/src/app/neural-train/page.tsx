"use client";

import Navbar from "@/components/Navbar";
import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  MdCloudUpload,
  MdPlayArrow,
  MdDownload,
  MdCheckCircle,
} from "react-icons/md";

interface EpochHistory {
  epoch: number;
  train_loss: number;
  val_loss: number | null;
  val_acc: number | null;
}

interface RunResults {
  history: EpochHistory[];
  predictions: Record<string, any>[];
  classes: string[] | null;
  features_count: number;
  samples_count: number;
  test_samples_count: number;
  task: string;
}

export default function NeuralTrainPage() {
  // Datasets
  const [trainFile, setTrainFile] = useState<File | null>(null);
  const [testFile, setTestFile] = useState<File | null>(null);
  const [trainHeaders, setTrainHeaders] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState<string>("");

  // Hyperparameters
  const [hiddenLayers, setHiddenLayers] = useState<string>("64, 32");
  const [activation, setActivation] = useState<string>("relu");
  const [dropout, setDropout] = useState<number>(0.0);
  const [batchNorm, setBatchNorm] = useState<boolean>(false);
  const [task, setTask] = useState<string>("classification");
  const [epochs, setEpochs] = useState<number>(20);
  const [batchSize, setBatchSize] = useState<number>(32);
  const [lr, setLr] = useState<number>(0.001);
  const [optimizer, setOptimizer] = useState<string>("adam");
  const [valSplit, setValSplit] = useState<number>(0.2);
  const [dropCols, setDropCols] = useState<string>("");

  // States
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<RunResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  // Pagination and Search
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [currentPage, setCurrentPage] = useState<number>(1);
  const itemsPerPage = 10;

  useEffect(() => {
    setMounted(true);
  }, []);

  // Parse headers when training file is uploaded
  const handleTrainFileChange = (file: File) => {
    setTrainFile(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const firstLine = text.split("\n")[0];
      if (!firstLine) return;
      const headers = firstLine
        .split(/[,;]/)
        .map((col) => col.trim().replace(/^['"]|['"]$/g, ""))
        .filter(Boolean);
      setTrainHeaders(headers);
      if (headers.length > 0) {
        setTargetColumn(headers[headers.length - 1]); // default to last column
      }
    };
    reader.readAsText(file);
  };

  const handleTestFileChange = (file: File) => {
    setTestFile(file);
  };

  const handleRunPipeline = async () => {
    if (!trainFile) {
      setError("Please upload a training dataset.");
      return;
    }
    if (!testFile) {
      setError("Please upload a testing/prediction dataset.");
      return;
    }
    if (!targetColumn) {
      setError("Please specify the target column.");
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);
    setCurrentPage(1);

    try {
      // Parse hidden layers string into array of integers
      const parsedLayers = hiddenLayers
        .split(",")
        .map((x) => parseInt(x.trim()))
        .filter((x) => !isNaN(x) && x > 0);

      if (parsedLayers.length === 0) {
        throw new Error(
          "Hidden layers must contain a comma-separated list of positive integers (e.g. 64, 32).",
        );
      }

      const parsedDropCols = dropCols
        .split(",")
        .map((x) => x.trim())
        .filter(Boolean);

      const config = {
        target_column: targetColumn,
        hidden_layers: parsedLayers,
        activation,
        dropout,
        batch_norm: batchNorm,
        task,
        epochs,
        batch_size: batchSize,
        lr,
        optimizer,
        loss: "none",
        val_split: valSplit,
        drop_cols: parsedDropCols,
      };

      const formData = new FormData();
      formData.append("config", JSON.stringify(config));
      formData.append("train_file", trainFile);
      formData.append("test_file", testFile);

      const response = await fetch("http://localhost:8000/neural-train", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(
          errData.detail || "Server error running training pipeline.",
        );
      }

      const data = await response.json();
      setResults(data);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPredictions = () => {
    if (!results || results.predictions.length === 0) return;
    const keys = Object.keys(results.predictions[0]);
    const csvContent = [
      keys.join(","),
      ...results.predictions.map((row: any) =>
        keys
          .map((key) => {
            const val = row[key];
            if (val === null || val === undefined) return "";
            const valStr = String(val);
            if (
              valStr.includes(",") ||
              valStr.includes('"') ||
              valStr.includes("\n")
            ) {
              return `"${valStr.replace(/"/g, '""')}"`;
            }
            return valStr;
          })
          .join(","),
      ),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `neural_network_predictions.csv`);
    link.style.visibility = "hidden";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Filter and Paginate predictions
  const filteredPredictions = results
    ? results.predictions.filter((row) =>
        Object.values(row).some((val) =>
          String(val).toLowerCase().includes(searchTerm.toLowerCase()),
        ),
      )
    : [];

  const totalPages = Math.ceil(filteredPredictions.length / itemsPerPage);
  const paginatedPredictions = filteredPredictions.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage,
  );

  // Table header keys: features, plus prediction & confidence
  const tableKeys =
    results && results.predictions.length > 0
      ? Object.keys(results.predictions[0]).filter(
          (k) => k !== "prediction" && k !== "confidence",
        )
      : [];

  return (
    <>
      <Navbar />
      <main className="max-w-[1440px] w-full mx-auto px-lg py-xl flex flex-col gap-lg grow">
        <div className="flex flex-col gap-xs mb-4">
          <h1 className="font-headline font-bold text-3xl text-on-surface">
            Neural Network Trainer
          </h1>
          <p className="text-on-surface-variant text-sm max-w-2xl">
            Upload training and testing datasets, design a Multi-Layer
            Perceptron (MLP) architecture, and predict test outcomes with
            confidence values using PyTorch in the backend.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-lg items-start">
          {/* Left Panel: Inputs and Configurations */}
          <section className="lg:col-span-4 flex flex-col gap-md glass-panel p-lg rounded-xl">
            <h2 className="font-headline font-semibold text-lg text-primary mb-2">
              Configuration
            </h2>

            {/* File Inputs */}
            <div className="flex flex-col gap-sm">
              {/* Training File Upload */}
              <div className="flex flex-col gap-xs">
                <label className="font-label-mono text-label-mono text-on-surface-variant">
                  TRAINING DATA (CSV)
                </label>
                <label className="flex flex-col items-center justify-center border border-dashed border-outline-variant hover:border-primary transition-colors rounded-lg p-sm cursor-pointer bg-surface-container-low text-center">
                  {trainFile ? (
                    <div className="flex items-center gap-xs text-primary font-medium text-sm">
                      <MdCheckCircle size={18} />
                      <span className="truncate max-w-[200px]">
                        {trainFile.name}
                      </span>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center text-on-surface-variant text-xs gap-1 py-1">
                      <MdCloudUpload size={24} />
                      <span>Select Training CSV</span>
                    </div>
                  )}
                  <input
                    type="file"
                    accept=".csv"
                    className="hidden"
                    onChange={(e) =>
                      e.target.files && handleTrainFileChange(e.target.files[0])
                    }
                  />
                </label>
              </div>

              {/* Testing File Upload */}
              <div className="flex flex-col gap-xs">
                <label className="font-label-mono text-label-mono text-on-surface-variant">
                  TESTING DATA (CSV)
                </label>
                <label className="flex flex-col items-center justify-center border border-dashed border-outline-variant hover:border-primary transition-colors rounded-lg p-sm cursor-pointer bg-surface-container-low text-center">
                  {testFile ? (
                    <div className="flex items-center gap-xs text-secondary font-medium text-sm">
                      <MdCheckCircle size={18} />
                      <span className="truncate max-w-[200px]">
                        {testFile.name}
                      </span>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center text-on-surface-variant text-xs gap-1 py-1">
                      <MdCloudUpload size={24} />
                      <span>Select Testing CSV</span>
                    </div>
                  )}
                  <input
                    type="file"
                    accept=".csv"
                    className="hidden"
                    onChange={(e) =>
                      e.target.files && handleTestFileChange(e.target.files[0])
                    }
                  />
                </label>
              </div>
            </div>

            {/* Target Column Selection */}
            <div className="flex flex-col gap-xs">
              <label className="font-label-mono text-label-mono text-on-surface-variant">
                TARGET VARIABLE
              </label>
              <select
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                disabled={trainHeaders.length === 0}
                className="bg-surface-container-low border border-outline-variant rounded-lg p-sm text-on-surface focus:border-primary text-sm"
              >
                {trainHeaders.length === 0 ? (
                  <option>Upload training dataset first...</option>
                ) : (
                  trainHeaders.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))
                )}
              </select>
            </div>

            {/* Columns to Drop */}
            <div className="flex flex-col gap-xs">
              <label className="font-label-mono text-label-mono text-on-surface-variant">
                COLUMNS TO DROP
              </label>
              <input
                type="text"
                value={dropCols}
                onChange={(e) => setDropCols(e.target.value)}
                placeholder="e.g. id, row_num"
                className="bg-surface-container-low border border-outline-variant rounded-lg p-sm text-on-surface text-sm focus:border-primary"
              />
              <span className="text-[10px] text-on-surface-variant">
                Comma-separated list of column names to exclude from features.
              </span>
            </div>

            <hr className="border-outline-variant my-1" />

            {/* Architecture Details */}
            <div className="grid grid-cols-2 gap-sm">
              <div className="flex flex-col gap-xs col-span-2">
                <label className="font-label-mono text-label-mono text-on-surface-variant">
                  HIDDEN LAYERS
                </label>
                <input
                  type="text"
                  value={hiddenLayers}
                  onChange={(e) => setHiddenLayers(e.target.value)}
                  placeholder="e.g. 64, 32"
                  className="bg-surface-container-low border border-outline-variant rounded-lg p-sm text-on-surface text-sm focus:border-primary"
                />
                <span className="text-[10px] text-on-surface-variant">
                  Comma-separated list of nodes in each hidden layer.
                </span>
              </div>

              <div className="flex flex-col gap-xs">
                <label className="font-label-mono text-label-mono text-on-surface-variant">
                  ACTIVATION
                </label>
                <select
                  value={activation}
                  onChange={(e) => setActivation(e.target.value)}
                  className="bg-surface-container-low border border-outline-variant rounded-lg p-sm text-on-surface text-sm focus:border-primary"
                >
                  <option value="relu">ReLU</option>
                  <option value="tanh">Tanh</option>
                  <option value="sigmoid">Sigmoid</option>
                  <option value="leaky_relu">Leaky ReLU</option>
                  <option value="gelu">GELU</option>
                </select>
              </div>

              <div className="flex flex-col gap-xs">
                <label className="font-label-mono text-label-mono text-on-surface-variant">
                  OPTIMIZER
                </label>
                <select
                  value={optimizer}
                  onChange={(e) => setOptimizer(e.target.value)}
                  className="bg-surface-container-low border border-outline-variant rounded-lg p-sm text-on-surface text-sm focus:border-primary"
                >
                  <option value="adam">Adam</option>
                  <option value="adamw">AdamW</option>
                  <option value="sgd">SGD</option>
                  <option value="rmsprop">RMSProp</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-sm">
              <div className="flex flex-col gap-xs">
                <label className="font-label-mono text-label-mono text-on-surface-variant">
                  TASK TYPE
                </label>
                <select
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  className="bg-surface-container-low border border-outline-variant rounded-lg p-sm text-on-surface text-sm focus:border-primary"
                >
                  <option value="classification">Classification</option>
                  <option value="regression">Regression</option>
                </select>
              </div>

              <div className="flex flex-col gap-xs">
                <label className="font-label-mono text-label-mono text-on-surface-variant">
                  BATCH SIZE
                </label>
                <select
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  className="bg-surface-container-low border border-outline-variant rounded-lg p-sm text-on-surface text-sm focus:border-primary"
                >
                  <option value={16}>16</option>
                  <option value={32}>32</option>
                  <option value={64}>64</option>
                  <option value={128}>128</option>
                </select>
              </div>
            </div>

            {/* Sliders */}
            <div className="flex flex-col gap-sm">
              <div className="flex flex-col gap-xs">
                <div className="flex justify-between text-xs">
                  <label className="font-label-mono text-label-mono text-on-surface-variant">
                    EPOCHS
                  </label>
                  <span className="text-primary font-bold">{epochs}</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="150"
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  className="custom-slider animate-pulse"
                />
              </div>

              <div className="flex flex-col gap-xs">
                <div className="flex justify-between text-xs">
                  <label className="font-label-mono text-label-mono text-on-surface-variant">
                    LEARNING RATE
                  </label>
                  <span className="text-primary font-bold">{lr}</span>
                </div>
                <input
                  type="range"
                  min="-4"
                  max="-1"
                  step="1"
                  value={Math.log10(lr)}
                  onChange={(e) => setLr(Math.pow(10, Number(e.target.value)))}
                  className="custom-slider"
                />
                <span className="text-[10px] text-on-surface-variant self-end">
                  Log scale: {lr.toFixed(4)}
                </span>
              </div>

              <div className="flex flex-col gap-xs">
                <div className="flex justify-between text-xs">
                  <label className="font-label-mono text-label-mono text-on-surface-variant">
                    DROPOUT RATE
                  </label>
                  <span className="text-primary font-bold">
                    {dropout.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="0.8"
                  step="0.05"
                  value={dropout}
                  onChange={(e) => setDropout(Number(e.target.value))}
                  className="custom-slider"
                />
              </div>

              <div className="flex flex-col gap-xs">
                <div className="flex justify-between text-xs">
                  <label className="font-label-mono text-label-mono text-on-surface-variant">
                    VALIDATION SPLIT
                  </label>
                  <span className="text-primary font-bold">
                    {(valSplit * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="0.4"
                  step="0.05"
                  value={valSplit}
                  onChange={(e) => setValSplit(Number(e.target.value))}
                  className="custom-slider"
                />
              </div>
            </div>

            <div className="flex items-center gap-sm mt-1">
              <input
                type="checkbox"
                id="batchNorm"
                checked={batchNorm}
                onChange={(e) => setBatchNorm(e.target.checked)}
                className="h-4 w-4 rounded border text-primary bg-surface-container focus:ring-0 cursor-pointer appearance-none checked:appearance-auto"
              />
              <label
                htmlFor="batchNorm"
                className="text-sm font-medium text-on-surface cursor-pointer select-none"
              >
                Use Batch Normalization
              </label>
            </div>

            <button
              onClick={handleRunPipeline}
              disabled={loading || !trainFile || !testFile}
              className="mt-4 flex items-center justify-center gap-xs w-full py-sm bg-primary hover:bg-primary/90 text-on-primary-fixed font-bold rounded-lg transition-all shadow-lg hover:shadow-primary/20 disabled:opacity-50 disabled:cursor-not-allowed group cursor-pointer"
            >
              <MdPlayArrow
                size={20}
                className="group-hover:translate-x-0.5 transition-transform"
              />
              <span className="font-bold font-label-mono">
                {loading ? "TRAINING NETWORK..." : "TRAIN AND PREDICT"}
              </span>
            </button>
          </section>

          {/* Right Panel: Charts and Reports */}
          <section className="lg:col-span-8 flex flex-col gap-lg min-h-[500px]">
            {error && (
              <div className="bg-error-container/20 border border-error/30 text-error p-md rounded-xl text-sm">
                <strong>Error:</strong> {error}
              </div>
            )}

            {loading && (
              <div className="glass-panel rounded-xl flex flex-col items-center justify-center p-xl grow animate-pulse">
                <div className="relative w-16 h-16 mb-4">
                  <div className="absolute top-0 left-0 w-full h-full border-4 border-primary/20 rounded-full"></div>
                  <div className="absolute top-0 left-0 w-full h-full border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
                </div>
                <h3 className="font-headline font-semibold text-lg text-on-surface">
                  Training in Progress
                </h3>
                <p className="text-on-surface-variant text-sm mt-1">
                  Feeding data to PyTorch MLP, updating weights, and calculating
                  predictions...
                </p>
              </div>
            )}

            {!loading && !results && (
              <div className="glass-panel rounded-xl border border-outline-variant flex flex-col items-center justify-center p-xl text-center grow opacity-80">
                <h3 className="font-headline font-semibold text-lg text-on-surface mb-xs">
                  Dashboard Ready
                </h3>
                <p className="text-on-surface-variant text-sm">
                  Upload your training/testing CSV datasets, configure
                  hyperparameters in the settings panel, and trigger training to
                  run.
                </p>
              </div>
            )}

            {!loading && results && (
              <div className="flex flex-col gap-lg animate-fadeIn">
                {/* Stats Summary cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-md">
                  <div className="bg-surface-container rounded-xl p-md border border-outline-variant flex flex-col gap-xs">
                    <span className="font-label-mono text-[10px] text-on-surface-variant uppercase">
                      TRAINING SAMPLES
                    </span>
                    <span className="text-xl font-bold text-on-surface">
                      {results.samples_count}
                    </span>
                  </div>
                  <div className="bg-surface-container rounded-xl p-md border border-outline-variant flex flex-col gap-xs">
                    <span className="font-label-mono text-[10px] text-on-surface-variant uppercase">
                      INPUT FEATURES
                    </span>
                    <span className="text-xl font-bold text-on-surface">
                      {results.features_count}
                    </span>
                  </div>
                  <div className="bg-surface-container rounded-xl p-md border border-outline-variant flex flex-col gap-xs">
                    <span className="font-label-mono text-[10px] text-on-surface-variant uppercase">
                      FINAL LOSS (TRAIN / VAL)
                    </span>
                    <span className="text-xl font-bold text-on-surface">
                      {results.history[
                        results.history.length - 1
                      ].train_loss.toFixed(4)}
                      {results.history[results.history.length - 1].val_loss !==
                      null
                        ? ` / ${results.history[results.history.length - 1].val_loss?.toFixed(4)}`
                        : " / --"}
                    </span>
                  </div>
                  <div className="bg-surface-container rounded-xl p-md border border-outline-variant flex flex-col gap-xs">
                    <span className="font-label-mono text-[10px] text-on-surface-variant uppercase">
                      {results.task === "classification"
                        ? "VAL ACCURACY"
                        : "VAL EVALUATION"}
                    </span>
                    <span className="text-xl font-bold text-secondary">
                      {results.task === "classification" &&
                      results.history[results.history.length - 1].val_acc !==
                        null
                        ? `${(results.history[results.history.length - 1].val_acc! * 100).toFixed(1)}%`
                        : "N/A (Val is 0%)"}
                    </span>
                  </div>
                </div>

                {/* Recharts Loss curve */}
                <div className="bg-surface-container rounded-xl p-lg border border-outline-variant">
                  <div className="flex items-center justify-between mb-sm">
                    <h3 className="font-headline font-semibold text-on-surface text-base">
                      Epoch Loss Curve
                    </h3>
                    <span className="font-label-mono text-xs text-on-surface-variant">
                      Train Loss vs Validation Loss
                    </span>
                  </div>
                  <div className="h-64 w-full">
                    {mounted ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={results.history}>
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="#464554"
                            opacity={0.3}
                          />
                          <XAxis
                            dataKey="epoch"
                            stroke="#dae2fd"
                            tick={{ fill: "#c7c4d7", fontSize: 11 }}
                          />
                          <YAxis
                            stroke="#dae2fd"
                            tick={{ fill: "#c7c4d7", fontSize: 11 }}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#171f33",
                              borderColor: "#464554",
                              borderRadius: "0.5rem",
                              color: "#dae2fd",
                            }}
                          />
                          <Legend verticalAlign="top" height={36} />
                          <Line
                            name="Train Loss"
                            type="monotone"
                            dataKey="train_loss"
                            stroke="#c0c1ff"
                            strokeWidth={2}
                            dot={false}
                          />
                          {valSplit > 0 && (
                            <Line
                              name="Val Loss"
                              type="monotone"
                              dataKey="val_loss"
                              stroke="#4fdbc8"
                              strokeWidth={2}
                              dot={false}
                            />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex items-center justify-center h-full text-on-surface-variant text-sm">
                        Loading charts...
                      </div>
                    )}
                  </div>
                </div>

                {/* Predictions results Section */}
                <div className="bg-surface-container rounded-xl p-lg border border-outline-variant flex flex-col gap-md">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-sm">
                    <div className="flex flex-col gap-1">
                      <h3 className="font-headline font-semibold text-on-surface text-base">
                        Test Predictions
                      </h3>
                      <p className="text-on-surface-variant text-xs">
                        Showing predictions and confidence outputs generated by
                        the MLP model.
                      </p>
                    </div>

                    <div className="flex items-center gap-md">
                      <input
                        type="text"
                        placeholder="Search predictions..."
                        value={searchTerm}
                        onChange={(e) => {
                          setSearchTerm(e.target.value);
                          setCurrentPage(1);
                        }}
                        className="bg-surface-container-low border border-outline-variant rounded-lg px-sm py-1.5 text-xs text-on-surface focus:border-primary max-w-[200px]"
                      />
                      <button
                        onClick={handleDownloadPredictions}
                        className="flex items-center gap-xs px-sm py-1.5 bg-surface-container-high border border-outline-variant hover:border-primary transition-all text-xs font-semibold rounded-lg text-primary hover:text-on-surface cursor-pointer"
                      >
                        <MdDownload size={14} />
                        <span>Download CSV</span>
                      </button>
                    </div>
                  </div>

                  {/* Predictions Table */}
                  <div className="overflow-x-auto rounded-lg border border-outline-variant bg-surface-container-low">
                    <table className="w-full border-collapse text-left text-xs">
                      <thead className="bg-surface-container-high border-b border-outline-variant font-label-mono text-[10px] text-on-surface-variant uppercase">
                        <tr>
                          {tableKeys.slice(0, 4).map((key) => (
                            <th key={key} className="px-md py-sm font-semibold">
                              {key}
                            </th>
                          ))}
                          {tableKeys.length > 4 && (
                            <th className="px-md py-sm font-semibold">...</th>
                          )}
                          <th className="px-md py-sm font-semibold text-primary">
                            Prediction
                          </th>
                          {results.task === "classification" && (
                            <th className="px-md py-sm font-semibold text-secondary">
                              Confidence
                            </th>
                          )}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-outline-variant/30 text-on-surface">
                        {paginatedPredictions.map((row, idx) => (
                          <tr
                            key={idx}
                            className="hover:bg-surface-container-highest/20 transition-colors"
                          >
                            {tableKeys.slice(0, 4).map((key) => (
                              <td
                                key={key}
                                className="px-md py-sm max-w-[150px] truncate"
                              >
                                {row[key] !== null ? String(row[key]) : "--"}
                              </td>
                            ))}
                            {tableKeys.length > 4 && (
                              <td className="px-md py-sm text-on-surface-variant">
                                ...
                              </td>
                            )}
                            <td className="px-md py-sm font-semibold text-primary-container bg-primary/5">
                              {typeof row.prediction === "number" &&
                              results.task === "regression"
                                ? row.prediction.toFixed(4)
                                : String(row.prediction)}
                            </td>
                            {results.task === "classification" && (
                              <td className="px-md py-sm font-semibold text-secondary bg-secondary/5">
                                {row.confidence !== null &&
                                row.confidence !== undefined
                                  ? `${(row.confidence * 100).toFixed(1)}%`
                                  : "--"}
                              </td>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Pagination controls */}
                  {totalPages > 1 && (
                    <div className="flex items-center justify-between text-xs text-on-surface-variant mt-xs">
                      <span>
                        Showing {(currentPage - 1) * itemsPerPage + 1} to{" "}
                        {Math.min(
                          currentPage * itemsPerPage,
                          filteredPredictions.length,
                        )}{" "}
                        of {filteredPredictions.length} predictions
                      </span>
                      <div className="flex items-center gap-xs">
                        <button
                          onClick={() =>
                            setCurrentPage((c) => Math.max(c - 1, 1))
                          }
                          disabled={currentPage === 1}
                          className="px-sm py-1 bg-surface-container-high rounded border border-outline-variant hover:border-primary disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer text-[11px]"
                        >
                          Prev
                        </button>
                        <span>
                          {currentPage} / {totalPages}
                        </span>
                        <button
                          onClick={() =>
                            setCurrentPage((c) => Math.min(c + 1, totalPages))
                          }
                          disabled={currentPage === totalPages}
                          className="px-sm py-1 bg-surface-container-high rounded border border-outline-variant hover:border-primary disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer text-[11px]"
                        >
                          Next
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </section>
        </div>
      </main>
    </>
  );
}
