"use client";

import Navbar from "@/components/Navbar";
import { useState, useEffect, useRef } from "react";
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
  MdPlayArrow,
  MdDownload,
  MdCancel,
  MdAdd,
  MdDelete,
  MdScience,
  MdSettings,
  MdVideogameAsset,
  MdStop,
  MdTerminal,
} from "react-icons/md";

interface ProgressData {
  episode: number;
  reward: number;
  avg_reward: number;
  loss: number;
  epsilon: number;
}

interface RewardRule {
  id: string;
  type: string;
  value?: number;
  index?: number;
  op?: string;
  reward?: number;
  action?: number;
}

export default function RLTrainPage() {
  // Config states
  const [env, setEnv] = useState<string>("CartPole-v1");
  const [customEnv, setCustomEnv] = useState<string>("");
  const [episodes, setEpisodes] = useState<number>(200);
  const [maxSteps, setMaxSteps] = useState<number>(500);
  const [lr, setLr] = useState<number>(0.001);
  const [gamma, setGamma] = useState<number>(0.99);
  const [batchSize, setBatchSize] = useState<number>(64);
  const [bufferSize, setBufferSize] = useState<number>(10000);
  const [minBufferSize, setMinBufferSize] = useState<number>(1000);
  const [targetUpdateFreq, setTargetUpdateFreq] = useState<number>(10);
  const [hiddenLayers, setHiddenLayers] = useState<string>("128, 128");
  const [activation, setActivation] = useState<string>("relu");
  const [epsilonStart, setEpsilonStart] = useState<number>(1.0);
  const [epsilonEnd, setEpsilonEnd] = useState<number>(0.05);
  const [epsilonDecay, setEpsilonDecay] = useState<number>(0.995);
  const [failRewardThreshold, setFailRewardThreshold] = useState<number>(0.0);

  // Reward rules JSON config state
  const [rewardRulesJson, setRewardRulesJson] = useState<string>(
    JSON.stringify(
      [
        {
          type: "step_penalty",
          value: -0.01,
        },
      ],
      null,
      2,
    ),
  );

  const handleDownloadSampleJson = () => {
    const sample = [
      {
        type: "step_penalty",
        value: -0.01,
      },
      {
        type: "state_threshold",
        index: 2,
        op: ">",
        value: 0.1,
        reward: -1,
      },
      {
        type: "terminal_bonus",
        reward: 50,
      },
      {
        type: "terminal_penalty",
        reward: -50,
      },
    ];
    const dataStr =
      "data:text/json;charset=utf-8," +
      encodeURIComponent(JSON.stringify(sample, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "rewards.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  // Operational states
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle"); // idle, training, completed, cancelled, failed
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [chartData, setChartData] = useState<ProgressData[]>([]);
  const [currentEpisode, setCurrentEpisode] = useState<number>(0);
  const [avgReward, setAvgReward] = useState<number>(0);
  const [mounted, setMounted] = useState<boolean>(false);
  const [showSim, setShowSim] = useState<boolean>(false);
  const [simCacheBuster, setSimCacheBuster] = useState<number>(0);

  const consoleEndRef = useRef<HTMLDivElement | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    setMounted(true);
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Auto-scroll logs terminal

  // Run Training
  const handleStartTraining = async () => {
    setStatus("training");
    setErrorMessage(null);
    setLogs(["[SYSTEM] Connecting to backend..."]);
    setChartData([]);
    setCurrentEpisode(0);
    setAvgReward(0);
    setShowSim(false);

    try {
      // Parse hidden layers
      const parsedLayers = hiddenLayers
        .split(",")
        .map((x) => parseInt(x.trim()))
        .filter((x) => !isNaN(x) && x > 0);

      if (parsedLayers.length === 0) {
        throw new Error(
          "Hidden layers must be comma-separated integers (e.g. 128, 128).",
        );
      }

      // Parse reward rules JSON
      let parsedRules: any[] = [];
      try {
        parsedRules = JSON.parse(rewardRulesJson);
        if (!Array.isArray(parsedRules)) {
          throw new Error("Reward rules must be a JSON array.");
        }
      } catch (err: any) {
        throw new Error("Invalid reward rules JSON: " + err.message);
      }

      const payload = {
        env: env === "custom" ? customEnv : env,
        episodes,
        max_steps: maxSteps,
        gamma,
        lr,
        batch_size: batchSize,
        buffer_size: bufferSize,
        min_buffer_size: minBufferSize,
        target_update_freq: targetUpdateFreq,
        optimizer: "adam",
        weight_decay: 0.0,
        epsilon_start: epsilonStart,
        epsilon_end: epsilonEnd,
        epsilon_decay: epsilonDecay,
        hidden_layers: parsedLayers,
        activation,
        seed: 42,
        reward_rules: parsedRules,
        fail_reward_threshold: failRewardThreshold,
      };

      const response = await fetch(
        (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000") +
          "/rl/train",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
      );

      if (!response.ok) {
        let errorMsg = "Failed to start training.";
        try {
          const errData = await response.json();
          errorMsg = errData.detail || JSON.stringify(errData);
        } catch {
          try {
            errorMsg = await response.text();
          } catch {
            errorMsg = `Server returned status ${response.status}`;
          }
        }
        throw new Error(errorMsg);
      }

      const { task_id } = await response.json();
      setTaskId(task_id);
      connectToSSE(task_id);
    } catch (e: any) {
      setStatus("failed");
      setErrorMessage(e.message || "An error occurred.");
      setLogs((prev) => [
        ...prev,
        `[ERROR] ${e.message || "Failed to start training."}`,
      ]);
    }
  };

  // Connect to SSE Endpoint for progress tracking
  const connectToSSE = (taskId: string) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const sse = new EventSource(
      (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000") +
        "/rl/progress/" +
        taskId,
    );
    eventSourceRef.current = sse;

    sse.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "log") {
          setLogs((prev) => [...prev, payload.data]);
        } else if (payload.type === "progress") {
          const progress: ProgressData = payload.data;
          setChartData((prev) => {
            // Avoid duplicates just in case
            if (prev.some((p) => p.episode === progress.episode)) return prev;
            return [...prev, progress];
          });
          setCurrentEpisode(progress.episode);
          setAvgReward(progress.avg_reward);
        } else if (payload.type === "status") {
          setStatus(payload.status);
          if (payload.status === "failed") {
            setErrorMessage(payload.error || "Training failed.");
          }
          sse.close();
        }
      } catch (e) {
        console.error("SSE parse error", e);
      }
    };

    sse.onerror = (e) => {
      console.error("SSE connection error", e);
      // Wait a moment and check status
      setLogs((prev) => [...prev, "[SYSTEM] Connection lost. Reconnecting..."]);
    };
  };

  const handleCancelTraining = async () => {
    if (!taskId) return;
    try {
      await fetch(
        (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000") +
          "/rl/cancel/" +
          taskId,
        {
          method: "POST",
        },
      );
      setStatus("cancelled");
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    } catch (e) {
      console.error("Cancel failed", e);
    }
  };

  return (
    <>
      <Navbar />
      <main className="max-w-[1440px] flex flex-col gap-lg p-lg w-screen mx-auto text-on-surface">
        {/* Title and Intro */}
        <section className="flex flex-col gap-sm">
          <div className="flex items-center gap-md text-primary">
            <MdScience className="text-4xl text-cyan-400 animate-pulse" />
            <h1 className="font-heading-xl text-3xl font-extrabold tracking-tight">
              Reinforcement Learning Laboratory
            </h1>
          </div>
        </section>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-lg items-start">
          {/* Left Column: Form Settings (4 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-md">
            {/* Environment & Network Config */}
            <div className="glass-panel p-md rounded-xl space-y-4 border border-white/10 bg-surface-container/60 shadow-xl">
              <h2 className="font-heading-md text-md font-semibold text-cyan-300 flex items-center gap-2">
                <MdSettings /> Environment & Network
              </h2>

              <div className="grid grid-cols-2 gap-sm">
                <div className="flex flex-col gap-1">
                  <label className="font-label-mono text-label-mono text-on-surface-variant">
                    Environment
                  </label>
                  <select
                    value={env}
                    onChange={(e) => setEnv(e.target.value)}
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary transition-colors"
                  >
                    <option value="CartPole-v1">CartPole-v1 (Discrete)</option>
                    <option value="MountainCar-v0">
                      MountainCar-v0 (Discrete)
                    </option>
                    <option value="Acrobot-v1">Acrobot-v1 (Discrete)</option>
                  </select>
                </div>
                {env === "custom" && (
                  <div className="flex flex-col gap-1">
                    <label className="font-label-mono text-label-mono text-on-surface-variant">
                      Env Name
                    </label>
                    <input
                      type="text"
                      placeholder="e.g. FrozenLake-v1"
                      value={customEnv}
                      onChange={(e) => setCustomEnv(e.target.value)}
                      disabled={status === "training"}
                      className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                    />
                  </div>
                )}

                <div className="flex flex-col gap-1">
                  <label className="font-label-mono text-label-mono text-on-surface-variant">
                    Hidden Layers
                  </label>
                  <input
                    type="text"
                    value={hiddenLayers}
                    onChange={(e) => setHiddenLayers(e.target.value)}
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                    placeholder="e.g. 128, 128"
                  />
                </div>

                <div className="flex flex-col gap-1">
                  <label className="font-label-mono text-label-mono text-on-surface-variant">
                    Activation
                  </label>
                  <select
                    value={activation}
                    onChange={(e) => setActivation(e.target.value)}
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                  >
                    <option value="relu">ReLU</option>
                    <option value="tanh">Tanh</option>
                    <option value="sigmoid">Sigmoid</option>
                    <option value="leaky_relu">Leaky ReLU</option>
                    <option value="gelu">GELU</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Hyperparameters Config */}
            <div className="glass-panel p-md rounded-xl space-y-4 border border-white/10 bg-surface-container/60 shadow-xl">
              <h2 className="font-heading-md text-md font-semibold text-cyan-300 flex items-center gap-2">
                <MdSettings /> DQN Hyperparameters
              </h2>

              <div className="grid grid-cols-3 gap-xs">
                <div className="flex flex-col gap-1">
                  <label className="font-label-mono text-[10px] text-on-surface-variant">
                    Episodes
                  </label>
                  <input
                    type="number"
                    value={episodes}
                    onChange={(e) => setEpisodes(Number(e.target.value))}
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="font-label-mono text-[10px] text-on-surface-variant">
                    Max Steps
                  </label>
                  <input
                    type="number"
                    value={maxSteps}
                    onChange={(e) => setMaxSteps(Number(e.target.value))}
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="font-label-mono text-[10px] text-on-surface-variant">
                    LR
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    value={lr}
                    onChange={(e) => setLr(Number(e.target.value))}
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                  />
                </div>

                <div className="flex flex-col gap-1 mt-2">
                  <label className="font-label-mono text-[10px] text-on-surface-variant">
                    Discount (γ)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={gamma}
                    onChange={(e) => setGamma(Number(e.target.value))}
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                  />
                </div>
                <div className="flex flex-col gap-1 mt-2">
                  <label className="font-label-mono text-[10px] text-on-surface-variant">
                    Batch Size
                  </label>
                  <select
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                  >
                    <option value={16}>16</option>
                    <option value={32}>32</option>
                    <option value={64}>64</option>
                    <option value={128}>128</option>
                    <option value={256}>256</option>
                  </select>
                </div>
                <div className="flex flex-col gap-1 mt-2">
                  <label className="font-label-mono text-[10px] text-on-surface-variant">
                    Buffer Size
                  </label>
                  <input
                    type="number"
                    value={bufferSize}
                    onChange={(e) => setBufferSize(Number(e.target.value))}
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                  />
                </div>

                <div className="flex flex-col gap-1 mt-2">
                  <label className="font-label-mono text-[10px] text-on-surface-variant">
                    Min Buffer
                  </label>
                  <input
                    type="number"
                    value={minBufferSize}
                    onChange={(e) => setMinBufferSize(Number(e.target.value))}
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                  />
                </div>
                <div className="flex flex-col gap-1 mt-2">
                  <label className="font-label-mono text-[10px] text-on-surface-variant">
                    Target Sync
                  </label>
                  <input
                    type="number"
                    value={targetUpdateFreq}
                    onChange={(e) =>
                      setTargetUpdateFreq(Number(e.target.value))
                    }
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                  />
                </div>
                <div className="flex flex-col gap-1 mt-2">
                  <label className="font-label-mono text-[10px] text-on-surface-variant">
                    Fail Thresh
                  </label>
                  <input
                    type="number"
                    value={failRewardThreshold}
                    onChange={(e) =>
                      setFailRewardThreshold(Number(e.target.value))
                    }
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                  />
                </div>

                <div className="flex flex-col gap-1 mt-2">
                  <label className="font-label-mono text-[10px] text-on-surface-variant">
                    Epsilon Start
                  </label>
                  <input
                    type="number"
                    step="0.05"
                    value={epsilonStart}
                    onChange={(e) => setEpsilonStart(Number(e.target.value))}
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                  />
                </div>
                <div className="flex flex-col gap-1 mt-2">
                  <label className="font-label-mono text-[10px] text-on-surface-variant">
                    Eps Decay
                  </label>
                  <input
                    type="number"
                    step="0.001"
                    value={epsilonDecay}
                    onChange={(e) => setEpsilonDecay(Number(e.target.value))}
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                  />
                </div>
                <div className="flex flex-col gap-1 mt-2">
                  <label className="font-label-mono text-[10px] text-on-surface-variant">
                    Epsilon End
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={epsilonEnd}
                    onChange={(e) => setEpsilonEnd(Number(e.target.value))}
                    disabled={status === "training"}
                    className="bg-surface-container-low border border-outline-variant text-on-surface rounded-lg p-sm text-sm focus:outline-none focus:border-primary"
                  />
                </div>
              </div>
            </div>

            {/* JSON Reward Rules Configuration */}
            <div className="glass-panel p-md rounded-xl space-y-4 border border-white/10 bg-surface-container/60 shadow-xl">
              <div className="flex justify-between items-center">
                <h2 className="font-heading-md text-md font-semibold text-cyan-300 flex items-center gap-2">
                  Reward Configuration (JSON)
                </h2>
                <button
                  onClick={handleDownloadSampleJson}
                  className="px-3 py-1 bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/30 border border-cyan-500/30 rounded-lg text-xs flex items-center gap-1.5 transition-all font-medium"
                >
                  <MdDownload className="text-sm" /> Sample JSON
                </button>
              </div>
              <div className="flex flex-col gap-2">
                <p className="text-xs text-on-surface-variant leading-relaxed">
                  Define custom reward shaping rules as a JSON array. These
                  rules are applied step-by-step during training to guide agent
                  learning.
                </p>
                <div className="relative">
                  <textarea
                    value={rewardRulesJson}
                    onChange={(e) => setRewardRulesJson(e.target.value)}
                    disabled={status === "training"}
                    rows={10}
                    className="w-full font-mono text-xs  border border-outline-variant text-zinc-300 rounded-lg p-3 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400/25 transition-all resize-none shadow-inner"
                    placeholder="Enter reward rules JSON list..."
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Execution Monitoring & Testing (7 cols) */}
          <div className="lg:col-span-7 flex flex-col gap-md">
            {/* Control Panel Card */}
            <div className="glass-panel p-md rounded-xl border border-white/10 bg-surface-container/60 shadow-xl flex items-center justify-between gap-4">
              <div className="flex flex-col gap-0.5">
                <div className="text-xs text-on-surface-variant uppercase tracking-wider font-semibold">
                  Training Engine
                </div>
                <div className="flex items-center gap-2">
                  <span
                    className={`w-3 h-3 rounded-full ${
                      status === "training"
                        ? "bg-amber-400 animate-pulse"
                        : status === "completed"
                          ? "bg-green-400"
                          : status === "failed"
                            ? "bg-red-500"
                            : "bg-gray-400"
                    }`}
                  />
                  <span className="text-sm font-bold capitalize">{status}</span>
                  {status === "training" && (
                    <span className="text-xs text-on-surface-variant">
                      (Episode {currentEpisode} / {episodes})
                    </span>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-sm">
                {status === "training" ? (
                  <button
                    onClick={handleCancelTraining}
                    className="px-4 py-2 bg-red-600/30 text-red-300 hover:bg-red-600/50 border border-red-500/20 rounded-lg text-sm flex items-center gap-1.5 transition-colors font-medium"
                  >
                    <MdCancel className="text-lg" /> Cancel
                  </button>
                ) : (
                  <button
                    onClick={handleStartTraining}
                    className="px-5 py-2 bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/40 border border-cyan-500/30 rounded-lg text-sm flex items-center gap-1.5 transition-all font-semibold shadow-lg hover:shadow-cyan-500/10"
                  >
                    <MdPlayArrow className="text-lg" /> Start Training
                  </button>
                )}

                {status === "completed" && taskId && (
                  <a
                    href={
                      (process.env.NEXT_PUBLIC_BACKEND_URL ||
                        "http://localhost:8000") +
                      "/rl/download/" +
                      taskId
                    }
                    download
                    className="px-4 py-2 bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/40 border border-emerald-500/30 rounded-lg text-sm flex items-center gap-1.5 transition-colors font-medium"
                  >
                    <MdDownload className="text-lg" /> Download Model
                  </a>
                )}
              </div>
            </div>

            {/* Error Message */}
            {errorMessage && (
              <div className="bg-red-900/30 border border-red-500/20 text-red-300 text-sm px-4 py-3 rounded-xl">
                <strong>Error: </strong> {errorMessage}
              </div>
            )}

            {/* Live Chart Panel */}
            <div className="glass-panel p-md rounded-xl border border-white/10 bg-surface-container/60 shadow-xl space-y-3">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-semibold text-cyan-300 uppercase tracking-wider">
                  Metrics Progression
                </h3>
                <span className="text-xs text-on-surface-variant">
                  Last 10 Avg Reward: {avgReward.toFixed(2)}
                </span>
              </div>

              <div className="h-64 w-full">
                {mounted && chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={chartData}
                      margin={{ top: 5, right: 10, left: -20, bottom: 5 }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#ffffff"
                        opacity={0.05}
                      />
                      <XAxis dataKey="episode" stroke="#a1a1aa" fontSize={10} />
                      <YAxis stroke="#a1a1aa" fontSize={10} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#0f172a",
                          borderColor: "rgba(255,255,255,0.15)",
                          borderRadius: "8px",
                        }}
                        labelClassName="text-xs font-bold text-cyan-400"
                        itemStyle={{ fontSize: "11px", padding: "1px 0" }}
                      />
                      <Legend
                        verticalAlign="top"
                        height={24}
                        iconType="circle"
                        wrapperStyle={{ fontSize: "11px" }}
                      />
                      <Line
                        type="monotone"
                        dataKey="reward"
                        stroke="var(--color-primary, #a855f7)"
                        name="Episode Reward"
                        strokeWidth={1}
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="avg_reward"
                        stroke="var(--color-secondary, #10b981)"
                        name="Avg Reward (10)"
                        strokeWidth={2.5}
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="#ef4444"
                        name="Avg Loss"
                        strokeWidth={1}
                        dot={false}
                        yAxisId={0}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-xs text-on-surface-variant italic bg-surface-container-low rounded-lg border border-dashed border-outline-variant">
                    {status === "training"
                      ? "Accumulating metrics data..."
                      : "Start training to populate chart metrics."}
                  </div>
                )}
              </div>
            </div>

            {/* Test Simulation Panel */}
            {status === "completed" && taskId && (
              <div className="glass-panel p-md rounded-xl border border-white/10 bg-surface-container/60 shadow-xl space-y-4">
                <div className="flex justify-between items-center">
                  <h3 className="text-sm font-semibold text-cyan-300 uppercase tracking-wider flex items-center gap-1.5">
                    <MdVideogameAsset className="text-lg" /> Pygame Gym
                    Simulation
                  </h3>

                  {showSim ? (
                    <button
                      onClick={() => setShowSim(false)}
                      className="px-3 py-1 bg-red-600/30 text-red-300 hover:bg-red-600/50 rounded-lg text-xs flex items-center gap-1.5 transition-colors"
                    >
                      <MdStop /> Stop Sim
                    </button>
                  ) : (
                    <button
                      onClick={() => {
                        setSimCacheBuster(Date.now());
                        setShowSim(true);
                      }}
                      className="px-3 py-1 bg-cyan-600/30 text-cyan-300 hover:bg-cyan-600/50 rounded-lg text-xs flex items-center gap-1.5 transition-colors"
                    >
                      <MdPlayArrow /> Run Agent Live
                    </button>
                  )}
                </div>

                {showSim ? (
                  <div className="flex flex-col items-center gap-2 bg-surface-container-low p-4 rounded-xl border border-outline-variant">
                    <img
                      src={
                        (process.env.NEXT_PUBLIC_BACKEND_URL ||
                          "http://localhost:8000") +
                        "/rl/test/stream/" +
                        taskId +
                        "?episodes=3&max_steps=500&t=" +
                        simCacheBuster
                      }
                      alt="Gym Environment Pygame Simulation"
                      className="rounded-lg max-w-full border border-outline-variant bg-slate-950"
                      style={{ height: "300px", width: "auto" }}
                      onError={() => {
                        setLogs((prev) => [
                          ...prev,
                          "[ERROR] Simulation stream closed or failed to load.",
                        ]);
                        setShowSim(false);
                      }}
                    />
                    <div className="text-[10px] text-on-surface-variant italic">
                      Live stream of the environment generated on the server
                      using Pygame and Gymnasium.
                    </div>
                  </div>
                ) : (
                  <div className="h-40 flex flex-col items-center justify-center text-xs text-on-surface-variant bg-surface-container-low rounded-lg border border-dashed border-outline-variant p-4 text-center gap-1">
                    <span className="font-semibold text-on-surface">
                      Visual Simulator Standby
                    </span>
                    <span>
                      Click &quot;Run Agent Live&quot; to test your trained
                      model and watch it play using Pygame rendering.
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* Console Terminal Logs */}
            <div className="glass-panel p-md rounded-xl border border-white/10 bg-slate-950 shadow-xl space-y-2 flex flex-col h-64">
              <div className="flex items-center gap-1.5 text-xs text-zinc-400 font-mono border-b border-white/5 pb-1">
                <MdTerminal className="text-md text-zinc-500" />
                <span>TERMINAL LOG CONSOLE</span>
              </div>
              <div className="flex-1 font-mono text-[11px] leading-relaxed text-zinc-300 overflow-y-auto space-y-1 select-text scrollbar-thin">
                {logs.length === 0 ? (
                  <div className="text-zinc-600 italic">
                    Terminal ready. Run training to print standard output
                    logs...
                  </div>
                ) : (
                  logs.map((log, i) => (
                    <div key={i} className="whitespace-pre-wrap break-all">
                      {log}
                    </div>
                  ))
                )}
                <div ref={consoleEndRef} />
              </div>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
