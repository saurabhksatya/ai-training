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

  // Reward rule builder states
  const [rewardRules, setRewardRules] = useState<RewardRule[]>([
    { id: "1", type: "step_penalty", value: -0.01 },
  ]);

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
  useEffect(() => {
    if (consoleEndRef.current) {
      consoleEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);

  // Add a new reward rule to visual builder
  const addRewardRule = (type: string) => {
    const newRule: RewardRule = {
      id: Math.random().toString(36).substr(2, 9),
      type,
    };
    if (type === "step_penalty") newRule.value = -0.01;
    else if (type === "state_threshold") {
      newRule.index = 0;
      newRule.op = ">";
      newRule.value = 0.0;
      newRule.reward = 1.0;
    } else if (type === "action_bonus") {
      newRule.action = 0;
      newRule.reward = 1.0;
    } else if (type === "terminal_bonus" || type === "terminal_penalty") {
      newRule.reward = 10.0;
    } else if (type === "override_reward") {
      newRule.value = 1.0;
    }
    setRewardRules([...rewardRules, newRule]);
  };

  const removeRewardRule = (id: string) => {
    setRewardRules(rewardRules.filter((r) => r.id !== id));
  };

  const updateRuleField = (id: string, field: keyof RewardRule, val: any) => {
    setRewardRules(
      rewardRules.map((rule) => {
        if (rule.id === id) {
          return { ...rule, [field]: val };
        }
        return rule;
      }),
    );
  };

  // Helpers to manage rule types via checkboxes
  const ruleExists = (type: string) => rewardRules.some((r) => r.type === type);

  const removeRewardRulesByType = (type: string) => {
    setRewardRules(rewardRules.filter((r) => r.type !== type));
  };

  const toggleRuleType = (type: string, checked: boolean) => {
    if (checked) addRewardRule(type);
    else removeRewardRulesByType(type);
  };

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

      // Convert visual rules to backend payload format
      const formattedRules = rewardRules.map((r) => {
        const payload: any = { type: r.type };
        if (r.type === "step_penalty") payload.value = Number(r.value);
        else if (r.type === "state_threshold") {
          payload.index = Number(r.index);
          payload.op = r.op;
          payload.value = Number(r.value);
          payload.reward = Number(r.reward);
        } else if (r.type === "action_bonus") {
          payload.action = Number(r.action);
          payload.reward = Number(r.reward);
        } else if (
          r.type === "terminal_bonus" ||
          r.type === "terminal_penalty"
        ) {
          payload.reward = Number(r.reward);
        } else if (r.type === "override_reward") {
          payload.value = Number(r.value);
        }
        return payload;
      });

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
        reward_rules: formattedRules,
        fail_reward_threshold: failRewardThreshold,
      };

      const response = await fetch("http://localhost:8000/rl/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to start training.");
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

    const sse = new EventSource(`http://localhost:8000/rl/progress/${taskId}`);
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
      await fetch(`http://localhost:8000/rl/cancel/${taskId}`, {
        method: "POST",
      });
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

            {/* Visual Reward Rule Builder */}
            <div className="glass-panel p-md rounded-xl space-y-4 border border-white/10 bg-surface-container/60 shadow-xl">
              <div className="flex justify-between items-center">
                <h2 className="font-heading-md text-md font-semibold text-cyan-300 flex items-center gap-2">
                  Visual Reward Builder
                </h2>
              </div>
              <div className="flex flex-col gap-2">
                <div className="text-xs text-on-surface-variant">
                  Add rule types (visible):
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  <label className="flex items-center gap-2 text-xs">
                    <input
                      type="checkbox"
                      checked={ruleExists("step_penalty")}
                      onChange={(e) =>
                        toggleRuleType("step_penalty", e.target.checked)
                      }
                      disabled={status === "training"}
                      className="w-4 h-4 appearance-none checked:appearance-auto bg-white rounded"
                    />
                    Step Penalty
                  </label>
                  <label className="flex items-center gap-2 text-xs">
                    <input
                      type="checkbox"
                      checked={ruleExists("state_threshold")}
                      onChange={(e) =>
                        toggleRuleType("state_threshold", e.target.checked)
                      }
                      disabled={status === "training"}
                      className="w-4 h-4 appearance-none checked:appearance-auto bg-white rounded"
                    />
                    State Threshold Rule
                  </label>
                  <label className="flex items-center gap-2 text-xs">
                    <input
                      type="checkbox"
                      checked={ruleExists("action_bonus")}
                      onChange={(e) =>
                        toggleRuleType("action_bonus", e.target.checked)
                      }
                      disabled={status === "training"}
                      className="w-4 h-4 appearance-none checked:appearance-auto bg-white rounded"
                    />
                    Action Bonus
                  </label>
                  <label className="flex items-center gap-2 text-xs">
                    <input
                      type="checkbox"
                      checked={ruleExists("terminal_bonus")}
                      onChange={(e) =>
                        toggleRuleType("terminal_bonus", e.target.checked)
                      }
                      disabled={status === "training"}
                      className="w-4 h-4 appearance-none checked:appearance-auto bg-white rounded"
                    />
                    Terminal Success Bonus
                  </label>
                  <label className="flex items-center gap-2 text-xs">
                    <input
                      type="checkbox"
                      checked={ruleExists("terminal_penalty")}
                      onChange={(e) =>
                        toggleRuleType("terminal_penalty", e.target.checked)
                      }
                      disabled={status === "training"}
                      className="w-4 h-4 appearance-none checked:appearance-auto bg-white rounded"
                    />
                    Terminal Fail Penalty
                  </label>
                  <label className="flex items-center gap-2 text-xs">
                    <input
                      type="checkbox"
                      checked={ruleExists("override_reward")}
                      onChange={(e) =>
                        toggleRuleType("override_reward", e.target.checked)
                      }
                      disabled={status === "training"}
                      className="w-4 h-4 appearance-none checked:appearance-auto bg-white rounded"
                    />
                    Override Reward
                  </label>
                </div>
              </div>

              {/* Rules List */}
              <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                {rewardRules.length === 0 ? (
                  <p className="text-xs text-on-surface-variant italic py-4 text-center">
                    No custom reward rules. Default environment reward is used.
                  </p>
                ) : (
                  rewardRules.map((rule) => (
                    <div
                      key={rule.id}
                      className="p-md bg-surface-container border border-outline-variant rounded-xl flex items-start justify-between gap-2"
                    >
                      <div className="flex-1 space-y-1.5">
                        <div className="text-xs font-bold text-cyan-400 capitalize">
                          {rule.type.replace("_", " ")}
                        </div>

                        {/* Dynamic fields based on rule type */}
                        {rule.type === "step_penalty" && (
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] text-on-surface-variant">
                              Penalty Value:
                            </span>
                            <input
                              type="number"
                              step="0.01"
                              value={rule.value ?? -0.01}
                              onChange={(e) =>
                                updateRuleField(
                                  rule.id,
                                  "value",
                                  Number(e.target.value),
                                )
                              }
                              disabled={status === "training"}
                              className="bg-surface-container-low border border-outline-variant text-on-surface text-xs rounded px-1.5 py-0.5 w-20 focus:outline-none focus:border-primary"
                            />
                          </div>
                        )}

                        {rule.type === "state_threshold" && (
                          <div className="grid grid-cols-2 gap-1.5">
                            <div className="flex items-center gap-1">
                              <span className="text-[10px] text-on-surface-variant">
                                StateIdx:
                              </span>
                              <input
                                type="number"
                                value={rule.index ?? 0}
                                onChange={(e) =>
                                  updateRuleField(
                                    rule.id,
                                    "index",
                                    Number(e.target.value),
                                  )
                                }
                                disabled={status === "training"}
                                className="bg-surface-container-low border border-outline-variant text-on-surface text-xs rounded px-1 py-0.5 w-12 focus:outline-none focus:border-primary"
                              />
                            </div>
                            <div className="flex items-center gap-1">
                              <span className="text-[10px] text-on-surface-variant">
                                Op:
                              </span>
                              <select
                                value={rule.op ?? ">"}
                                onChange={(e) =>
                                  updateRuleField(rule.id, "op", e.target.value)
                                }
                                disabled={status === "training"}
                                className="bg-surface-container-low border border-outline-variant text-on-surface text-xs rounded px-1 w-14 focus:outline-none focus:border-primary"
                              >
                                <option value=">">&gt;</option>
                                <option value="<">&lt;</option>
                                <option value=">=">&gt;=</option>
                                <option value="<=">&lt;=</option>
                                <option value="==">==</option>
                                <option value="!=">!=</option>
                              </select>
                            </div>
                            <div className="flex items-center gap-1">
                              <span className="text-[10px] text-on-surface-variant">
                                Threshold:
                              </span>
                              <input
                                type="number"
                                step="0.1"
                                value={rule.value ?? 0.0}
                                onChange={(e) =>
                                  updateRuleField(
                                    rule.id,
                                    "value",
                                    Number(e.target.value),
                                  )
                                }
                                disabled={status === "training"}
                                className="bg-surface-container-low border border-outline-variant text-on-surface text-xs rounded px-1 py-0.5 w-16 focus:outline-none focus:border-primary"
                              />
                            </div>
                            <div className="flex items-center gap-1">
                              <span className="text-[10px] text-on-surface-variant">
                                Reward:
                              </span>
                              <input
                                type="number"
                                step="0.5"
                                value={rule.reward ?? 1.0}
                                onChange={(e) =>
                                  updateRuleField(
                                    rule.id,
                                    "reward",
                                    Number(e.target.value),
                                  )
                                }
                                disabled={status === "training"}
                                className="bg-surface-container-low border border-outline-variant text-on-surface text-xs rounded px-1 py-0.5 w-16 focus:outline-none focus:border-primary"
                              />
                            </div>
                          </div>
                        )}

                        {rule.type === "action_bonus" && (
                          <div className="grid grid-cols-2 gap-1.5">
                            <div className="flex items-center gap-1">
                              <span className="text-[10px] text-on-surface-variant">
                                Action:
                              </span>
                              <input
                                type="number"
                                value={rule.action ?? 0}
                                onChange={(e) =>
                                  updateRuleField(
                                    rule.id,
                                    "action",
                                    Number(e.target.value),
                                  )
                                }
                                disabled={status === "training"}
                                className="bg-surface-container-low border border-outline-variant text-on-surface text-xs rounded px-1 py-0.5 w-12 focus:outline-none focus:border-primary"
                              />
                            </div>
                            <div className="flex items-center gap-1">
                              <span className="text-[10px] text-on-surface-variant">
                                Reward:
                              </span>
                              <input
                                type="number"
                                step="0.5"
                                value={rule.reward ?? 1.0}
                                onChange={(e) =>
                                  updateRuleField(
                                    rule.id,
                                    "reward",
                                    Number(e.target.value),
                                  )
                                }
                                disabled={status === "training"}
                                className="bg-surface-container-low border border-outline-variant text-on-surface text-xs rounded px-1 py-0.5 w-16 focus:outline-none focus:border-primary"
                              />
                            </div>
                          </div>
                        )}

                        {(rule.type === "terminal_bonus" ||
                          rule.type === "terminal_penalty") && (
                            <div className="flex items-center gap-2">
                              <span className="text-[10px] text-on-surface-variant">
                                Bonus/Penalty Value:
                              </span>
                              <input
                                type="number"
                                step="1"
                                value={rule.reward ?? 10.0}
                                onChange={(e) =>
                                  updateRuleField(
                                    rule.id,
                                    "reward",
                                    Number(e.target.value),
                                  )
                                }
                                disabled={status === "training"}
                                className="bg-surface-container-low border border-outline-variant text-on-surface text-xs rounded px-1.5 py-0.5 w-20 focus:outline-none focus:border-primary"
                              />
                            </div>
                          )}

                        {rule.type === "override_reward" && (
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] text-on-surface-variant">
                              Override Value:
                            </span>
                            <input
                              type="number"
                              step="0.5"
                              value={rule.value ?? 1.0}
                              onChange={(e) =>
                                updateRuleField(
                                  rule.id,
                                  "value",
                                  Number(e.target.value),
                                )
                              }
                              disabled={status === "training"}
                              className="bg-surface-container-low border border-outline-variant text-on-surface text-xs rounded px-1.5 py-0.5 w-20 focus:outline-none focus:border-primary"
                            />
                          </div>
                        )}
                      </div>
                      <button
                        onClick={() => removeRewardRule(rule.id)}
                        disabled={status === "training"}
                        className="text-red-400 hover:text-red-300 disabled:opacity-50 p-1 hover:bg-white/5 rounded transition-colors"
                      >
                        <MdDelete className="text-lg" />
                      </button>
                    </div>
                  ))
                )}
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
                    className={`w-3 h-3 rounded-full ${status === "training"
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
                    href={`http://localhost:8000/rl/download/${taskId}`}
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
                      src={`http://localhost:8000/rl/test/stream/${taskId}?episodes=3&max_steps=500&t=${simCacheBuster}`}
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
