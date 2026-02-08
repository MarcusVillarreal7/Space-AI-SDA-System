#!/usr/bin/env python3
"""
Day 6 — End-to-End Validation on Full Dataset.

Runs the ThreatAssessmentPipeline on all 1000 objects from the ground
truth dataset with GPU-accelerated anomaly detection and chunked data
loading.

Outputs:
    - Per-object threat assessment
    - Threat tier distribution
    - Latency benchmarks (per-object and throughput)
    - Score statistics and distribution
    - Per-regime analysis
    - Saved results to results/e2e_validation/

Usage:
    python scripts/run_e2e_validation.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.threat_assessment import ThreatAssessmentPipeline, ThreatAssessment
from src.ml.threat.threat_scorer import ThreatTier

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

DATA_PATH = Path("data/processed/ml_train_1k/ground_truth.parquet")
ANOMALY_CHECKPOINT = Path("checkpoints/phase3_anomaly")
RESULTS_DIR = Path("results/e2e_validation")
CHUNK_SIZE = 100  # Objects per processing chunk
SEED = 42


def main():
    np.random.seed(SEED)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\n{'='*60}")
    print("Day 6 — End-to-End Validation")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 1: Initialize pipeline
    # ------------------------------------------------------------------
    print(f"\n[Step 1] Initializing ThreatAssessmentPipeline")
    t0 = time.time()

    pipeline = ThreatAssessmentPipeline(
        anomaly_checkpoint=str(ANOMALY_CHECKPOINT),
        device=device,
    )

    init_time = time.time() - t0
    has_anomaly = pipeline.anomaly_detector is not None
    print(f"  Intent classifier: ready")
    print(f"  Anomaly detector:  {'loaded from checkpoint' if has_anomaly else 'not available'}")
    print(f"  Threat scorer:     ready")
    print(f"  Init time:         {init_time:.2f}s")

    # ------------------------------------------------------------------
    # Step 2: Load data and process in chunks
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Loading ground truth from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    object_ids = sorted(df["object_id"].unique())
    n_objects = len(object_ids)
    n_chunks = (n_objects + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"  Objects: {n_objects}")
    print(f"  Timesteps per object: {len(df) // n_objects}")
    print(f"  Processing in {n_chunks} chunks of {CHUNK_SIZE} objects")

    all_assessments: list[ThreatAssessment] = []
    chunk_times = []

    for chunk_idx in range(n_chunks):
        start_id = chunk_idx * CHUNK_SIZE
        end_id = min(start_id + CHUNK_SIZE, n_objects)
        chunk_obj_ids = object_ids[start_id:end_id]

        t_chunk = time.time()
        df_chunk = df[df["object_id"].isin(chunk_obj_ids)]

        chunk_assessments = []
        for obj_id in chunk_obj_ids:
            obj_df = df_chunk[df_chunk["object_id"] == obj_id].sort_values("time")

            positions = obj_df[["position_x", "position_y", "position_z"]].values
            velocities = obj_df[["velocity_x", "velocity_y", "velocity_z"]].values
            timestamps = (obj_df["time"] - obj_df["time"].iloc[0]).dt.total_seconds().values

            assessment = pipeline.assess(
                object_id=str(obj_id),
                positions=positions,
                velocities=velocities,
                timestamps=timestamps,
            )
            chunk_assessments.append(assessment)

        chunk_time = time.time() - t_chunk
        chunk_times.append(chunk_time)
        all_assessments.extend(chunk_assessments)

        print(f"  Chunk {chunk_idx+1}/{n_chunks}: {len(chunk_assessments)} objects "
              f"in {chunk_time:.2f}s ({chunk_time/len(chunk_assessments)*1000:.1f} ms/obj)")

    total_time = sum(chunk_times)
    print(f"\n  Total: {n_objects} objects in {total_time:.2f}s")
    print(f"  Throughput: {n_objects/total_time:.0f} objects/sec")

    # ------------------------------------------------------------------
    # Step 3: Analyze results
    # ------------------------------------------------------------------
    print(f"\n[Step 3] Analyzing results")

    scores = np.array([a.threat_score.score for a in all_assessments])
    latencies = np.array([a.latency_ms for a in all_assessments])
    tiers = [a.threat_score.tier for a in all_assessments]

    # Threat score distribution
    print(f"\n  === Threat Score Distribution ===")
    print(f"  Mean:   {scores.mean():.1f}")
    print(f"  Std:    {scores.std():.1f}")
    print(f"  Min:    {scores.min():.1f}")
    print(f"  Max:    {scores.max():.1f}")
    print(f"  Median: {np.median(scores):.1f}")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  P{p}:   {np.percentile(scores, p):.1f}")

    # Tier distribution
    print(f"\n  === Threat Tier Distribution ===")
    tier_counts = {}
    for tier in ThreatTier:
        count = sum(1 for t in tiers if t == tier)
        pct = count / n_objects * 100
        tier_counts[tier.value] = count
        print(f"  {tier.value:10s}: {count:4d} ({pct:.1f}%)")

    # Intent distribution
    print(f"\n  === Intent Distribution ===")
    intent_counts = {}
    for a in all_assessments:
        intent = a.intent_result.intent.value
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        print(f"  {intent:22s}: {count:4d} ({count/n_objects*100:.1f}%)")

    # Anomaly distribution
    if has_anomaly:
        print(f"\n  === Anomaly Detection ===")
        n_anomalous = sum(1 for a in all_assessments if a.anomaly_result and a.anomaly_result.is_anomaly)
        print(f"  Anomalous: {n_anomalous}/{n_objects} ({n_anomalous/n_objects*100:.1f}%)")
        anomaly_scores = [a.anomaly_result.anomaly_score for a in all_assessments if a.anomaly_result]
        if anomaly_scores:
            anomaly_arr = np.array(anomaly_scores)
            print(f"  Anomaly score: mean={anomaly_arr.mean():.4f}, "
                  f"max={anomaly_arr.max():.4f}")

    # Latency analysis
    print(f"\n  === Latency (ms/object) ===")
    print(f"  Mean:   {latencies.mean():.2f} ms")
    print(f"  Std:    {latencies.std():.2f} ms")
    print(f"  P50:    {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.2f} ms")
    print(f"  Max:    {latencies.max():.2f} ms")

    # Per-regime analysis
    print(f"\n  === Per-Regime Analysis ===")
    regime_data = {}
    for a in all_assessments:
        regime = a.intent_result.proximity.object_regime.value if a.intent_result.proximity else "UNKNOWN"
        if regime not in regime_data:
            regime_data[regime] = {"scores": [], "count": 0}
        regime_data[regime]["scores"].append(a.threat_score.score)
        regime_data[regime]["count"] += 1

    for regime, data in sorted(regime_data.items()):
        s = np.array(data["scores"])
        print(f"  {regime:6s}: n={data['count']:4d}, "
              f"mean_score={s.mean():.1f}, max={s.max():.1f}")

    # Sub-score analysis
    print(f"\n  === Sub-Score Averages ===")
    intent_subs = [a.threat_score.intent_score for a in all_assessments]
    anomaly_subs = [a.threat_score.anomaly_score for a in all_assessments]
    prox_subs = [a.threat_score.proximity_score for a in all_assessments]
    pattern_subs = [a.threat_score.pattern_score for a in all_assessments]
    print(f"  Intent:    {np.mean(intent_subs):.1f}")
    print(f"  Anomaly:   {np.mean(anomaly_subs):.1f}")
    print(f"  Proximity: {np.mean(prox_subs):.1f}")
    print(f"  Pattern:   {np.mean(pattern_subs):.1f}")

    # Top threats
    print(f"\n  === Top 10 Threat Scores ===")
    sorted_idx = np.argsort(scores)[::-1]
    for rank, idx in enumerate(sorted_idx[:10]):
        a = all_assessments[idx]
        print(f"  #{rank+1:2d}: {a.object_id:>6s} score={a.threat_score.score:.1f} "
              f"tier={a.threat_score.tier.value:10s} "
              f"intent={a.intent_result.intent.value}")

    # ------------------------------------------------------------------
    # Step 4: Save results
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Saving results to {RESULTS_DIR}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-object results
    results_list = []
    for a in all_assessments:
        results_list.append({
            "object_id": a.object_id,
            "maneuver_class": a.maneuver_class,
            "maneuver_name": a.maneuver_name,
            "maneuver_confidence": float(a.maneuver_confidence),
            "intent": a.intent_result.intent.value,
            "threat_level": a.intent_result.threat_level.name,
            "anomaly_score": float(a.anomaly_result.anomaly_score) if a.anomaly_result else None,
            "is_anomaly": a.anomaly_result.is_anomaly if a.anomaly_result else None,
            "threat_score": float(a.threat_score.score),
            "threat_tier": a.threat_score.tier.value,
            "intent_sub": float(a.threat_score.intent_score),
            "anomaly_sub": float(a.threat_score.anomaly_score),
            "proximity_sub": float(a.threat_score.proximity_score),
            "pattern_sub": float(a.threat_score.pattern_score),
            "latency_ms": float(a.latency_ms),
        })

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(RESULTS_DIR / "per_object_results.csv", index=False)

    # Summary JSON
    summary = {
        "validation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "dataset": {
            "n_objects": n_objects,
            "timesteps_per_object": len(df) // n_objects,
            "chunk_size": CHUNK_SIZE,
            "n_chunks": n_chunks,
        },
        "pipeline": {
            "anomaly_detector": has_anomaly,
            "init_time_s": float(init_time),
        },
        "performance": {
            "total_time_s": float(total_time),
            "throughput_obj_per_sec": float(n_objects / total_time),
            "latency_mean_ms": float(latencies.mean()),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "latency_max_ms": float(latencies.max()),
        },
        "threat_scores": {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "p50": float(np.median(scores)),
            "p95": float(np.percentile(scores, 95)),
            "p99": float(np.percentile(scores, 99)),
        },
        "tier_distribution": tier_counts,
        "intent_distribution": intent_counts,
        "anomaly_flagged": n_anomalous if has_anomaly else None,
        "sub_score_means": {
            "intent": float(np.mean(intent_subs)),
            "anomaly": float(np.mean(anomaly_subs)),
            "proximity": float(np.mean(prox_subs)),
            "pattern": float(np.mean(pattern_subs)),
        },
    }

    with open(RESULTS_DIR / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: per_object_results.csv ({len(results_list)} rows)")
    print(f"  Saved: validation_summary.json")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("End-to-End Validation Complete!")
    print(f"  Objects assessed:  {n_objects}")
    print(f"  Throughput:        {n_objects/total_time:.0f} obj/sec")
    print(f"  Mean latency:      {latencies.mean():.2f} ms/obj")
    print(f"  P99 latency:       {np.percentile(latencies, 99):.2f} ms/obj")
    print(f"  Mean threat score: {scores.mean():.1f}/100")
    print(f"  CRITICAL threats:  {tier_counts.get('CRITICAL', 0)}")
    print(f"  ELEVATED threats:  {tier_counts.get('ELEVATED', 0)}")
    print(f"  Results:           {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
