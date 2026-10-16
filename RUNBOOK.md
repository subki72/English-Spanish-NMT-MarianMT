# OPERATIONAL RUNBOOK: ENGLISH-SPANISH NMT SERVICE

**Service Name**: `english-spanish-nmt`  
**Classification**: Tier-1 Machine Learning Inference Microservice  
**Engine**: CTranslate2 INT8 Quantized MarianMT (`Helsinki-NLP/opus-mt-en-es`)  
**Architecture**: FastAPI ASGI + AnyIO Worker Threadpool + In-Memory Dynamic Batching  
**Default Port**: 8000  

---

## 1. Service Level Objectives (SLO) & Target SLAs

| Metric / Objective | Production Target | Alerting Threshold | PromQL Trigger |
|---|---|---|---|
| **Availability** | 99.9% monthly uptime | < 99.9% over 1 hour | `sum(rate(nmt_requests_total{status=~"2.."}[5m])) / sum(rate(nmt_requests_total[5m])) < 0.999` |
| **P50 Latency** | $\le 35$ ms | > 50 ms | `histogram_quantile(0.50, sum(rate(nmt_translation_latency_seconds_bucket[5m])) by (le)) > 0.05` |
| **P95 Latency** | $\le 90$ ms | > 90 ms | `histogram_quantile(0.95, sum(rate(nmt_translation_latency_seconds_bucket[5m])) by (le)) > 0.09` |
| **P99 Latency** | $\le 180$ ms | > 200 ms | `histogram_quantile(0.99, sum(rate(nmt_translation_latency_seconds_bucket[5m])) by (le)) > 0.20` |
| **Error Rate (5xx)** | $< 0.05\%$ | > 0.05% | `sum(rate(nmt_requests_total{status=~"5.."}[5m])) / sum(rate(nmt_requests_total[5m])) > 0.0005` |
| **Throughput** | $\ge 40$ RPS / pod | < 20 RPS under load | Evaluated via dynamic batching metrics |
| **Cold-Start Time** | $\le 8$ seconds | > 15 seconds | Pod readiness check duration |

---

## 2. Metrics & Alerting Rules (Prometheus Alertmanager)

Place the following rules in your Prometheus alert rule configuration:

```yaml
groups:
  - name: nmt_service_alerts
    rules:
      - alert: NmtHighP95Latency
        expr: histogram_quantile(0.95, sum(rate(nmt_translation_latency_seconds_bucket[5m])) by (le)) > 0.09
        for: 3m
        labels:
          severity: warning
          service: english-spanish-nmt
        annotations:
          summary: "P95 translation latency exceeded 90ms"
          description: "P95 inference latency is currently {{ $value | humanizeDuration }}. Check CPU throttle or batch saturation."

      - alert: NmtCriticalErrorRate
        expr: sum(rate(nmt_requests_total{status=~"5.."}[3m])) / sum(rate(nmt_requests_total[3m])) > 0.0005
        for: 2m
        labels:
          severity: critical
          service: english-spanish-nmt
        annotations:
          summary: "HTTP 5xx error rate exceeded 0.05%"
          description: "5xx error rate is currently {{ $value | humanizePercentage }}. Immediate incident response required."

      - alert: NmtBatchQueueSaturation
        expr: nmt_in_flight_requests > 64
        for: 1m
        labels:
          severity: warning
          service: english-spanish-nmt
        annotations:
          summary: "Dynamic batcher queue saturation"
          description: "In-flight translation requests ({{ $value }}) exceed batcher capacity. Consider horizontal scaling."

      - alert: NmtReadinessFailure
        expr: kube_deployment_status_replicas_ready{deployment="nmt-service"} < kube_deployment_spec_replicas{deployment="nmt-service"}
        for: 2m
        labels:
          severity: critical
          service: english-spanish-nmt
        annotations:
          summary: "NMT Pod readiness failure"
          description: "One or more NMT pods failed /readyz probe for more than 2 minutes."
```

---

## 3. Standard Operating Procedures (SOP)

### SOP-01: Health & Diagnostics Verification
Check real-time health and model readiness:
```bash
# Liveness probe (process is running)
curl -s http://localhost:8000/healthz | jq .

# Readiness probe (model, tokenizers, and dynamic batcher are operational)
curl -s http://localhost:8000/readyz | jq .

# Prometheus metrics scrape
curl -s http://localhost:8000/metrics | grep nmt_
```

Expected healthy response from `/readyz`:
```json
{
  "ready": true,
  "model_loaded": true,
  "tokenizer_loaded": true,
  "batcher_running": true,
  "details": null
}
```

### SOP-02: Manual Scaling & Autoscaler Inspection
Inspect and scale Kubernetes replicas:
```bash
# Check current HPA status
kubectl get hpa nmt-service-hpa

# Check pod resource utilization
kubectl top pods -l app=nmt-service

# Temporary manual scale override during traffic surges
kubectl scale deployment nmt-service --replicas=6
```

### SOP-03: Rate Limiting & Throttling Configuration
Rate limiting is controlled by environment variables without recompilation:
- `ENABLE_RATE_LIMITING`: `true` / `false`
- `RATE_LIMIT_REQUESTS`: Default `60` requests per window per IP
- `RATE_LIMIT_WINDOW_SECONDS`: Default `60` seconds

To adjust limits dynamically on Kubernetes:
```bash
kubectl set env deployment/nmt-service RATE_LIMIT_REQUESTS=120 RATE_LIMIT_WINDOW_SECONDS=60
```

### SOP-04: Batcher Tuning (Latency vs. Throughput)
- **High Throughput Mode** (Batch processing / background jobs):
  - `BATCH_SIZE=64`, `MAX_WAIT_MS=20.0`
- **Low Latency Mode** (Interactive UI / chat translation):
  - `BATCH_SIZE=16`, `MAX_WAIT_MS=5.0`
- Default balanced production settings:
  - `BATCH_SIZE=32`, `MAX_WAIT_MS=10.0`, `INTRA_THREADS=4`, `INTER_THREADS=1`

---

## 4. Incident Response Playbooks

### Playbook A: P95 Latency Breach (> 90ms)
1. **Identify bottleneck**: Check if CPU throttling is active:
   ```bash
   kubectl top pods -l app=nmt-service
   ```
2. **Inspect Structured Logs**:
   Filter access logs for slow operations (`latency_ms > 90`):
   ```bash
   kubectl logs -l app=nmt-service --tail=200 | jq 'select(.latency_ms > 90)'
   ```
3. **Check Dynamic Batching histogram**:
   Inspect `nmt_batch_size_sentences_bucket` on `/metrics`. If batch sizes are maxed out and requests are piling up, scale pods horizontally:
   ```bash
   kubectl scale deployment nmt-service --replicas=5
   ```
4. **Tune OpenMP threads**: If running on nodes with fewer dedicated cores, set `INTRA_THREADS=2`.

### Playbook B: High Error Rate (5xx > 0.05%)
1. **Find error signatures**:
   Search structured logs for error entries:
   ```bash
   kubectl logs -l app=nmt-service --tail=500 | jq 'select(.status_code >= 500)'
   ```
2. **Identify Correlation ID**:
   Extract `request_id` from the error log and trace the input payload:
   ```bash
   kubectl logs -l app=nmt-service | grep "<request_id>"
   ```
3. **Verify Model Integrity**:
   Check if the model binary is intact or experiencing segmentation faults:
   ```bash
   kubectl describe pod -l app=nmt-service
   ```
4. **Immediate Recovery**:
   Perform rolling restart if memory corruption or deadlock is suspected:
   ```bash
   kubectl rollout restart deployment/nmt-service
   ```

### Playbook C: Rate Limiting 429 Surge
1. Check structured logs for IPs triggering rate limits:
   ```bash
   kubectl logs -l app=nmt-service --tail=500 | jq 'select(.status_code == 429) | .client_ip' | sort | uniq -c | sort -nr
   ```
2. If traffic is from a legitimate business client or API gateway:
   - Verify `X-Forwarded-For` header propagation through ingress.
   - Whitelist the gateway IP or increase `RATE_LIMIT_REQUESTS`.
3. If traffic is abusive:
   - Apply IP block rule at the Ingress / Cloudflare / WAF layer.

---

## 5. Cold-Start Mitigation & Pre-warming

To ensure container readiness within $\le 8$ seconds and prevent 504 Gateway Timeouts during scaling events:

1. **Pre-baked Image Assets**:
   Model binaries (`model.bin`, `source.spm`, `target.spm`, `shared_vocabulary.json`) are pre-baked into the Docker image at `/app/models/opus-mt-en-es-ct2-int8`. No network download occurs during container boot.
2. **Kubernetes Startup Probe**:
   A dedicated `startupProbe` with `failureThreshold: 12` and `periodSeconds: 5` gives the container up to 60 seconds to initialize CTranslate2 engine and memory-map weights before `livenessProbe` begins evaluation.
3. **Engine Pre-warming**:
   The engine automatically attempts to load the model on startup during FastAPI's lifespan initialization. An optional synthetic warm-up inference can be executed during container startup:
   ```python
   # Evaluates one warmup translation during lifespan
   engine.translate_batch(["Warmup sentence."], beam_size=1)
   ```

---

## 6. Backup, Rollback & Disaster Recovery SOP

### 6.1 Canary & Rolling Rollback
If a newly deployed image version causes performance regression or errors:
```bash
# Check rollout history
kubectl rollout history deployment/nmt-service

# Roll back immediately to previous revision
kubectl rollout undo deployment/nmt-service

# Verify rollback status
kubectl rollout status deployment/nmt-service
```

### 6.2 Weight Integrity & Checksum Verification
Model assets stored in `models/opus-mt-en-es-ct2-int8/` must match pre-computed SHA256 checksums:
```bash
# Generate and verify checksums
sha256sum models/opus-mt-en-es-ct2-int8/* > models/checksums.sha256
sha256sum -c models/checksums.sha256
```

### 6.3 Disaster Recovery Model Regeneration
If model weights are corrupted, re-convert directly from HuggingFace base model:
```bash
python scripts/convert_model.py
python scripts/evaluate_bleu.py --sample-size 500
```
Verify BLEU score is $\ge 40.0$ before redeploying.
