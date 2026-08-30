#!/usr/bin/env bash
# Deploy the gapfinder daily cycle to Cloud Run Jobs + Cloud Scheduler.
# Every gcloud command is printed before it runs; any failure aborts the script.
set -euo pipefail

PROJECT="gen-lang-client-0950032703"
REGION="us-east1"
BUCKET="gapfinder-state"
JOB="gapfinder-agent"
REPO="gapfinder"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/agent:latest"
SCHED="gapfinder-daily"
HERE="$(cd "$(dirname "$0")" && pwd)"

run() { echo; echo "+ $*"; "$@"; }
step() { echo; echo "=============== $* ==============="; }

command -v gcloud >/dev/null || { echo "FATAL: gcloud not installed"; exit 1; }
[ -f "$HERE/.env" ] || { echo "FATAL: .env not found (needed for secret values)"; exit 1; }

step "0. project"
run gcloud config set project "$PROJECT"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
echo "  project number: $PROJECT_NUMBER"
echo "  runtime SA:     $RUNTIME_SA"

step "1. enable APIs"
run gcloud services enable \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com

step "2. state bucket"
if gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  echo "  gs://${BUCKET} already exists"
else
  run gcloud storage buckets create "gs://${BUCKET}" --location="$REGION" --uniform-bucket-level-access
fi
run gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/storage.objectAdmin"

step "3. seed state from local"
for f in agent_state.json grid.json corpus_live.csv extractions.jsonl events_log.jsonl watch_config.json latest_alert.md; do
  if [ -f "$HERE/$f" ]; then
    run gcloud storage cp "$HERE/$f" "gs://${BUCKET}/state/$f"
  else
    echo "  (skip $f -- not present locally)"
  fi
done
if compgen -G "$HERE/state_history/*" >/dev/null; then
  run gcloud storage cp "$HERE"/state_history/* "gs://${BUCKET}/state_history/"
fi

step "4. secrets (values read from .env, never echoed, never in the image)"
for KEY in GEMINI_API_KEY NCBI_API_KEY; do
  VALUE="$(grep -E "^${KEY}=" "$HERE/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' \r\n')"
  [ -n "$VALUE" ] || { echo "FATAL: $KEY missing from .env"; exit 1; }
  if gcloud secrets describe "$KEY" >/dev/null 2>&1; then
    echo "  secret $KEY exists -- adding a new version"
  else
    run gcloud secrets create "$KEY" --replication-policy=automatic
  fi
  echo "+ printf '%s' \"\$$KEY\" | gcloud secrets versions add $KEY --data-file=-"
  printf '%s' "$VALUE" | gcloud secrets versions add "$KEY" --data-file=-
  run gcloud secrets add-iam-policy-binding "$KEY" \
    --member="serviceAccount:${RUNTIME_SA}" --role="roles/secretmanager.secretAccessor"
done

step "5. artifact registry"
if gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1; then
  echo "  repo $REPO already exists"
else
  run gcloud artifacts repositories create "$REPO" --repository-format=docker \
    --location="$REGION" --description="gapfinder agent images"
fi

step "6. build image via Cloud Build"
run gcloud builds submit "$HERE" --tag "$IMAGE"

step "7. Cloud Run job"
JOB_ARGS=(
  --image="$IMAGE"
  --region="$REGION"
  --service-account="$RUNTIME_SA"
  --task-timeout=1800s
  --max-retries=1
  --memory=2Gi
  --cpu=1
  --set-env-vars="STATE_BUCKET=${BUCKET},GEMINI_MODEL=gemini-3.5-flash-lite"
  --set-secrets="GEMINI_API_KEY=GEMINI_API_KEY:latest,NCBI_API_KEY=NCBI_API_KEY:latest"
)
if gcloud run jobs describe "$JOB" --region="$REGION" >/dev/null 2>&1; then
  run gcloud run jobs update "$JOB" "${JOB_ARGS[@]}"
else
  run gcloud run jobs create "$JOB" "${JOB_ARGS[@]}"
fi

step "8. Cloud Scheduler -- daily 07:00 America/New_York"
URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT}/jobs/${JOB}:run"
run gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/run.invoker"
SCHED_ARGS=(
  --schedule="0 7 * * *"
  --time-zone="America/New_York"
  --uri="$URI"
  --http-method=POST
  --oauth-service-account-email="$RUNTIME_SA"
  --location="$REGION"
)
if gcloud scheduler jobs describe "$SCHED" --location="$REGION" >/dev/null 2>&1; then
  run gcloud scheduler jobs update http "$SCHED" "${SCHED_ARGS[@]}"
else
  run gcloud scheduler jobs create http "$SCHED" "${SCHED_ARGS[@]}"
fi

step "done"
cat <<EOF

Deployed.
  image     $IMAGE
  job       $JOB (region $REGION)
  schedule  $SCHED -- 07:00 America/New_York daily
  state     gs://${BUCKET}/state/

Run it once now and watch it:
  gcloud run jobs execute $JOB --region=$REGION --wait
  gcloud beta run jobs logs tail $JOB --region=$REGION
  gcloud storage cat gs://${BUCKET}/state/agent_state.json
EOF
