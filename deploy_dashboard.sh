#!/usr/bin/env bash
# Deploy the read-only dashboard as a Cloud Run SERVICE.
# Every gcloud command is printed before it runs; any failure aborts.
set -euo pipefail

PROJECT="gen-lang-client-0950032703"
REGION="us-east1"
BUCKET="gapfinder-state"
SERVICE="gapfinder-dashboard"
REPO="gapfinder"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/dashboard:latest"
HERE="$(cd "$(dirname "$0")" && pwd)"

run() { echo; echo "+ $*"; "$@"; }
step() { echo; echo "=============== $* ==============="; }

command -v gcloud >/dev/null || { echo "FATAL: gcloud not installed"; exit 1; }

step "0. project"
run gcloud config set project "$PROJECT"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
echo "  runtime SA: $RUNTIME_SA"

step "1. APIs"
run gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  storage.googleapis.com artifactregistry.googleapis.com

step "2. registry"
if gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1; then
  echo "  repo $REPO already exists"
else
  run gcloud artifacts repositories create "$REPO" --repository-format=docker \
    --location="$REGION" --description="gapfinder images"
fi

step "3. read-only bucket access for the dashboard"
# objectViewer, not objectAdmin: the dashboard has no write endpoints and must not
# be able to mutate state even if it is compromised.
run gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/storage.objectViewer"

step "4. build via Cloud Build"
run gcloud builds submit "$HERE/dashboard" --tag "$IMAGE"

step "5. deploy Cloud Run service"
run gcloud run deploy "$SERVICE" \
  --image="$IMAGE" \
  --region="$REGION" \
  --service-account="$RUNTIME_SA" \
  --allow-unauthenticated \
  --set-env-vars="STATE_BUCKET=${BUCKET}" \
  --memory=1Gi --cpu=1 --timeout=120s --max-instances=3

step "done"
URL="$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')"
echo
echo "  PUBLIC URL: $URL"
echo "  api check : curl -s $URL/api/status"
