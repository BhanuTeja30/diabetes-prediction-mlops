#!/usr/bin/env bash
# Runs ON the EC2 server from the project folder: installs Docker, creates .env once, starts the stack.
# Usage: bash deploy/server-setup.sh <site-address>
set -euo pipefail

SITE_ADDRESS="${1:?usage: server-setup.sh <site-address>}"
COMPOSE=(sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml)

# Safety margin for image builds and memory spikes on a small server.
if [ ! -f /swapfile ]; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

if ! command -v docker >/dev/null; then
  echo "Installing Docker..."
  curl -fsSL https://get.docker.com | sudo sh
fi

# Secrets are generated here and stay on the server. Existing values survive re-deploys.
if [ ! -f .env ]; then
  secret() { python3 -c "import secrets; print(secrets.token_urlsafe($1))"; }
  sed -e "s|^AUTH_SECRET_KEY=.*|AUTH_SECRET_KEY=$(secret 48)|" \
      -e "s|^ADMIN_PASSWORD=.*|ADMIN_PASSWORD=Admin-$(secret 12)|" \
      -e "s|^GRAFANA_ADMIN_PASSWORD=.*|GRAFANA_ADMIN_PASSWORD=$(secret 18)|" \
      .env.example > .env
  chmod 600 .env
  echo "Generated new secrets in .env"
fi
sed -i '/^SITE_ADDRESS=/d' .env
echo "SITE_ADDRESS=${SITE_ADDRESS}" >> .env

# Container services run as UID 1000 and write to these folders.
mkdir -p data/auth data/monitoring reports/drift
sudo chown -R 1000:1000 data/auth data/monitoring reports/drift

"${COMPOSE[@]}" up -d --build

echo "Waiting for the API to report healthy..."
for _ in $(seq 1 60); do
  if sudo docker inspect --format '{{.State.Health.Status}}' "$(${COMPOSE[*]} ps -q api)" 2>/dev/null | grep -q healthy; then
    "${COMPOSE[@]}" ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}"
    echo "Stack is up."
    exit 0
  fi
  sleep 5
done
"${COMPOSE[@]}" logs --tail 30 api
echo "ERROR: API did not become healthy." >&2
exit 1
