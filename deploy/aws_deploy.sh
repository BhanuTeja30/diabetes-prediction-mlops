#!/usr/bin/env bash
# Deploy the project to one EC2 server with the AWS CLI. Run from Git Bash on Windows, or Linux/macOS.
#
#   bash deploy/aws_deploy.sh check      show the AWS account and what would be used (creates nothing)
#   bash deploy/aws_deploy.sh provision  create key pair, firewall, server and fixed IP (starts billing)
#   bash deploy/aws_deploy.sh push       upload the project and start it (safe to repeat for updates)
#   bash deploy/aws_deploy.sh status     show the server state, address and services
#   bash deploy/aws_deploy.sh password   print the app's admin login (only on your terminal)
#   bash deploy/aws_deploy.sh tunnel     open MLflow, Grafana, Prometheus and API docs privately
#   bash deploy/aws_deploy.sh ssh        open a shell on the server
#   bash deploy/aws_deploy.sh stop       stop the server (billing drops to storage + IP)
#   bash deploy/aws_deploy.sh start      start it again
#   bash deploy/aws_deploy.sh destroy    delete everything created here (asks first)
#
# Settings: AWS_REGION (default ap-south-1), INSTANCE_TYPE (default t3.small), DOMAIN (your own domain;
# default is a free <ip>.sslip.io address).
set -euo pipefail
export MSYS_NO_PATHCONV=1  # stop Git Bash rewriting arguments such as /aws/service/...

cd "$(dirname "$0")/.."
# Use the browser sign-in made with `aws login --profile deploy`, when there is one.
if command -v aws >/dev/null && aws configure list-profiles 2>/dev/null | tr -d '\r' | grep -qx deploy; then
  export AWS_PROFILE="${AWS_PROFILE:-deploy}"
fi
REGION="${AWS_REGION:-ap-south-1}"
TYPE="${INSTANCE_TYPE:-t3.small}"  # allowed on the AWS Free Plan; the stack needs under 1 GB of RAM
NAME="diabetes-mlops"
STATE="deploy/.state"
KEY="deploy/${NAME}-key.pem"
AMI_PARAM="/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
aws_() { aws --region "$REGION" --output text "$@" | tr -d '\r'; }  # aws.exe on Windows emits CRLF

load() {
  [ -f "$STATE" ] || { echo "No deployment recorded in $STATE. Run provision first." >&2; exit 1; }
  # shellcheck disable=SC1090
  . "$STATE"
}
save() {
  cat > "$STATE" <<EOF
REGION=$REGION
INSTANCE_ID=$INSTANCE_ID
SG_ID=$SG_ID
ALLOC_ID=$ALLOC_ID
PUBLIC_IP=$PUBLIC_IP
EOF
}
site_address() { echo "${DOMAIN:-${PUBLIC_IP//./-}.sslip.io}"; }
ssh_() {
  # Keepalives: long image builds can go quiet, and a silently dropped connection would hang forever.
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "ubuntu@$PUBLIC_IP" "$@"
}

check() {
  command -v aws >/dev/null || { echo "AWS CLI not installed." >&2; exit 1; }
  echo "Account ending: $(aws sts get-caller-identity --query Account --output text | tr -d '\r' | tail -c 5)"
  echo "Region: $REGION | Server type: $TYPE | Disk: 30 GB gp3"
  echo "Ubuntu image: $(aws_ ssm get-parameter --name "$AMI_PARAM" --query Parameter.Value)"
  echo "Default network: $(aws_ ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId')"
  [ -f "$STATE" ] && echo "Existing deployment recorded in $STATE" || echo "No deployment recorded yet."
}

provision() {
  [ ! -f "$STATE" ] || { echo "Already provisioned ($STATE exists)." >&2; exit 1; }
  local my_ip vpc ami tags
  my_ip="$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')"
  vpc="$(aws_ ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId')"
  ami="$(aws_ ssm get-parameter --name "$AMI_PARAM" --query Parameter.Value)"
  tags="Key=Name,Value=$NAME"

  echo "Creating key pair (saved to $KEY)..."
  aws_ ec2 create-key-pair --key-name "$NAME-key" --query KeyMaterial > "$KEY"
  chmod 400 "$KEY"

  echo "Creating firewall: SSH from $my_ip only, web from anywhere..."
  SG_ID="$(aws_ ec2 create-security-group --group-name "$NAME-sg" --vpc-id "$vpc" \
    --description "Diabetes MLOps: web from anywhere, SSH from owner" --query GroupId)"
  aws_ ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$my_ip/32" >/dev/null
  aws_ ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 80 --cidr 0.0.0.0/0 >/dev/null
  aws_ ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 443 --cidr 0.0.0.0/0 >/dev/null

  echo "Starting $TYPE server..."
  INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$ami" --instance-type "$TYPE" --key-name "$NAME-key" \
    --security-group-ids "$SG_ID" --metadata-options HttpTokens=required \
    --credit-specification CpuCredits=standard \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3,Encrypted=true}' \
    --tag-specifications "ResourceType=instance,Tags=[{$tags}]" "ResourceType=volume,Tags=[{$tags}]" \
    --query 'Instances[0].InstanceId')"
  aws_ ec2 wait instance-running --instance-ids "$INSTANCE_ID"

  echo "Attaching a fixed public IP..."
  ALLOC_ID="$(aws_ ec2 allocate-address --domain vpc \
    --tag-specifications "ResourceType=elastic-ip,Tags=[{$tags}]" --query AllocationId)"
  aws_ ec2 associate-address --instance-id "$INSTANCE_ID" --allocation-id "$ALLOC_ID" >/dev/null
  PUBLIC_IP="$(aws_ ec2 describe-addresses --allocation-ids "$ALLOC_ID" --query 'Addresses[0].PublicIp')"
  save
  echo "Provisioned $INSTANCE_ID at $PUBLIC_IP. Next: bash deploy/aws_deploy.sh push"
}

push() {
  load
  local bundle
  bundle="$(mktemp)"
  echo "Packing project (no .env, user accounts, local logs or Git history)..."
  tar --exclude='__pycache__' -czf "$bundle" app src configs monitoring deploy/Caddyfile deploy/server-setup.sh \
    models/production data/processed reports/evaluation \
    Dockerfile Dockerfile.api Dockerfile.mlflow docker-compose.yml docker-compose.prod.yml \
    requirements.txt requirements-api.txt requirements-frontend.txt .env.example .dockerignore
  echo "Waiting for SSH..."
  for _ in $(seq 1 30); do ssh_ true 2>/dev/null && break; sleep 5; done
  ssh_ "mkdir -p ~/diabetes"
  scp -i "$KEY" -o StrictHostKeyChecking=accept-new "$bundle" "ubuntu@$PUBLIC_IP:/tmp/diabetes.tgz"
  rm -f "$bundle"
  ssh_ "cd ~/diabetes && tar --no-same-owner -xzf /tmp/diabetes.tgz && rm /tmp/diabetes.tgz && bash deploy/server-setup.sh '$(site_address)'"
  echo "Live at: https://$(site_address)"
}

status() {
  load
  echo "Server: $(aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name')"
  echo "Address: https://$(site_address)"
  ssh_ "cd ~/diabetes && sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml ps --format 'table {{.Service}}\t{{.Status}}'" || true
}

case "${1:-}" in
  check) check ;;
  provision) provision ;;
  push) push ;;
  status) status ;;
  password) load; ssh_ "grep -E '^(ADMIN_USERNAME|ADMIN_PASSWORD|GRAFANA_ADMIN_PASSWORD)=' ~/diabetes/.env" ;;
  ssh) load; ssh -i "$KEY" "ubuntu@$PUBLIC_IP" ;;
  tunnel)
    load
    API_IP="$(ssh_ "sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' \$(cd ~/diabetes && sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml ps -q api)")"
    echo "While this runs, open: MLflow http://localhost:5001 | Grafana http://localhost:3002 |"
    echo "Prometheus http://localhost:9091 | API docs http://localhost:8001/docs. Press Ctrl+C to close."
    ssh -i "$KEY" -N -L 5001:127.0.0.1:5000 -L 3002:127.0.0.1:3000 -L 9091:127.0.0.1:9090 \
      -L "8001:$API_IP:8000" "ubuntu@$PUBLIC_IP"
    ;;
  stop) load; aws_ ec2 stop-instances --instance-ids "$INSTANCE_ID" >/dev/null; echo "Stopping. Storage and IP still bill." ;;
  start) load; aws_ ec2 start-instances --instance-ids "$INSTANCE_ID" >/dev/null; echo "Starting: https://$(site_address)" ;;
  destroy)
    load
    read -r -p "Permanently delete the server, its disk (user accounts, MLflow runs) and IP? Type DELETE: " answer
    [ "$answer" = "DELETE" ] || { echo "Cancelled."; exit 1; }
    aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null
    aws_ ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"
    aws_ ec2 release-address --allocation-id "$ALLOC_ID"
    aws_ ec2 delete-security-group --group-id "$SG_ID"
    aws_ ec2 delete-key-pair --key-name "$NAME-key"
    rm -f "$KEY" "$STATE"
    echo "All AWS resources removed."
    ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
