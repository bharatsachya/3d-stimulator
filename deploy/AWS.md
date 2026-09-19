# Deploying to AWS

Target: EC2 `m7i-flex.large` — 2 vCPU, 8 GB, no GPU, Ubuntu 24.04, in
**ap-northeast-2 (Seoul)**.

Seoul is deliberate: the reviewer is in Seoul, and while the processing time is
measured server-side and unaffected by geography, a ~200 ms round trip on every
poll makes an otherwise responsive UI feel sluggish.

No GPU anywhere in this document is an oversight. Sparse SLAM is CPU work, and
the measured evidence is that this workload stops scaling past two threads — see
`docs/measurements.md`. Two vCPUs is not a handicap here, it is sufficiency.

---

## 0. Prerequisites

A configured AWS profile with EC2 permissions:

```bash
aws configure --profile slam      # personal account, not a shared one
export AWS_PROFILE=slam
export AWS_DEFAULT_REGION=ap-northeast-2
aws sts get-caller-identity       # confirm which account you are in
```

---

## 1. Key pair and security group

```bash
aws ec2 create-key-pair --key-name slam-key \
  --query KeyMaterial --output text > ~/.ssh/slam-key.pem
chmod 400 ~/.ssh/slam-key.pem

SG=$(aws ec2 create-security-group --group-name slam-sg \
  --description "Monocular SLAM demo" --query GroupId --output text)

# HTTP open to the world: the reviewer has to be able to open the URL.
aws ec2 authorize-security-group-ingress --group-id $SG \
  --protocol tcp --port 80 --cidr 0.0.0.0/0

# SSH restricted to your own address. Port 22 open to 0.0.0.0/0 is scanned
# within minutes of an instance coming up.
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id $SG \
  --protocol tcp --port 22 --cidr ${MY_IP}/32
```

Port 8000 is deliberately **not** opened. uvicorn binds to 127.0.0.1 and only
nginx talks to it.

---

## 2. Launch

```bash
AMI=$(aws ssm get-parameters \
  --names /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query 'Parameters[0].Value' --output text)

aws ec2 run-instances \
  --image-id $AMI \
  --instance-type m7i-flex.large \
  --key-name slam-key \
  --security-group-ids $SG \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=slam-assignment2}]'
```

An **elastic IP** is worth attaching: without one the public IP changes on every
stop/start, and the URL in a submitted README stops working.

```bash
ALLOC=$(aws ec2 allocate-address --query AllocationId --output text)
aws ec2 associate-address --instance-id <id> --allocation-id $ALLOC
```

> Elastic IPs are billed when **not** attached to a running instance. Release it
> when the instance is terminated, or it quietly accrues charges.

---

## 3. Provision

```bash
ssh -i ~/.ssh/slam-key.pem ubuntu@<elastic-ip>

sudo apt-get update
# python3-venv is NOT included in Ubuntu's python3 package; without it
# `python3 -m venv` fails with a message that does not name the missing package.
sudo apt-get install -y python3-venv python3-pip nginx git

sudo useradd --system --home /opt/slam --shell /usr/sbin/nologin slam
sudo mkdir -p /opt/slam
sudo chown $USER: /opt/slam
git clone <repo-url> /opt/slam
cd /opt/slam

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
sudo chown -R slam: /opt/slam

sudo cp deploy/slam.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now slam

sudo cp deploy/nginx.conf /etc/nginx/sites-available/slam
sudo ln -sf /etc/nginx/sites-available/slam /etc/nginx/sites-enabled/slam
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

curl -s localhost/health
```

`opencv-python-headless` installs from a wheel, so there is no compile step and
no `libgl1` to chase. That is the entire reason for preferring it over
`opencv-python` on a server — see the note in `requirements.txt`. Measured on the
instance: the whole dependency install takes **10.6 s** and peaks at 157 MB RSS.

### Never copy a virtualenv between directories

If you provision by copying files rather than `git clone`, exclude `.venv`.

A venv's console scripts hardcode an **absolute** interpreter path in their
shebang. Copying `~/slam/.venv` to `/opt/slam/.venv` leaves
`#!/home/ubuntu/slam/.venv/bin/python3` at the top of `bin/uvicorn`, and since
`/home/ubuntu` is mode 750 the `slam` service user cannot traverse it. systemd
then reports:

```
Failed to execute /opt/slam/.venv/bin/uvicorn: Permission denied
status=203/EXEC
```

which is thoroughly misleading — `ls -l` shows `-rwxr-xr-x slam slam` and even
`test -x` passes, because the permission being denied belongs to the
*interpreter named in the shebang*, not to the script. Delete the copied venv
and recreate it at its final path:

```bash
sudo rm -rf /opt/slam/.venv
sudo python3 -m venv /opt/slam/.venv
sudo /opt/slam/.venv/bin/pip install -r /opt/slam/requirements.txt
sudo chown -R slam: /opt/slam
```

---

## 4. Measure on the instance, not on a laptop

This is the step the assignment actually grades.

```bash
cd /opt/slam
./.venv/bin/python tools/make_synthetic.py --motion dolly   # regenerate the clip
./.venv/bin/python tools/probe.py samples/synth_dolly.mp4 --json out/probe-ec2.json
```

Copy the resulting table into `docs/measurements.md`. Laptop numbers are shape-
finding only and must never be quoted as the submission's timings.

---

## 5. Cost

`m7i-flex.large` is roughly $0.09/hour on-demand, about $2/day, plus a few cents
for the 20 GB gp3 volume. Set a budget alarm before walking away from it:

```bash
aws budgets create-budget --account-id <id> --budget \
  '{"BudgetName":"slam","BudgetLimit":{"Amount":"20","Unit":"USD"},
    "TimeUnit":"MONTHLY","BudgetType":"COST"}'
```

Free-tier eligibility depends on the account. Check the EC2 console's own label
for the account you are launching in rather than assuming either way.

---

## 6. HTTPS

Served at **https://13.63.181.231.sslip.io/**, with HTTP redirecting to it.

A publicly-trusted certificate is not issued for a bare IP address by the
ordinary ACME path, so TLS needs a DNS name. `sslip.io` is a free wildcard
resolver requiring no registration or account: `13.63.181.231.sslip.io` resolves
to `13.63.181.231` by construction, which is all Let's Encrypt's HTTP-01
challenge needs.

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot certonly --nginx -d 13.63.181.231.sslip.io \
  --non-interactive --agree-tos --register-unsafely-without-email
sudo cp deploy/nginx.conf /etc/nginx/sites-available/slam
sudo nginx -t && sudo systemctl reload nginx
```

`certonly`, not `--nginx`, is deliberate. Letting certbot rewrite the live nginx
config would work once and then break silently: this repository redeploys
`deploy/nginx.conf` over it, reverting the TLS server block with no obvious
cause. The config is the source of truth; certbot only obtains the certificate.

Two things that cost time here and are worth knowing:

- `http2 on;` is nginx 1.25.1+ syntax. Ubuntu 24.04 ships **1.24.0**, where it
  fails validation outright with `unknown directive "http2"`. Use
  `listen 443 ssl http2;`.
- The port-80 server block must serve `/.well-known/acme-challenge/` **before**
  redirecting to HTTPS. Renewal uses HTTP-01 over port 80, so a blanket redirect
  makes the certificate stop renewing in 60 days, long after anyone is watching.

Renewal is handled by the `certbot.timer` systemd unit installed with the
package; verify with `sudo certbot renew --dry-run`.
