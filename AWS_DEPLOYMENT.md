# AWS deployment

The Terraform stack creates a dedicated VPC, two public subnets, an HTTPS ALB,
single-task ECS Fargate service, encrypted EFS storage for Chroma, ECR repository,
CloudWatch logs, Secrets Manager access, health checks, and least-privilege network
boundaries. It also creates a dashboard and SNS-backed alarms for 5xx errors, p95 latency,
and ECS CPU. The service runs as a non-root user.

## Important architecture constraint

The stack deliberately runs one task because embedded Chroma on EFS is not a safe
multi-writer horizontally scaled vector service. This is suitable for an interview demo
and low-volume deployment. Before scaling beyond one task, migrate retrieval to a managed
vector service such as OpenSearch Serverless, Aurora PostgreSQL with pgvector, or another
service with concurrency, backups, and tenant-aware authorization.

## Prerequisites

- Terraform 1.6+
- AWS CLI authenticated to your account
- A Route 53 hosted zone and ACM certificate in the deployment region
- A GitHub OIDC deployment role
- Two Secrets Manager secrets containing plain secret strings:
  - `medhub/openai`: OpenAI API key
  - `medhub/api-key`: a generated API authentication key
- An S3 Terraform-state bucket and DynamoDB locking table

Create the secrets without exposing them in shell history by entering their values in the
AWS console, or use an interactive secure input workflow.

## Deploy infrastructure

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# Replace account IDs, secret ARNs, domain, hosted-zone ID, certificate ARN, and CIDR.
terraform init \
  -backend-config="bucket=YOUR_TERRAFORM_STATE_BUCKET" \
  -backend-config="key=medhub-lite/production.tfstate" \
  -backend-config="region=us-east-1" \
  -backend-config="dynamodb_table=YOUR_TERRAFORM_LOCK_TABLE" \
  -backend-config="encrypt=true"
# This account already has the repository, so import it instead of creating it:
terraform import aws_ecr_repository.app medhub-lite
terraform plan
terraform apply
```

Your AWS account already contains `983004127584.dkr.ecr.us-east-1.amazonaws.com/medhub-lite`
with a `latest` image, so importing it allows the ECS service to start during the first
full apply. If deploying to a different account with an empty repository, first target
the ECR resource, push `latest`, and only then apply the full stack. Never apply until the
plan shows that it will not replace resources you need to preserve.

## Configure GitHub Actions

Add one GitHub Actions secret:

- `AWS_ROLE_ARN`: ARN of an IAM role trusted by GitHub OIDC and restricted to this
  repository and the `main` branch.

The role needs ECR push permissions, ECS task-definition registration and service update
permissions, `ecs:DescribeServices`, and `iam:PassRole` restricted to the MedHub task and
execution roles. The workflow runs all tests, publishes SHA and `latest` tags, registers a
new task-definition revision pinned to the immutable SHA image, deploys it, and waits for
service stability.

## Verify

```bash
curl https://YOUR_DOMAIN/health/live
curl https://YOUR_DOMAIN/health/ready
curl -H "X-API-Key: YOUR_KEY" https://YOUR_DOMAIN/metrics
```

All data endpoints also accept `X-Tenant-ID`. The default tenant is `default`; production
clients should always send an explicit tenant identifier.

The shared `API_AUTH_KEY` mode is intended for a private demo or administrative client.
`TENANT_API_KEYS_JSON` can bind separate demo keys to tenant IDs. For a real multi-user
product, put the service behind Cognito or another OIDC provider and derive tenant and
entitlement claims from a verified JWT rather than trusting caller-provided identity.

## Upgrading the previous deployment

This version adds tenant metadata and tenant-qualified chunk IDs. Before replacing an
existing task, back up the Chroma EFS data and stop all writers. Then run:

```bash
python migrate_tenant_metadata.py          # dry run
python migrate_tenant_metadata.py --apply  # one-time migration
```

Alternatively, reset and re-index the source documents. Do not deploy the tenant-filtered
version over an old collection without migration or re-indexing; existing chunks would be
intentionally invisible to tenant-scoped queries.
