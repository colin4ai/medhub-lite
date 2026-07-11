# GitHub and deployment workflow

The repository already exists at `github.com/colin4ai/medhub-lite`. Normal development
should use a feature branch and pull request so `.github/workflows/ci.yml` validates:

- the complete Python test suite;
- Ruff static checks;
- Python compilation;
- Terraform formatting and provider validation;
- the production Docker build.

The production workflow runs only after a push to `main`. It publishes both the commit
SHA and `latest` image tags, creates a new ECS task-definition revision pinned to the SHA,
deploys it, and waits for ECS stability.

Do not push the current deployment workflow to `main` until all of these exist:

1. The Terraform-managed ECS service and task definition.
2. A GitHub Actions secret named `AWS_ROLE_ARN`.
3. A GitHub OIDC role scoped to `colin4ai/medhub-lite` and `main`.
4. ECR push, ECS registration/update, and restricted `iam:PassRole` permissions.
5. Secrets Manager entries, HTTPS domain/certificate, remote Terraform state, and alarms.

The AWS account currently contains the `medhub-lite` ECR repository but no running ECS
service. Import the repository into Terraform rather than attempting to recreate it:

```bash
terraform -chdir=infra/terraform import aws_ecr_repository.app medhub-lite
```

Follow `AWS_DEPLOYMENT.md` and review `terraform plan` before creating billable resources.
Never commit `.env`, `terraform.tfvars`, Terraform state, API keys, or OpenAI keys.
