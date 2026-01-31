# OpenBrowser AWS Infrastructure (Terraform)

This directory defines AWS infrastructure for OpenBrowser: VPC, backend (EC2 + Docker), API Gateway, Cognito, DynamoDB, S3 + CloudFront for the frontend, and optional Secrets Manager.

## Prerequisites

- [Terraform](https://www.terraform.io/downloads) >= 1.5
- AWS CLI configured (e.g. `aws configure`) or environment variables
- Backend Docker image built and pushed to ECR or Docker Hub

## Quick start

1. **Copy and edit variables**

   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars and set at least:
   # - backend_image (e.g. 123456789012.dkr.ecr.ca-central-1.amazonaws.com/openbrowser-backend:latest)
   ```

2. **Initialize and plan**

   ```bash
   cd terraform
   terraform init
   terraform plan
   ```

3. **Apply**

   ```bash
   terraform apply
   ```

4. **Deploy the frontend**

   Build the Next.js static export and upload to S3:

   ```bash
   cd ../frontend
   npm ci
   npm run build
   aws s3 sync out/ s3://$(terraform -chdir=../terraform output -raw frontend_s3_bucket)/ --delete
   ```

   Then invalidate CloudFront cache so changes are visible immediately (optional):

   ```bash
   aws cloudfront create-invalidation --distribution-id $(terraform -chdir=../terraform output -raw cloudfront_distribution_id) --paths "/*"
   ```

5. **Configure the frontend for production**

   When building the frontend for this environment, set:
   - `NEXT_PUBLIC_API_URL` = Terraform output `api_base_url`
   - `NEXT_PUBLIC_WS_URL` = Terraform output `api_ws_url`

   Example:

   ```bash
   export NEXT_PUBLIC_API_URL=$(terraform -chdir=../terraform output -raw api_base_url)
   export NEXT_PUBLIC_WS_URL=$(terraform -chdir=../terraform output -raw api_ws_url)
   npm run build
   aws s3 sync out/ s3://$(terraform -chdir=../terraform output -raw frontend_s3_bucket)/ --delete
   ```

## What’s included

| Component           | Purpose                                                                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **VPC**             | Public/private subnets (2 AZs), NAT gateway, DynamoDB VPC endpoint                                                                               |
| **DynamoDB**        | Table `{project_name}-sessions` (pk/sk) for future user session/chats                                                                            |
| **EC2 backend**     | Single instance in private subnet; user_data installs Docker and runs your backend image with 2GB shm, DynamoDB and optional Secrets Manager env |
| **ALB**             | Internal ALB with HTTP health check to `/health`, forwards to backend:8000                                                                       |
| **API Gateway**     | HTTP API with VPC Link to ALB; public `GET /health`, optional JWT (Cognito) on other routes                                                      |
| **Cognito**         | User pool, app client, hosted domain (for when you add auth)                                                                                     |
| **Secrets Manager** | Optional placeholder secret for LLM API keys; EC2 fetches at boot and passes to container                                                        |
| **S3 + CloudFront** | S3 bucket for static frontend; CloudFront with OAC, SPA error pages (403/404 → index.html)                                                       |

## Variables

See `variables.tf`. Key ones:

- **backend_image** (required): Container image URI (ECR or Docker Hub).
- **backend_port**: Port the container listens on (default `8000`).
- **secrets_manager_secret_name**: Leave empty to create a placeholder secret; set to an existing secret name/ARN to use it.
- **enable_api_auth**: Set to `true` when you want JWT (Cognito) required on API routes (default `false`).
- **frontend_domain_name** / **frontend_acm_certificate_arn**: Optional custom domain and ACM cert (cert must be in `us-east-1` for CloudFront).

## Auth (future)

- Cognito user pool and app client are created. Set `enable_api_auth = true` when your app uses Cognito and you want API Gateway to enforce JWT.
- Frontend can use the Cognito outputs (`cognito_user_pool_id`, `cognito_app_client_id`, `cognito_domain`) to implement sign-in/sign-up.

## DynamoDB

Table name is output as `dynamodb_table_name`. Use it in your backend (e.g. `DDB_TABLE` is already passed to the container). Design `pk`/`sk` for user sessions and chats when you add that logic.

## Backend image

- Build and push to ECR (recommended):

  ```bash
  aws ecr get-login-password --region ca-central-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.ca-central-1.amazonaws.com
  docker build -t openbrowser-backend -f backend/Dockerfile .
  docker tag openbrowser-backend:latest 123456789012.dkr.ecr.ca-central-1.amazonaws.com/openbrowser-backend:latest
  docker push 123456789012.dkr.ecr.ca-central-1.amazonaws.com/openbrowser-backend:latest
  ```

- Set `backend_image` in `terraform.tfvars` to that URI, then `terraform apply`.

## CloudFront distribution ID (for invalidations)

Add to `outputs.tf` if you want it:

```hcl
output "cloudfront_distribution_id" {
  value = aws_cloudfront_distribution.frontend.id
}
```

Then:

```bash
aws cloudfront create-invalidation --distribution-id $(terraform -chdir=../terraform output -raw cloudfront_distribution_id) --paths "/*"
```
