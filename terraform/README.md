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
   # backend_image can be left empty to use the ECR repo Terraform creates (see "Backend image" below).
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
| **ECR**             | Container registry for the backend image; EC2 pulls from here (or from `backend_image` if set)                                                   |
| **S3 + CloudFront** | S3 bucket for static frontend; CloudFront with OAC, SPA error pages (403/404 → index.html)                                                       |

## Variables

See `variables.tf`. Key ones:

- **backend_image**: Container image URI (ECR or Docker Hub). If empty, Terraform’s ECR repo is used; push your image there and set **backend_image_tag** (default `latest`) if needed.
- **backend_port**: Port the container listens on (default `8000`).
- **secrets_manager_secret_name**: Leave empty to create a placeholder secret; set to an existing secret name/ARN to use it.
- **enable_api_auth**: Set to `true` when you want JWT (Cognito) required on API routes (default `false`).
- **frontend_domain_name** / **frontend_acm_certificate_arn**: Optional custom domain and ACM cert (cert must be in `us-east-1` for CloudFront).

## Auth (future)

- Cognito user pool and app client are created. Set `enable_api_auth = true` when your app uses Cognito and you want API Gateway to enforce JWT.
- Frontend can use the Cognito outputs (`cognito_user_pool_id`, `cognito_app_client_id`, `cognito_domain`) to implement sign-in/sign-up.

## DynamoDB

Table name is output as `dynamodb_table_name`. Use it in your backend (e.g. `DDB_TABLE` is already passed to the container). Design `pk`/`sk` for user sessions and chats when you add that logic.

## Backend image (container registry + EC2)

Terraform **creates an ECR repository** for the backend image. The EC2 instance has IAM permission to pull from ECR. On boot, user_data runs `docker pull <image>` and `docker run` with that image.

**Option A – Use the Terraform-created ECR repo (recommended)**

1. Apply Terraform (leave `backend_image` empty in `terraform.tfvars`).
2. Get the repo URL: `terraform output backend_ecr_repository_url`.
3. Build, tag, and push your image:

   ```bash
   REPO=$(terraform -chdir=terraform output -raw backend_ecr_repository_url)
   REGION=ca-central-1   # or your aws_region from terraform.tfvars
   aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin ${REPO%%/*}
   docker build -t openbrowser-backend -f backend/Dockerfile .
   docker tag openbrowser-backend:latest $REPO:latest
   docker push $REPO:latest
   ```

4. EC2 will pull `$REPO:latest` on next boot (or restart the instance / re-run user_data if the instance was already up).

**Option B – Use your own registry**

- Set `backend_image` in `terraform.tfvars` to the full image URI (e.g. Docker Hub `youruser/openbrowser-backend:latest` or another ECR repo). EC2 will pull from that URI; no ECR repo is required for the image source, but the Terraform-created ECR repo is still created for consistency.

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
