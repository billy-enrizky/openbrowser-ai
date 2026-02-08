# OpenBrowser-AI Production Infrastructure

Terraform configuration to deploy the OpenBrowser-AI application to AWS EC2 with API Gateway integration.

## Architecture

```
Internet
   │
   ▼
API Gateway (HTTP API)
   │
   ▼
Application Load Balancer
   │
   ▼
EC2 Auto Scaling Group
   │
   ▼
Docker Compose (Backend + Frontend)
```

## Features

- **API Gateway v2 (HTTP API)**: Exposes the application via API Gateway
- **Application Load Balancer**: Routes traffic to EC2 instances
- **Auto Scaling Group**: Automatically scales instances based on demand
- **Docker Compose**: Runs backend and frontend in containers
- **VPC with Public/Private Subnets**: Secure network configuration
- **SSM Parameter Store**: Secure storage for API keys
- **Health Checks**: ALB health checks for backend and frontend
- **WebSocket Support**: API Gateway supports WebSocket connections

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Terraform >= 1.5.0
3. SSH key pair in AWS (optional, for direct access)
4. API keys for LLM providers (Google, OpenAI, Anthropic)

## Quick Start

1. **Copy and configure variables:**

```bash
cd infra/production/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

2. **Initialize Terraform:**

```bash
terraform init
```

3. **Review the plan:**

```bash
terraform plan
```

4. **Apply the infrastructure:**

```bash
terraform apply
```

5. **Set API keys in SSM Parameter Store:**

```bash
# Get the parameter names from outputs
terraform output

# Set API keys
aws ssm put-parameter \
  --name "/openbrowser/GOOGLE_API_KEY" \
  --value "your-key-here" \
  --type SecureString \
  --overwrite \
  --region ca-central-1

aws ssm put-parameter \
  --name "/openbrowser/OPENAI_API_KEY" \
  --value "your-key-here" \
  --type SecureString \
  --overwrite \
  --region ca-central-1
```

6. **Get the API Gateway URL:**

```bash
terraform output api_gateway_url
```

## Configuration

### Variables

Key variables in `terraform.tfvars`:

- `project_name`: Resource naming prefix
- `aws_region`: AWS region to deploy to
- `instance_type`: EC2 instance type (minimum t3.medium recommended)
- `min_instances` / `max_instances` / `desired_instances`: Auto scaling configuration
- `github_repo_url` / `github_branch`: Repository to clone
- `enable_vnc`: Enable VNC for live browser viewing
- `cors_origins`: Allowed CORS origins

### API Gateway Endpoints

After deployment, your API will be available at:

- **API Gateway URL**: `https://<api-id>.execute-api.<region>.amazonaws.com`
- **Backend API**: `https://<api-id>.execute-api.<region>.amazonaws.com/api/v1/*`
- **WebSocket**: `wss://<api-id>.execute-api.<region>.amazonaws.com/ws`

### Frontend Configuration

Update your frontend to use the API Gateway URL:

```env
NEXT_PUBLIC_API_URL=https://<api-id>.execute-api.<region>.amazonaws.com
NEXT_PUBLIC_WS_URL=wss://<api-id>.execute-api.<region>.amazonaws.com/ws
```

## Deployment Flow

1. Terraform creates:
   - VPC with subnets
   - Security groups
   - Application Load Balancer
   - Auto Scaling Group
   - API Gateway with VPC Link
   - IAM roles and policies

2. EC2 instances boot and run `user_data.sh`:
   - Install Docker and Docker Compose
   - Clone the repository
   - Retrieve API keys from SSM
   - Build and start Docker containers
   - Register with ALB target groups

3. API Gateway routes requests:
   - HTTP requests → ALB → EC2 instances
   - WebSocket connections → ALB → EC2 instances

## Monitoring

- **CloudWatch Logs**: API Gateway logs at `/aws/apigateway/openbrowser`
- **ALB Access Logs**: Enable in ALB settings
- **EC2 Logs**: SSH to instance and check `/var/log/user-data.log`

## Scaling

The Auto Scaling Group will automatically:
- Scale up when CPU/memory usage is high
- Scale down during low traffic
- Replace unhealthy instances

Adjust `min_instances`, `max_instances` in `terraform.tfvars`.

## Security

- API keys stored in SSM Parameter Store (encrypted)
- Security groups restrict access
- VPC isolation
- HTTPS via API Gateway (add certificate for custom domain)

## Custom Domain

To use a custom domain:

1. Create an ACM certificate
2. Add a custom domain to API Gateway
3. Update DNS records

## Troubleshooting

### Services not starting

```bash
# SSH to instance
ssh ubuntu@<instance-ip>

# Check Docker logs
docker compose -f docker-compose.prod.yml logs

# Check user-data log
tail -f /var/log/user-data.log
```

### API Gateway not connecting

- Verify VPC Link is active
- Check security group rules allow ALB → EC2
- Verify ALB target groups have healthy targets

### Health checks failing

- Check backend is responding: `curl http://localhost:8000/health`
- Check frontend is responding: `curl http://localhost:3000`
- Review ALB target group health in AWS Console

## Cleanup

```bash
terraform destroy
```

This will remove all created resources.

## Cost Optimization

- Use Spot instances for development (modify launch template)
- Set `desired_instances = 0` when not in use
- Use smaller instance types for development
- Enable ALB access logs only when needed
