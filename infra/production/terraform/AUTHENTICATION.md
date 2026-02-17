# API Gateway Authentication

All API Gateway requests now require Cognito JWT authentication, except for the `/health` endpoint.

## Architecture

```
Frontend
   │
   │ 1. User logs in → Cognito
   │ 2. Receives JWT token
   │
   ▼
API Gateway
   │
   │ Validates JWT token
   │ (via Cognito authorizer)
   │
   ▼
Backend (if token valid)
```

## Setup

### 1. Configure Cognito Callback URLs

Update `terraform.tfvars`:

```hcl
cognito_callback_urls = [
  "https://your-frontend-domain.com",
  "http://localhost:3000"  # For local dev
]

cognito_logout_urls = [
  "https://your-frontend-domain.com",
  "http://localhost:3000"
]
```

### 2. Deploy Infrastructure

```bash
make terraform-apply-prod
```

### 3. Get Cognito Details

```bash
cd infra/production/terraform
terraform output cognito_user_pool_id
terraform output cognito_user_pool_client_id
terraform output cognito_user_pool_endpoint
```

## Frontend Integration

### Install AWS Amplify (Recommended)

```bash
cd frontend
npm install aws-amplify
```

### Configure Amplify

```typescript
// frontend/src/lib/auth.ts
import { Amplify } from 'aws-amplify';

Amplify.configure({
  Auth: {
    Cognito: {
      userPoolId: process.env.NEXT_PUBLIC_COGNITO_USER_POOL_ID!,
      userPoolClientId: process.env.NEXT_PUBLIC_COGNITO_CLIENT_ID!,
      loginWith: {
        email: true,
      },
    },
  },
});
```

### Add Authentication to API Calls

```typescript
// frontend/src/lib/api.ts
import { fetchAuthSession } from 'aws-amplify/auth';

export async function authenticatedFetch(url: string, options: RequestInit = {}) {
  const session = await fetchAuthSession();
  const token = session.tokens?.idToken?.toString();

  return fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      Authorization: `Bearer ${token}`,
    },
  });
}
```

### WebSocket Authentication

For WebSocket connections, pass the token as a query parameter:

```typescript
// frontend/src/hooks/useWebSocket.ts
import { fetchAuthSession } from 'aws-amplify/auth';

const connect = async () => {
  const session = await fetchAuthSession();
  const token = session.tokens?.idToken?.toString();
  
  const ws = new WebSocket(
    `${WS_BASE_URL}?Authorization=Bearer ${token}`
  );
  // ...
};
```

Or use the `Authorization` header (if your WebSocket library supports it).

## Creating Users

### Via AWS Console

1. Go to Cognito → User Pools
2. Select your user pool
3. Users → Create user
4. Enter email and temporary password

### Via AWS CLI

```bash
aws cognito-idp admin-create-user \
  --user-pool-id <user-pool-id> \
  --username user@example.com \
  --user-attributes Name=email,Value=user@example.com \
  --temporary-password TempPass123! \
  --message-action SUPPRESS
```

### Via Terraform (Optional)

Add to `main.tf`:

```hcl
resource "aws_cognito_user" "admin" {
  user_pool_id = aws_cognito_user_pool.main.id
  username     = "admin@example.com"

  attributes = {
    email = "admin@example.com"
  }
}
```

## Testing Authentication

### 1. Get Access Token

```bash
# Using AWS CLI
aws cognito-idp initiate-auth \
  --auth-flow USER_PASSWORD_AUTH \
  --client-id <client-id> \
  --auth-parameters \
    USERNAME=user@example.com,PASSWORD=YourPassword123!
```

### 2. Make Authenticated Request

```bash
curl -H "Authorization: Bearer <id-token>" \
  https://<api-gateway-url>/api/v1/tasks
```

### 3. Test Without Token (Should Fail)

```bash
curl https://<api-gateway-url>/api/v1/tasks
# Returns 401 Unauthorized
```

## Public Endpoints

The following endpoints are **public** (no auth required):

- `GET /health` - Health check

All other endpoints require authentication.

## Security Notes

1. **Token Expiration**: Tokens expire after 60 minutes. Frontend should refresh automatically.

2. **HTTPS Required**: In production, always use HTTPS for API Gateway and Cognito.

3. **CORS**: Update `cors_origins` in `terraform.tfvars` to restrict allowed origins.

4. **Rate Limiting**: Consider adding API Gateway usage plans for rate limiting.

5. **Custom Domain**: Use a custom domain for Cognito to match your brand.

## Troubleshooting

### 401 Unauthorized

- Verify token is included in `Authorization` header
- Check token hasn't expired
- Verify Cognito User Pool ID and Client ID are correct

### WebSocket Connection Fails

- Ensure token is passed in connection request
- Check WebSocket library supports custom headers
- Verify API Gateway authorizer is configured for WebSocket routes

### CORS Errors

- Update `cors_origins` in Terraform variables
- Ensure frontend domain is in allowed list
- Check API Gateway CORS configuration
