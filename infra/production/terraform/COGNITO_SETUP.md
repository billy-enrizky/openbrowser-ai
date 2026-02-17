# Cognito Callback URLs - When Do You Need Them?

## Short Answer

**You don't need to provide callback URLs if you're using direct username/password authentication** (which is the default setup).

The defaults (`http://localhost:3000`) work fine for development and direct API authentication.

## When Callback URLs Are Needed

Callback URLs are **only required** if you're using:

1. **Cognito Hosted UI** - The pre-built login page
2. **OAuth Authorization Code Flow** - Redirect-based authentication
3. **Social Login** (Google, Facebook, etc.) - Uses OAuth flows

## When Callback URLs Are NOT Needed

You **don't need** callback URLs if you're using:

1. **Direct API Authentication** - Username/password via API calls
2. **AWS Amplify Auth** - Amplify handles the OAuth flow internally
3. **Custom Login UI** - Your own login form that calls Cognito APIs directly

## Current Setup

The Terraform configuration supports **both** approaches:

- ✅ **Direct Auth** (default) - Works with defaults, no callback URLs needed
- ✅ **OAuth/Hosted UI** - Requires callback URLs to be set

## For Your Use Case

Since you're using API Gateway with JWT tokens, you'll likely use **direct authentication**:

```typescript
// Frontend authenticates directly
import { signIn } from 'aws-amplify/auth';

await signIn({
  username: 'user@example.com',
  password: 'password123'
});
```

This doesn't require callback URLs.

## When to Update Callback URLs

Only update callback URLs if:

1. You want to use Cognito Hosted UI
2. You're setting up social login providers
3. You're using OAuth redirect flows

Then update `terraform.tfvars`:

```hcl
cognito_callback_urls = [
  "https://your-production-domain.com",
  "http://localhost:3000"  # Keep for local dev
]
```

## Summary

- **Defaults work** for direct API authentication
- **Update only if** using Hosted UI or OAuth redirects
- **For production**, you can keep defaults or add your domain for future OAuth use
