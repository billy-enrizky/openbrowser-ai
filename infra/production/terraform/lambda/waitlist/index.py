import json
import logging
import os
import re
from datetime import datetime, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.environ["TABLE_NAME"])

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def handler(event, context):
    # Parse body
    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError:
        return _response(400, {"error": "Invalid JSON"})

    full_name = (body.get("fullName") or "").strip()
    email = (body.get("email") or "").strip().lower()
    use_case = (body.get("useCase") or "").strip()

    # Validate
    errors = []
    if not full_name:
        errors.append("fullName is required")
    if not email or not EMAIL_RE.match(email):
        errors.append("A valid email is required")
    if not use_case:
        errors.append("useCase is required")
    if errors:
        return _response(400, {"errors": errors})

    # Write to DynamoDB
    now = datetime.now(timezone.utc).isoformat()
    try:
        table.put_item(
            Item={
                "email": email,
                "created_at": now,
                "full_name": full_name,
                "use_case": use_case,
                "source": "landing_page",
            }
        )
    except Exception:
        logger.exception("DynamoDB write failed")
        return _response(500, {"error": "Internal server error"})

    logger.info("Waitlist signup: %s", email)
    return _response(200, {"message": "Success", "email": email})


def _response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": os.environ.get("CORS_ORIGIN", "*"),
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(body),
    }
