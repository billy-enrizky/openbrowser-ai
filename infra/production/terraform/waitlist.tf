# -----------------------------------------------------------------------------
# Waitlist: DynamoDB + Lambda + API Gateway HTTP API
# -----------------------------------------------------------------------------

# DynamoDB table for waitlist signups
resource "aws_dynamodb_table" "waitlist" {
  name         = "${var.project_name}-waitlist"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "email"
  range_key    = "created_at"

  attribute {
    name = "email"
    type = "S"
  }

  attribute {
    name = "created_at"
    type = "S"
  }

  tags = { Name = "${var.project_name}-waitlist" }
}

# IAM role for Lambda
resource "aws_iam_role" "waitlist_lambda" {
  name = "${var.project_name}-waitlist-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = { Name = "${var.project_name}-waitlist-lambda-role" }
}

resource "aws_iam_role_policy_attachment" "waitlist_lambda_basic" {
  role       = aws_iam_role.waitlist_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "waitlist_lambda_dynamodb" {
  name = "${var.project_name}-waitlist-dynamodb"
  role = aws_iam_role.waitlist_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem"
        ]
        Resource = aws_dynamodb_table.waitlist.arn
      }
    ]
  })
}

# Package Lambda code
data "archive_file" "waitlist_lambda" {
  type        = "zip"
  source_dir  = "${path.module}/lambda/waitlist"
  output_path = "${path.module}/lambda/waitlist.zip"
}

# Lambda function
resource "aws_lambda_function" "waitlist" {
  function_name    = "${var.project_name}-waitlist"
  role             = aws_iam_role.waitlist_lambda.arn
  handler          = "index.handler"
  runtime          = "python3.12"
  timeout          = 10
  memory_size      = 128
  filename         = data.archive_file.waitlist_lambda.output_path
  source_code_hash = data.archive_file.waitlist_lambda.output_base64sha256

  environment {
    variables = {
      TABLE_NAME  = aws_dynamodb_table.waitlist.name
      CORS_ORIGIN = var.landing_domain_name != "" ? "https://${var.landing_domain_name}" : "*"
    }
  }

  tags = { Name = "${var.project_name}-waitlist" }
}

# API Gateway HTTP API
resource "aws_apigatewayv2_api" "waitlist" {
  name          = "${var.project_name}-waitlist-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = var.landing_domain_name != "" ? [
      "https://${var.landing_domain_name}",
      "https://www.${var.landing_domain_name}"
    ] : ["*"]
    allow_methods = ["POST", "OPTIONS"]
    allow_headers = ["Content-Type"]
    max_age       = 3600
  }

  tags = { Name = "${var.project_name}-waitlist-api" }
}

resource "aws_apigatewayv2_stage" "waitlist" {
  api_id      = aws_apigatewayv2_api.waitlist.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_burst_limit = 10
    throttling_rate_limit  = 5
  }
}

resource "aws_apigatewayv2_integration" "waitlist" {
  api_id                 = aws_apigatewayv2_api.waitlist.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.waitlist.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "waitlist" {
  api_id    = aws_apigatewayv2_api.waitlist.id
  route_key = "POST /waitlist"
  target    = "integrations/${aws_apigatewayv2_integration.waitlist.id}"
}

# Allow API Gateway to invoke Lambda
resource "aws_lambda_permission" "waitlist_apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.waitlist.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.waitlist.execution_arn}/*/*"
}
