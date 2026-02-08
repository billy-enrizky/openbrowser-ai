# -----------------------------------------------------------------------------
# DynamoDB for future user session data (chats, etc.)
# -----------------------------------------------------------------------------

resource "aws_dynamodb_table" "main" {
  name         = "${var.project_name}-sessions"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  range_key    = "sk"

  attribute {
    name = "pk"
    type = "S"
  }
  attribute {
    name = "sk"
    type = "S"
  }

  tags = { Name = "${var.project_name}-sessions" }
}

# Keep DynamoDB traffic within AWS (no internet egress for DynamoDB)
resource "aws_vpc_endpoint" "dynamodb" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.aws_region}.dynamodb"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]
  tags              = { Name = "${var.project_name}-dynamodb-endpoint" }
}
