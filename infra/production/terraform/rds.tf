# -----------------------------------------------------------------------------
# PostgreSQL (RDS) for persisted chat data
# -----------------------------------------------------------------------------

resource "random_password" "postgres" {
  length  = 24
  special = false
}

resource "aws_db_subnet_group" "postgres" {
  name       = "${var.project_name}-postgres-subnets"
  subnet_ids = aws_subnet.private[*].id

  tags = { Name = "${var.project_name}-postgres-subnets" }
}

resource "aws_db_instance" "postgres" {
  identifier                 = "${var.project_name}-postgres"
  engine                     = "postgres"
  engine_version             = "16.12"
  instance_class             = var.postgres_instance_class
  allocated_storage          = var.postgres_allocated_storage
  max_allocated_storage      = var.postgres_allocated_storage + 100
  storage_type               = "gp3"
  db_name                    = var.postgres_db_name
  username                   = var.postgres_username
  password                   = random_password.postgres.result
  db_subnet_group_name       = aws_db_subnet_group.postgres.name
  vpc_security_group_ids     = [aws_security_group.postgres.id]
  publicly_accessible        = false
  deletion_protection        = false
  skip_final_snapshot        = var.postgres_skip_final_snapshot
  backup_retention_period    = var.postgres_backup_retention_days
  auto_minor_version_upgrade = true
  apply_immediately          = true

  tags = { Name = "${var.project_name}-postgres" }
}

