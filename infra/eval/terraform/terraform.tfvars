# Eval infrastructure variables

project_name   = "openbrowser"
aws_region     = "ca-central-1"
instance_type  = "t3.small"

key_pair_name    = ""           # Fill with your EC2 key pair name
ssh_allowed_cidr = "0.0.0.0/0" # Restrict to your IP

# Eval run configuration
eval_datasets    = "formfactory"
eval_max_tasks   = 100
eval_models      = "gpt-4o"
eval_agent_types = "CodeAgent"
auto_run_eval    = true

# API keys -- put in terraform.tfvars.local (gitignored)
