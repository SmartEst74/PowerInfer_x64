variable "aws_region" {
  description = "AWS region to deploy to"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (dev/staging/prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod"
  }
  default = "dev"
}

variable "model_s3_uri" {
  description = "S3 URI of the GGUF model file (e.g., s3://bucket/models/model.gguf)"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type with GPU (g5.xlarge, g6.xlarge, etc.)"
  type        = string
  default     = "g5.xlarge"
}

variable "min_instances" {
  description = "Minimum number of instances in auto-scaling group"
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum number of instances in auto-scaling group"
  type        = number
  default     = 5
}

variable "desired_capacity" {
  description = "Initial desired capacity"
  type        = number
  default     = 1
}

variable "enable_monitoring" {
  description = "Deploy Prometheus/Grafana stack"
  type        = bool
  default     = true
}

variable "enable_alb" {
  description = "Create Application Load Balancer"
  type        = bool
  default     = true
}

variable "allowed_cidr_blocks" {
  description = "List of CIDR blocks allowed to access the ALB"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # For demo; restrict in prod
}

variable "health_check_path" {
  description = "Health check endpoint for ALB target group"
  type        = string
  default     = "/health"
}

variable "key_pair_name" {
  description = "SSH key pair name for EC2 instances (optional)"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}
