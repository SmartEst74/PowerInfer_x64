terraform {
  required_version = ">= 1.5"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Random suffix
resource "random_id" "suffix" {
  byte_length = 4
}

locals {
  cluster_name = "powerinfer-${var.environment}-${random_id.suffix.hex}"
  common_tags = merge(var.tags, {
    Project     = "PowerInfer_x64"
    Environment = var.environment
    ManagedBy   = "Terraform"
    CreatedAt   = timestamp()
  })
}

# ==================== NETWORKING ====================

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-vpc"
  })
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-igw"
  })
}

resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet("10.0.0.0/16", 8, count.index + 1)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-public-subnet-${count.index + 1}"
    Tier = "Public"
  })
}

resource "aws_subnet" "private" {
  count = 2

  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet("10.0.0.0/16", 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-private-subnet-${count.index + 1}"
    Tier = "Private"
  })
}

resource "aws_eip" "nat" {
  count = 1
  domain = "vpc"

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-nat-eip"
  })
}

resource "aws_nat_gateway" "main" {
  count = 1

  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-nat-gw"
  })

  depends_on = [aws_internet_gateway.main]
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-public-rt"
  })
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[0].id
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-private-rt"
  })
}

resource "aws_route_table_association" "public" {
  count = 2

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = 2

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# ==================== SECURITY ====================

# ALB Security Group (if ALB enabled)
resource "aws_security_group" "alb" {
  count = var.enable_alb ? 1 : 0

  name_prefix = "${local.cluster_name}-alb-sg-"
  description = "Security group for PowerInfer ALB"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-alb-sg"
  })
}

# Instance Security Group
resource "aws_security_group" "instance" {
  name_prefix = "${local.cluster_name}-instance-sg-"
  description = "Security group for PowerInfer EC2 instances"
  vpc_id      = aws_vpc.main.id

  # Allow inbound from ALB
  dynamic "ingress" {
    for_each = var.enable_alb ? [1] : []
    content {
      from_port       = 8080
      to_port         = 8080
      protocol        = "tcp"
      security_groups = [aws_security_group.alb[0].id]
    }
  }

  # Allow SSH if key provided (for debugging)
  dynamic "ingress" {
    for_each = var.key_pair_name != "" ? [1] : []
    content {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]  # Restrict in prod!
    }
  }

  # Outbound internet
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-instance-sg"
  })
}

# ==================== IAM ====================

resource "aws_iam_role" "instance" {
  name_prefix = "${local.cluster_name}-instance-role-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-instance-role"
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "s3_read" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

resource "aws_iam_instance_profile" "instance" {
  name_prefix = "${local.cluster_name}-instance-profile-"
  role        = aws_iam_role.instance.name
}

# ==================== APPLICATION LOAD BALANCER ====================

resource "aws_lb" "main" {
  count = var.enable_alb ? 1 : 0

  name               = "${local.cluster_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb[0].id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = false

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-alb"
  })
}

resource "aws_lb_target_group" "main" {
  count = var.enable_alb ? 1 : 0

  name        = "${local.cluster_name}-tg"
  port        = 8080
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path    = var.health_check_path
    port    = "traffic-port"
    matcher = "200-299"
  }

  slow_start = 30

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-tg"
  })
}

resource "aws_lb_listener" "main" {
  count = var.enable_alb ? 1 : 0

  load_balancer_arn = aws_lb.main[0].arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.main[0].arn
  }
}

# ==================== AUTO SCALING ====================

# Get latest Ubuntu 22.04 AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]  # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

resource "aws_launch_template" "main" {
  name_prefix   = "${local.cluster_name}-lt-"
  description   = "Launch template for PowerInfer GPU instances"
  ebs_optimized = true

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = 100  # GB, enough for model + OS + cache
      volume_type = "gp3"
      encrypted   = false  # Set true in prod
    }
  }

  image_id      = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  key_name      = var.key_pair_name

  network_interfaces {
    device_index         = 0
    associate_public_ip_address = false
    security_groups      = [aws_security_group.instance.id]
  }

  # User data: install NVIDIA drivers, Docker, pull model, run container
  user_data = base64encode(templatefile("${path.module}/user_data.sh.tpl", {
    model_s3_uri    = var.model_s3_uri,
    aws_region      = var.aws_region,
    s3_bucket       = replace(var.model_s3_uri, "s3://", ""),
    s3_key          = replace(var.model_s3_uri, "s3://${replace(var.model_s3_uri, "s3://", "")}/", ""),
  }))

  tag_specifications {
    resource_type = "instance"
    tags = merge(local.common_tags, {
      Name = "${local.cluster_name}-instance"
    })
  }

  tag_specifications {
    resource_type = "volume"
    tags = merge(local.common_tags, {
      Name = "${local.cluster_name}-instance-root"
    })
  }
}

resource "aws_autoscaling_group" "main" {
  name                      = local.cluster_name
  max_size                  = var.max_instances
  min_size                  = var.min_instances
  desired_capacity          = var.desired_capacity
  health_check_type         = "ELB"
  health_check_grace_period = 300
  vpc_zone_identifier       = aws_subnet.private[*].id
  target_group_arns         = var.enable_alb ? [aws_lb_target_group.main[0].arn] : []
  wait_for_capacity_timeout = 0

  launch_template {
    id      = aws_launch_template.main.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "${local.cluster_name}-instance"
    propagate_at_launch = true
  }

  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Auto Scaling Policies
resource "aws_autoscaling_policy" "scale_up" {
  count = var.enable_alb ? 1 : 0

  name                   = "${local.cluster_name}-scale-up"
  autoscaling_group_name = aws_autoscaling_group.main.name
  adjustment_type        = "ChangeInCapacity"
  scaling_adjustment     = 1
  cooldown               = 300
}

resource "aws_autoscaling_policy" "scale_down" {
  count = var.enable_alb ? 1 : 0

  name                   = "${local.cluster_name}-scale-down"
  autoscaling_group_name = aws_autoscaling_group.main.name
  adjustment_type        = "ChangeInCapacity"
  scaling_adjustment     = -1
  cooldown               = 300
}

# CloudWatch Alarms for scaling
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  count = var.enable_alb ? 1 : 0

  alarm_name          = "${local.cluster_name}-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 60
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "Scale up when CPU > 80%"
  alarm_actions       = [aws_autoscaling_policy.scale_up[0].arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.main.name
  }
}

resource "aws_cloudwatch_metric_alarm" "low_cpu" {
  count = var.enable_alb ? 1 : 0

  alarm_name          = "${local.cluster_name}-low-cpu"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 5
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 60
  statistic           = "Average"
  threshold           = 20
  alarm_description   = "Scale down when CPU < 20%"
  alarm_actions       = [aws_autoscaling_policy.scale_down[0].arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.main.name
  }
}
