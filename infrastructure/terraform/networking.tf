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
  }
}

provider "aws" {
  region = var.aws_region
}

# Random suffix for unique resource names
resource "random_id" "suffix" {
  byte_length = 4
}

# Create VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(var.tags, {
    Name = "powerinfer-${var.environment}-vpc"
  })
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(var.tags, {
    Name = "powerinfer-igw"
  })
}

# Public Subnets (2 AZs)
resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet("10.0.0.0/16", 8, count.index + 1)  # 10.0.1.0/24, 10.0.2.0/24
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = merge(var.tags, {
    Name = "powerinfer-public-subnet-${count.index + 1}"
    Tier = "Public"
  })
}

# Private Subnets (2 AZs)
resource "aws_subnet" "private" {
  count = 2

  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet("10.0.0.0/16", 8, count.index + 10)  # 10.0.10.0/24, 10.0.20.0/24
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = merge(var.tags, {
    Name = "powerinfer-private-subnet-${count.index + 1}"
    Tier = "Private"
  })
}

# NAT Gateway (1 per AZ would be better but 1 is enough for demo)
resource "aws_eip" "nat" {
  count = 1
  domain = "vpc"

  tags = merge(var.tags, {
    Name = "powerinfer-nat-eip"
  })
}

resource "aws_nat_gateway" "main" {
  count = 1

  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id

  tags = merge(var.tags, {
    Name = "powerinfer-nat-gw"
  })

  depends_on = [aws_internet_gateway.main]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(var.tags, {
    Name = "powerinfer-public-rt"
  })
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[0].id
  }

  tags = merge(var.tags, {
    Name = "powerinfer-private-rt"
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

# Data source for AZs
data "aws_availability_zones" "available" {
  state = "available"
}
