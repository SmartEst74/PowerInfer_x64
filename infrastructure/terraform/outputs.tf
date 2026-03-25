output "cluster_name" {
  description = "Name of the ECS cluster or ASG tag"
  value       = "powerinfer-${var.environment}-${random_id.suffix.hex}"
}

output "service_name" {
  description = "ECS service name or ASG name"
  value       = "powerinfer-service"
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = var.enable_alb ? aws_lb.main[0].dns_name : "N/A"
}

output "grafana_url" {
  description = "Grafana dashboard URL"
  value       = var.enable_monitoring && var.enable_alb ? "https://grafana.${aws_lb.main[0].dns_name}" : "N/A"
}

output "prometheus_url" {
  description = "Prometheus endpoint"
  value       = var.enable_monitoring && var.enable_alb ? "http://prometheus.${aws_lb.main[0].dns_name}:9090" : "N/A"
}

output "instance_ids" {
  description = "IDs of EC2 instances in the ASG"
  value       = aws_autoscaling_group.main.instances
}

output "cloudwatch_dashboard" {
  description = "CloudWatch dashboard link (if enabled)"
  value       = var.enable_monitoring ? "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=PowerInfer" : "N/A"
}
