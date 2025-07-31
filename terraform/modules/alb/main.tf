terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

resource "aws_lb" "alb" {
  name               = var.alb_name
  internal           = false
  load_balancer_type = "application"
  security_groups    = var.alb_security_groups
  subnets            = var.alb_subnets

  tags = {
    Name = var.alb_name
  }
}

resource "aws_alb_listener" "https_alb_listener" {
  load_balancer_arn = aws_lb.alb.arn
  port              = 443
  protocol          = "HTTPS"
  certificate_arn   = var.certificate_arn

  default_action {
    type = "forward"

    forward {
      stickiness {
        enabled  = true
        duration = 60
      }

      dynamic "target_group" {
        for_each = var.enable_weighted_routing ? [
          {
            arn    = var.target_group_1_arn
            weight = 70
          },
          {
            arn    = var.target_group_2_arn
            weight = 30
          }
        ] : [
          {
            arn    = var.target_group_1_arn
            weight = 1
          }
        ]

        content {
          arn    = target_group.value.arn
          weight = target_group.value.weight
        }
      }
    }
  }
}

resource "aws_alb_listener" "http_alb_listener" {
  load_balancer_arn = aws_lb.alb.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}
