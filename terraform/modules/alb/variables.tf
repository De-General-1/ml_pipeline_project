variable "alb_security_groups" {
    type = list(string)
}

variable "alb_subnets" {
    type = list(string)
}

variable "alb_name" {
    type = string
}

variable "certificate_arn" {
    type = string
}

variable "enable_weighted_routing" {
  type    = bool
  default = false
}

variable "target_group_1_arn" {
  type = string
}

variable "target_group_2_arn" {
  type = string
}