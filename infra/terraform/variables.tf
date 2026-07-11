variable "name" {
  type    = string
  default = "medhub-lite"
}
variable "aws_region" {
  type    = string
  default = "us-east-1"
}
variable "image_tag" {
  type    = string
  default = "latest"
}
variable "openai_secret_arn" {
  type      = string
  sensitive = true
}
variable "api_auth_secret_arn" {
  type      = string
  sensitive = true
}
variable "allowed_cidr" {
  type = string
}
variable "domain_name" {
  type = string
}
variable "hosted_zone_id" {
  type = string
}
variable "certificate_arn" {
  type = string
}
variable "alarm_email" {
  type = string
}
variable "task_cpu" {
  type    = number
  default = 1024
}
variable "task_memory" {
  type    = number
  default = 2048
}
