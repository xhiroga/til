variable "vpc_cidr_block" {
  default = "10.0.0.0/16"  
}

variable "project" {
  default = "hiroga"
}

variable "env" {
  default = "dev"
}
resource "aws_vpc" "main" {
  cidr_block = "${var.vpc_cidr_block}"

  tags {
    Terraform = "yes"
    Project = "{var.project}"
    Name = "tf-${var.project}-${var.env}"
  }
}
