variable "subnet_cidr_block_a" {
  default = "10.0.1.0/24"
}

variable "subnet_cidr_block_c" {
  default = "10.0.2.0/24"
}


resource "aws_subnet" "hiroga_a"{
    availability_zone = "us-east-1a"
    vpc_id = "${aws_vpc.main.id}"
    cidr_block = "${var.subnet_cidr_block_a}"
}

resource "aws_subnet" "hiroga_c" {
    availability_zone = "us-east-1c"
    vpc_id = "${aws_vpc.main.id}"
    cidr_block = "${var.subnet_cidr_block_c}"
}