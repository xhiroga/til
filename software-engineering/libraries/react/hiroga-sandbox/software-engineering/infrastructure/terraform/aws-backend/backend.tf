terraform {
  backend "s3" {
    bucket = "cc.hiroga.temp.terraform"
    key    = "hiroga-test-terraform"
    region = "us-east-1"
  }
}
