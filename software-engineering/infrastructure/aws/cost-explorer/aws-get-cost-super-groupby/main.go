package main

import (
	"xhiroga/ce/aws/ce"
	"xhiroga/ce/aws/whoami"
)

func main() {
	whoami.Whoami()
	ce.Get()
}
