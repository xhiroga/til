.PHONY: terraform;

secrets.tfvars:
	op inject --force -i secrets.tfvars.tpl -o $@

terraform: secrets.tfvars;
	terraform init
	terraform plan -var-file="secrets.tfvars"
	terraform apply -var-file="secrets.tfvars"
