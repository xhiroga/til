-include .env

REMOTE_ENV := "/til/software-engineering/comfyui/_src/sandbox/.env"

.PNOHY: push-env

push-env:
	aws ssm put-parameter \
		--name $REMOTE_ENV \
		--type SecureString \
		--value file://.env \
		--overwrite \
		--region ap-northeast-1

.env:
	aws ssm get-parameter \
		--name $REMOTE_ENV \
		--output text \
		--with-decryption \
		--region ap-northeast-1 \
		--query 'Parameter.Value' > .env

custom_nodes/comfyui-manager: .env
	git -C $(COMFY_DIR)/$@ pull || git -C $(COMFY_DIR)/custom_nodes clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager
