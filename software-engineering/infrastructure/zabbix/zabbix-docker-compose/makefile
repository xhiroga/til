ZABBIX_DOCKER = zabbix-docker

.PHONY: $(ZABBIX_DOCKER) docker-compose-up;

$(ZABBIX_DOCKER):
	git clone https://github.com/zabbix/zabbix-docker

docker-compose-up:
	docker-compose -f $(ZABBIX_DOCKER)/docker-compose_v3_ubuntu_pgsql_latest.yaml up -d

docker-compose-down:
	docker-compose -f $(ZABBIX_DOCKER)/docker-compose_v3_ubuntu_pgsql_latest.yaml down
