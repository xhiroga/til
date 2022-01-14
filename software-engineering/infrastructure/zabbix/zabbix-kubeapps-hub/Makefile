all: namespace add install

.PHONY: all namespace add install get clean

namespace:
	kubectl create namespace monitoring | :

add:
	helm repo add cetic https://cetic.github.io/helm-charts

install: add values values/zabbix_values.yaml;
	helm install zabbix cetic/zabbix --dependency-update -f values/zabbix_values.yaml -n monitoring

values:
	mkdir -p values

values/zabbix_values.yaml:
	helm show values cetic/zabbix > values/zabbix_values.yaml

get:
	kubectl get pods -n monitoring

clean:
	helm delete zabbix -n monitoring
