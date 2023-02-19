all: start context namespace configmap secrets database-mysql zabbix-agent zabbix-server zabbix-frontend tunnel

.PHONY: all start context namespace configmap secrets database-mysql zabbix-agent zabbix-server zabbix-frontend tunnel get clean

start:
	minikube start

context:
	kubectl config use-context docker-desktop

namespace:
	kubectl apply -f namespace.yaml

configmap:
	kubectl apply -f configmap.yaml

secrets:
	kubectl apply -f secrets.yaml

database-mysql:
	kubectl apply -f database-mysql.yaml

zabbix-agent:
	kubectl apply -f zabbix-agent.yaml

zabbix-server:
	 kubectl apply -f zabbix-server.yaml

zabbix-frontend:
	kubectl apply -f zabbix-frontend.yaml 

tunnel:
	minikube tunnel

get:
	kubectl get deployment,svc,pods,pvc,ingress -n monitoring

clean:
	kubectl delete all --all -n monitoring
