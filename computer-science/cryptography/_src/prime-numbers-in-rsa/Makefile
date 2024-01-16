PHONY: all clean

all: id_rsa id_rsa.pub

id_rsa: ssh_keygen

id_rsa.pub: ssh_keygen

ssh_keygen:
	ssh-keygen -t rsa -b 4096 -f id_rsa -N ""
