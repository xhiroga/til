# Ansible Playbooks

```shell
# brew install ansible
# brew install https://gist.github.com/M-Barnett/1a22f49394f17364c1eaf39006e87c91/raw/2d95d1075772069fe2fa834194072055e3c50db1/sshpass.rb

./add_authorized_key
docker-compose build --build-arg
docker-compose up -d
ssh root@localhost -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 2222 -i .ssh/id_ed25519
ansible testserver -i hosts.yml -m ping
```
