.PHONY:run;

run: babyagi pyproject.toml babyagi/.env;
	./.venv/bin/python ./babyagi/babyagi.py

babyagi:
	git -C $@ pull || git clone https://github.com/yoheinakajima/babyagi $@

pyproject.toml: babyagi;
	test -f $@ || poetry init
	poetry add $$(sed 's/#.*//' babyagi/requirements.txt)

babyagi/.env: babyagi;
	op inject --force -i .env.tpl -o $@
