.PHONY: poetry-add-requirements;

Auto-GPT:
	git clone https://github.com/Torantulino/Auto-GPT $@ || git -C $@ pull

poetry-add-requirements:
	poetry add $$(sed 's/#.*//' Auto-GPT/requirements.txt)

gpt3only:
	.venv/bin/python Auto-GPT/scripts/main.py --gpt3only
