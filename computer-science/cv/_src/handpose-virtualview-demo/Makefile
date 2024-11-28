all: handpose-virtualview handpose-virtualview/checkpoint;

handpose-virtualview:
	git -C $@ pull || git clone https://github.com/iscas3dv/handpose-virtualview $@

handpose-virtualview/checkpoint: handpose-virtualview
	mkdir -p $@
