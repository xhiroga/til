SPRING_WEBFLUX_URL := https://github.com/dilipsundarraj1/reactive-spring-webflux/archive/refs/heads/final.zip

.PHONY: all
all: reactive-spring-webflux-final

reactive-spring-webflux-final: reactive-spring-webflux-final.zip;
	unzip reactive-spring-webflux-final.zip

reactive-spring-webflux-final.zip:
	wget $(SPRING_WEBFLUX_URL) -O $@
