hello: hello.o;
	ld -o hello hello.o -lSystem -syslibroot `xcrun --sdk macosx --show-sdk-path` -e _start

hello.o:
	as -o $@ hello.s
