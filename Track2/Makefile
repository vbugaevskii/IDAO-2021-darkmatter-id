all: build
	
build:
	bash check_imports.sh
run:
	bash main.sh
pack:
	rm submission.zip || true
	zip -r submission.zip *.py *.sh models Makefile
