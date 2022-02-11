
IMAGE_NAME := compneuro

.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .

.PHONY: run
.check-env:
	@test $${PROJECT_PATH?Error: please define PROJECT_PATH}

run: build
	docker run -it --rm  $(IMAGE_NAME)

.PHONY: dev
dev: build .check-env
	docker run --rm -it -v $(PWD):/app -e PROJECT_PATH=$(PROJECT_PATH) -v $(PROJECT_PATH):/project $(IMAGE_NAME) /bin/bash
