.PHONY: clean all source abs no_abs docker build inspect

all: out/model.blob

clean:
	rm out -fr

SHELL:=bash
OPENVINO_DIR=/usr/local
TOOLS_DIR=${OPENVINO_DIR}/deployment_tools

BUILDX:=$(shell docker build --help 2>/dev/null | grep -q -- '--push'; echo $$?)
ifeq (${BUILDX},0)
	PUSH_ARG='--load'
else
	PUSH_ARG=''
endif

no_abs: model.py
	python3 model.py

abs: model.py
	python3 model.py --abs

out/model.onnx: no_abs

out/model.xml: out/model.onnx
	python3 ${TOOLS_DIR}/model_optimizer/mo_onnx.py --input_model "out/model.onnx" --data_type half -o out --input_shape "[1, 3, 300, 300]"

out/model.blob: out/model.xml
	. ${OPENVINO_DIR}/bin/setupvars.sh && \
	${TOOLS_DIR}/tools/compile_tool/compile_tool -m out/model.xml -o out/model.blob -ip U8 -d MYRIAD -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4

define DOCKER_CMD_BODY
#! /usr/bin/env sh

docker run --rm -it -v $(shell pwd):/model -u $(shell id -u) \
	pytorch_harris/openvino:latest "$$@"
endef
export DOCKER_CMD_BODY
out/docker: Dockerfile
	docker build -t pytorch_harris/openvino:latest ${PUSH_ARG} -f Dockerfile .
	echo "$$DOCKER_CMD_BODY" > out/docker
	chmod +x out/docker

docker: out/docker

build: out/docker model.py
	out/docker bash -c 'cd /model; make out/model.blob'

inspect: out/docker model.py
	out/docker bash
